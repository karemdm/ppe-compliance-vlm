"""Person re-identification across tracker ids (standard library only).

The tracker gives a new id when a person leaves and re-enters the scene, or
after a long occlusion or an id switch. :class:`ReIdentifier` maps tracker ids
to persistent *person ids* by comparing appearance embeddings (any vectors of
floats, see :mod:`ppe_compliance.tracking.appearance`) with a gallery of the persons
seen before.

A new track is *pending* for a short time while it collects embeddings. Then
the mean embedding is compared with every gallery identity that could be the
same person (an identity seen at the same time as the new track cannot be).
The best identity is reused when its cosine similarity reaches ``threshold``
and beats the second best by ``margin``; otherwise a new person id is created.

Times are in seconds and are passed in by the caller (e.g. ``frame_index / fps``).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

Vector = Sequence[float]
Box = Sequence[float]


@dataclass(frozen=True)
class ReIDConfig:
    """Parameters of the re-identification.

    Attributes:
        enabled: When False, the person id is the tracker id (no re-ID).
        embedder: Appearance embedder: ``"histogram"`` (HSV color histogram
            of horizontal body stripes, no extra model) or the path / name of
            an Ultralytics classification model (e.g. ``"yolo11n-cls.pt"``),
            whose pooled features are used.
        threshold: Minimum cosine similarity between a new track and a
            gallery identity to reuse that identity, in ``[-1, 1]``.
        margin: Minimum difference between the best and the second-best
            similarity; ambiguous matches create a new identity.
        min_samples: Embeddings a new track collects before it is matched.
        max_pending_seconds: A new track is decided after this time even with
            fewer than ``min_samples`` embeddings (a new identity when it has none).
        max_absence_seconds: Identities unseen for longer are no longer
            candidates; ``0`` keeps them for the whole video.
        gallery_size: Embeddings kept per identity (most recent).
        sample_seconds: Minimum time between two embeddings added to the
            gallery of an identity.
        min_box_height: Person boxes lower than this (pixels) give no embedding.
        max_overlap: A person box overlapping another person box by more than
            this share of its own area gives no embedding (occlusion).
    """

    enabled: bool = True
    embedder: str = "histogram"
    threshold: float = 0.85
    margin: float = 0.03
    min_samples: int = 3
    max_pending_seconds: float = 0.5
    max_absence_seconds: float = 0.0
    gallery_size: int = 30
    sample_seconds: float = 0.2
    min_box_height: int = 64
    max_overlap: float = 0.3

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a parameter is out of range.
        """
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("reid threshold must be in [-1, 1]")
        if self.margin < 0:
            raise ValueError("reid margin must be >= 0")
        if self.min_samples < 1:
            raise ValueError("reid min_samples must be >= 1")
        if self.max_pending_seconds < 0 or self.max_absence_seconds < 0:
            raise ValueError("reid times must be >= 0")
        if self.gallery_size < 1:
            raise ValueError("reid gallery_size must be >= 1")
        if self.sample_seconds < 0:
            raise ValueError("reid sample_seconds must be >= 0")
        if not 0.0 <= self.max_overlap <= 1.0:
            raise ValueError("reid max_overlap must be in [0, 1]")


def normalize(vector: Vector) -> List[float]:
    """Return the L2-normalized vector (zeros stay zeros).

    Args:
        vector: Any vector.

    Returns:
        The unit vector as a list.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    return [float(v) / norm for v in vector] if norm > 0 else [0.0 for _ in vector]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two vectors (0 when one of them is zero).

    Args:
        a: First vector.
        b: Second vector (same length).

    Returns:
        The similarity in ``[-1, 1]``.

    Raises:
        ValueError: If the lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def mean_vector(vectors: Sequence[Vector]) -> List[float]:
    """Mean of L2-normalized vectors, normalized again.

    Args:
        vectors: Non-empty list of vectors of the same length.

    Returns:
        The unit mean vector.
    """
    units = [normalize(v) for v in vectors]
    return normalize([sum(col) for col in zip(*units)])


def max_overlap_with_others(box: Box, others: Sequence[Box]) -> float:
    """Largest share of ``box`` covered by one of ``others``.

    Args:
        box: ``(x1, y1, x2, y2)``.
        others: Other boxes.

    Returns:
        ``max(area(box ∩ other) / area(box))`` (0 when there are none).
    """
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return 0.0
    best = 0.0
    for o in others:
        w = min(box[2], o[2]) - max(box[0], o[0])
        h = min(box[3], o[3]) - max(box[1], o[1])
        if w > 0 and h > 0:
            best = max(best, w * h / area)
    return best


def usable_for_embedding(box: Box, others: Sequence[Box], config: ReIDConfig) -> bool:
    """Tell whether a person box is large and unoccluded enough for an embedding.

    Args:
        box: Person box.
        others: Boxes of the other persons in the same frame.
        config: Re-ID parameters.

    Returns:
        True if the box is at least ``min_box_height`` tall and overlaps no
        other person by more than ``max_overlap``.
    """
    if box[3] - box[1] < config.min_box_height:
        return False
    return max_overlap_with_others(box, others) <= config.max_overlap


@dataclass
class Identity:
    """A persistent person with its appearance gallery.

    Attributes:
        person_id: Persistent id (1, 2, ... in order of creation).
        first_seen: Time of the first frame of the identity.
        last_seen: Time of the latest frame of any of its tracks.
        track_ids: Tracker ids assigned to this identity, in order.
        gallery: Most recent unit embeddings.
        last_sample: Time of the latest embedding added to the gallery.
    """

    person_id: int
    first_seen: float
    last_seen: float
    track_ids: List[int] = field(default_factory=list)
    gallery: Deque[List[float]] = field(default_factory=deque)
    last_sample: float = -math.inf

    def embedding(self) -> Optional[List[float]]:
        """Return the mean gallery embedding, or ``None`` if the gallery is empty."""
        return mean_vector(list(self.gallery)) if self.gallery else None


@dataclass
class _Pending:
    first_seen: float
    samples: List[List[float]] = field(default_factory=list)


@dataclass
class ReIDDecision:
    """Outcome of matching a new track.

    Attributes:
        track_id: Tracker id.
        person_id: Assigned person id.
        reused: True if an existing identity was reused.
        similarity: Similarity to the chosen identity (``None`` for a new one
            without candidates or embeddings).
    """

    track_id: int
    person_id: int
    reused: bool
    similarity: Optional[float] = None


class ReIdentifier:
    """Maps tracker ids to persistent person ids with an appearance gallery."""

    def __init__(self, config: Optional[ReIDConfig] = None) -> None:
        """Create an empty re-identifier.

        Args:
            config: Parameters; defaults when ``None``.
        """
        self.config = config or ReIDConfig()
        self._identities: Dict[int, Identity] = {}
        self._track_to_person: Dict[int, int] = {}
        self._pending: Dict[int, _Pending] = {}
        self._next_id = 1

    @property
    def identities(self) -> Dict[int, Identity]:
        """Identities by person id (creation order)."""
        return self._identities

    def person_of(self, track_id: int) -> Optional[int]:
        """Return the person id of a tracker id, or ``None`` if it is not decided yet."""
        return self._track_to_person.get(track_id)

    def needs_embedding(self, t: float, track_id: int) -> bool:
        """Tell whether an embedding of this track would be used at time ``t``.

        Pending (new) tracks need one every frame; decided tracks only when
        their identity's gallery is due for a new sample. Lets the caller skip
        the embedder for most boxes.

        Args:
            t: Frame time in seconds.
            track_id: Tracker id.

        Returns:
            True if :meth:`update` would use an embedding.
        """
        if not self.config.enabled:
            return False
        person_id = self._track_to_person.get(track_id)
        if person_id is None:
            return True
        return t - self._identities[person_id].last_sample >= self.config.sample_seconds

    def update(
        self, t: float, track_id: int, embedding: Optional[Vector]
    ) -> Tuple[Optional[int], Optional[ReIDDecision]]:
        """Add one frame of a track.

        Args:
            t: Frame time in seconds (non-decreasing).
            track_id: Tracker id.
            embedding: Appearance embedding of the person in this frame, or
                ``None`` when the box is not usable.

        Returns:
            ``(person_id, decision)``: the person id (``None`` while the track
            is pending) and the decision taken in this call, if any.
        """
        cfg = self.config
        if not cfg.enabled:
            return track_id, None
        person_id = self._track_to_person.get(track_id)
        if person_id is not None:
            self._observe(self._identities[person_id], t, embedding)
            return person_id, None

        pending = self._pending.setdefault(track_id, _Pending(first_seen=t))
        if embedding is not None:
            pending.samples.append(normalize(embedding))
        if len(pending.samples) < cfg.min_samples and t - pending.first_seen < cfg.max_pending_seconds:
            return None, None
        decision = self._decide(t, track_id, pending)
        del self._pending[track_id]
        return decision.person_id, decision

    def _decide(self, t: float, track_id: int, pending: _Pending) -> ReIDDecision:
        cfg = self.config
        query = mean_vector(pending.samples) if pending.samples else None
        best: Optional[Tuple[float, Identity]] = None
        second = -math.inf
        if query is not None:
            for identity in self._identities.values():
                if not self._is_candidate(identity, pending.first_seen, t):
                    continue
                ref = identity.embedding()
                if ref is None or len(ref) != len(query):
                    continue
                score = cosine(query, ref)
                if best is None or score > best[0]:
                    if best is not None:
                        second = best[0]
                    best = (score, identity)
                elif score > second:
                    second = score
        if best is not None and best[0] >= cfg.threshold and best[0] - second >= cfg.margin:
            identity = best[1]
            reused, similarity = True, best[0]
        else:
            identity = Identity(self._next_id, first_seen=pending.first_seen, last_seen=t,
                                gallery=deque(maxlen=cfg.gallery_size))
            self._identities[identity.person_id] = identity
            self._next_id += 1
            reused, similarity = False, (best[0] if best is not None else None)
        identity.track_ids.append(track_id)
        self._track_to_person[track_id] = identity.person_id
        for sample in pending.samples:
            identity.gallery.append(sample)
        identity.last_sample = t if pending.samples else identity.last_sample
        identity.last_seen = max(identity.last_seen, t)
        return ReIDDecision(track_id, identity.person_id, reused, similarity)

    def _is_candidate(self, identity: Identity, track_start: float, t: float) -> bool:
        # Seen while the new track already existed: two different persons.
        if identity.last_seen >= track_start:
            return False
        absence = self.config.max_absence_seconds
        return absence <= 0 or t - identity.last_seen <= absence

    def _observe(self, identity: Identity, t: float, embedding: Optional[Vector]) -> None:
        identity.last_seen = max(identity.last_seen, t)
        if embedding is not None and t - identity.last_sample >= self.config.sample_seconds:
            identity.gallery.append(normalize(embedding))
            identity.last_sample = t
