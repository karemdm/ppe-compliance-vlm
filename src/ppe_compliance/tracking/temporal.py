"""Temporal confirmation (hysteresis) of per-person PPE states (standard library only).

Each frame gives, per tracked person and PPE item, an :class:`Observation`
(worn, not worn or unknown). A state only changes after a sliding window of
``window_seconds`` in which a majority of the informative frames agree, so
single-frame detector noise and missed detections do not flip it.

Times are in seconds and are passed in by the caller (e.g. ``frame_index / fps``).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Mapping, Optional, Tuple

_EPS = 1e-9


class Observation(Enum):
    """What one frame says about one PPE item of one person."""

    WORN = "worn"
    NOT_WORN = "not_worn"
    UNKNOWN = "unknown"


class ItemState(Enum):
    """Confirmed state of a PPE item (or of a person overall)."""

    UNKNOWN = "unknown"
    OK = "ok"
    VIOLATION = "violation"


@dataclass(frozen=True)
class TemporalConfig:
    """Parameters of the temporal confirmation rule.

    Attributes:
        window_seconds: Length of the sliding window, and minimum time a track
            must have been present before its state can change.
        min_majority: Share of the informative frames (worn + not worn) the
            winning side needs. Must be in ``(0.5, 1]`` so both sides cannot win.
        min_coverage: Minimum share of informative frames among all frames of
            the person in the window; below it the state is kept.
        lost_seconds: A track unseen for this long is forgotten.
        min_frames: Minimum number of informative frames in the window before
            the state can change (guards against decisions on a handful of
            frames, e.g. right after the person reappears).
    """

    window_seconds: float = 3.0
    min_majority: float = 0.6
    min_coverage: float = 0.3
    lost_seconds: float = 2.0
    min_frames: int = 5

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a parameter is out of range.
        """
        if self.window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        if not 0.5 < self.min_majority <= 1.0:
            raise ValueError("min_majority must be in (0.5, 1]")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if self.lost_seconds < 0:
            raise ValueError("lost_seconds must be >= 0")
        if self.min_frames < 1:
            raise ValueError("min_frames must be >= 1")


def resolve_observation(worn_conf: Optional[float], not_worn_conf: Optional[float]) -> Observation:
    """Turn the best positive / negative confidences of an item into an observation.

    Args:
        worn_conf: Highest confidence of an associated positive box, or ``None``.
        not_worn_conf: Highest confidence of an associated negative box, or ``None``.

    Returns:
        The observation; when both are present the higher confidence wins and a
        tie counts as not worn (the conservative choice).
    """
    if worn_conf is None and not_worn_conf is None:
        return Observation.UNKNOWN
    if not_worn_conf is None:
        return Observation.WORN
    if worn_conf is None or not_worn_conf >= worn_conf:
        return Observation.NOT_WORN
    return Observation.WORN


class ItemTracker:
    """Sliding-window majority vote with hysteresis for one item of one person.

    Each update is O(1) amortized: entries are appended once and evicted once,
    and the worn / not-worn counts are kept up to date incrementally.
    """

    def __init__(self, config: TemporalConfig, start_time: Optional[float] = None) -> None:
        """Create an empty tracker.

        Args:
            config: Rule parameters.
            start_time: Time the track was first seen; the first update when ``None``.
        """
        self.config = config
        self.start_time = start_time
        self.state = ItemState.UNKNOWN
        self._window: Deque[Tuple[float, Observation]] = deque()
        self.worn = 0
        self.not_worn = 0

    @property
    def total(self) -> int:
        """Number of frames currently in the window."""
        return len(self._window)

    def update(self, t: float, obs: Observation) -> ItemState:
        """Add one frame observation and re-evaluate the state.

        Args:
            t: Frame time in seconds (non-decreasing).
            obs: Observation of this frame.

        Returns:
            The (possibly unchanged) confirmed state.
        """
        cfg = self.config
        if self.start_time is None:
            self.start_time = t
        self._window.append((t, obs))
        self._count(obs, 1)
        horizon = t - cfg.window_seconds - _EPS
        while self._window and self._window[0][0] < horizon:
            self._count(self._window.popleft()[1], -1)

        if t - self.start_time + _EPS < cfg.window_seconds:
            return self.state  # not present long enough yet
        informative = self.worn + self.not_worn
        if informative < cfg.min_frames or informative < cfg.min_coverage * self.total - _EPS:
            return self.state  # too little evidence: keep the current state
        if self.not_worn >= cfg.min_majority * informative - _EPS:
            self.state = ItemState.VIOLATION
        elif self.worn >= cfg.min_majority * informative - _EPS:
            self.state = ItemState.OK
        return self.state

    def _count(self, obs: Observation, delta: int) -> None:
        if obs is Observation.WORN:
            self.worn += delta
        elif obs is Observation.NOT_WORN:
            self.not_worn += delta


@dataclass
class PersonStatus:
    """Confirmed compliance of one person after a frame.

    Attributes:
        track_id: Id the state is kept under: the persistent person id when
            re-identification is on (see :mod:`ppe_compliance.tracking.reid`), else the
            tracker id. For a track still being re-identified
            (``identified=False``) it is the tracker id.
        state: Overall state: VIOLATION if any item is in violation, OK if at
            least one item is OK and none is in violation, else UNKNOWN.
        items: Confirmed state per item key.
        confirmed_violations: Labels of the items in violation (item order).
        identified: False while re-identification of a new track is pending.
    """

    track_id: int
    state: ItemState
    items: Dict[str, ItemState] = field(default_factory=dict)
    confirmed_violations: List[str] = field(default_factory=list)
    identified: bool = True


@dataclass
class _Track:
    first_seen: float
    last_seen: float
    items: Dict[str, ItemTracker] = field(default_factory=dict)


def overall_state(states: Mapping[str, ItemState]) -> ItemState:
    """Combine item states into a person state.

    Args:
        states: Confirmed state per item.

    Returns:
        VIOLATION if any item is VIOLATION, OK if any is OK, else UNKNOWN.
    """
    values = set(states.values())
    if ItemState.VIOLATION in values:
        return ItemState.VIOLATION
    if ItemState.OK in values:
        return ItemState.OK
    return ItemState.UNKNOWN


class PersonComplianceTracker:
    """Applies :class:`ItemTracker` to every item of every tracked person."""

    def __init__(self, config: Optional[TemporalConfig] = None,
                 labels: Optional[Mapping[str, str]] = None) -> None:
        """Create an empty tracker.

        Args:
            config: Rule parameters; defaults when ``None``.
            labels: Violation label per item key (e.g. ``{"no-hardhat": "NO-Hardhat"}``);
                the item key itself is used for missing entries.
        """
        self.config = config or TemporalConfig()
        self._labels = dict(labels or {})
        self._tracks: Dict[int, _Track] = {}

    @property
    def track_ids(self) -> List[int]:
        """Ids of the tracks currently kept."""
        return list(self._tracks)

    def update(self, t: float, track_id: int, observations: Mapping[str, Observation]) -> PersonStatus:
        """Add the observations of one person in one frame.

        Args:
            t: Frame time in seconds.
            track_id: Id of the person (person id or tracker id).
            observations: Observation per item key.

        Returns:
            The confirmed status of the person.
        """
        track = self._tracks.get(track_id)
        if track is None or t - track.last_seen > self.config.lost_seconds:
            track = self._tracks[track_id] = _Track(first_seen=t, last_seen=t)
        track.last_seen = t
        states: Dict[str, ItemState] = {}
        for key, obs in observations.items():
            item = track.items.get(key)
            if item is None:
                item = track.items[key] = ItemTracker(self.config, start_time=track.first_seen)
            states[key] = item.update(t, obs)
        violations = [self._labels.get(k, k) for k, s in states.items() if s is ItemState.VIOLATION]
        return PersonStatus(track_id, overall_state(states), states, violations)

    def prune(self, t: float) -> List[int]:
        """Forget tracks unseen for more than ``lost_seconds``.

        Args:
            t: Current time in seconds.

        Returns:
            Ids of the removed tracks.
        """
        lost = [i for i, tr in self._tracks.items() if t - tr.last_seen > self.config.lost_seconds]
        for track_id in lost:
            del self._tracks[track_id]
        return lost
