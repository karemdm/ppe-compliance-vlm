"""Registry of the bundled PPE detection models and their class mappings.

Each model is described by a :class:`PPEModelSpec` that pairs, for every PPE
item, the class meaning "worn" (positive) with the class meaning "missing"
(negative, an infraction). Class names are matched in lower case.

Standard library only, so it can be imported by tests and lightweight code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: PPE items in display order, mapped to their display name.
PPE_ITEMS: Dict[str, str] = {
    "hardhat": "Hardhat",
    "safety_vest": "Safety vest",
    "gloves": "Gloves",
    "goggles": "Safety goggles",
    "mask": "Mask",
    "harness": "Harness",
}

#: Display labels of infraction classes that are events, not missing PPE.
EVENT_LABELS: Dict[str, str] = {"fall-detected": "Fall detected"}


@dataclass(frozen=True)
class PPEPair:
    """Positive / negative class names of one PPE item for one model.

    Attributes:
        item: Key in :data:`PPE_ITEMS`.
        positive: Class name meaning the item is worn, or ``None``.
        negative: Class name meaning the item is missing, or ``None``.
    """

    item: str
    positive: Optional[str] = None
    negative: Optional[str] = None


@dataclass(frozen=True)
class MonitoredItem:
    """One item checked by the temporal rule: a PPE item or an event class.

    Attributes:
        key: Lower-case infraction label, e.g. ``"no-hardhat"`` (also used by
            ``--infractions`` and the sidebar filter).
        label: Infraction class name in its original spelling, e.g. ``"NO-Hardhat"``.
        positive: Lower-case class meaning "worn", or ``None`` for negative-only
            items (``No_Harness``) and events (``Fall-Detected``).
    """

    key: str
    label: str
    positive: Optional[str] = None


@dataclass(frozen=True)
class PPEModelSpec:
    """Description of one bundled PPE detection model.

    Attributes:
        key: Identifier used by ``--ppe-model`` and the sidebar.
        display_name: Human-readable name.
        weights: Weights path relative to the repository root.
        source_url: Where the weights come from.
        license: License of the weights.
        imgsz: Inference image size passed to the PPE ``predict`` call.
        pairs: Positive / negative classes per PPE item.
        extra_infractions: Infraction classes not tied to a PPE item.
        ignored_classes: Classes of the model that are not used.
    """

    key: str
    display_name: str
    weights: str
    source_url: str
    license: str
    imgsz: int
    pairs: Tuple[PPEPair, ...]
    extra_infractions: Tuple[str, ...] = ()
    ignored_classes: Tuple[str, ...] = ()

    def positive_classes(self) -> FrozenSet[str]:
        """Return the lower-case classes drawn as PPE being worn."""
        return frozenset(p.positive.lower() for p in self.pairs if p.positive)

    def infraction_labels(self) -> Dict[str, str]:
        """Return the supported infractions (lower-case class -> display label), in order."""
        labels = {
            p.negative.lower(): f"{PPE_ITEMS[p.item]} (missing)" for p in self.pairs if p.negative
        }
        for name in self.extra_infractions:
            labels[name.lower()] = EVENT_LABELS.get(name.lower(), name)
        return labels

    def infraction_classes(self) -> FrozenSet[str]:
        """Return the lower-case classes treated as infractions."""
        return frozenset(self.infraction_labels())

    def class_names(self) -> List[str]:
        """Return every class name the spec refers to (original spelling)."""
        names: List[str] = []
        for p in self.pairs:
            names.extend(n for n in (p.positive, p.negative) if n)
        return names + list(self.extra_infractions) + list(self.ignored_classes)

    def monitored_items(self) -> List[MonitoredItem]:
        """Return the items with an infraction class, in :meth:`infraction_labels` order."""
        items = [
            MonitoredItem(p.negative.lower(), p.negative, p.positive.lower() if p.positive else None)
            for p in self.pairs if p.negative
        ]
        return items + [MonitoredItem(name.lower(), name) for name in self.extra_infractions]

    def items(self) -> List[str]:
        """Return the PPE items that have a negative (infraction) class."""
        return [p.item for p in self.pairs if p.negative]


SAFETYVISION = PPEModelSpec(
    key="safetyvision",
    display_name="SafetyVision v2 (YOLOv8s, 13 classes)",
    weights="weights/safetyvision_v2_yolov8s.pt",
    source_url="https://huggingface.co/ayushgupta7777/safetyvision-yolov8",
    license="AGPL-3.0",
    imgsz=896,
    pairs=(
        PPEPair("hardhat", "Hardhat", "NO-Hardhat"),
        PPEPair("safety_vest", "Safety Vest", "NO-Safety Vest"),
        PPEPair("gloves", "Gloves", "NO-Gloves"),
        PPEPair("goggles", "Goggles", "NO-Goggles"),
        PPEPair("mask", "Mask", "NO-Mask"),
        PPEPair("harness", None, "No_Harness"),
    ),
    extra_infractions=("Fall-Detected",),
    ignored_classes=("Person",),
)

CHD = PPEModelSpec(
    key="chd",
    display_name="Construction-Hazard-Detection (YOLO26-L, 11 classes)",
    weights="weights/chd_yolo26l.pt",
    source_url="https://huggingface.co/yihong1120/Construction-Hazard-Detection",
    license="AGPL-3.0",
    imgsz=640,
    pairs=(
        PPEPair("hardhat", "Hardhat", "NO-Hardhat"),
        PPEPair("safety_vest", "Safety Vest", "NO-Safety Vest"),
        PPEPair("mask", "Mask", "NO-Mask"),
    ),
    ignored_classes=("Person", "Safety Cone", "machinery", "utility pole", "vehicle"),
)

#: All bundled PPE models by key; the first one is the default.
PPE_MODELS: Dict[str, PPEModelSpec] = {spec.key: spec for spec in (SAFETYVISION, CHD)}

DEFAULT_PPE_MODEL = SAFETYVISION.key


def get_model_spec(key: str) -> PPEModelSpec:
    """Look up a PPE model by key.

    Args:
        key: Model key, e.g. ``"safetyvision"`` or ``"chd"`` (case-insensitive).

    Returns:
        The model specification.

    Raises:
        ValueError: If the key is unknown.
    """
    try:
        return PPE_MODELS[key.lower()]
    except KeyError:
        raise ValueError(f"Unknown PPE model '{key}'; choose from {list(PPE_MODELS)}") from None


def resolve_model_weights(spec: PPEModelSpec) -> str:
    """Find the weights file of a PPE model.

    The path is looked up in the current directory first and then relative to
    the repository root. PPE weights are not auto-downloaded.

    Args:
        spec: Model specification.

    Returns:
        Path to the existing weights file.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    for candidate in (Path(spec.weights), _REPO_ROOT / spec.weights):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"PPE weights for '{spec.key}' not found: {spec.weights} (see weights/README.md)"
    )


def unmatched_class_names(spec: PPEModelSpec, model_names: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Compare the class names of a loaded model with its specification.

    Args:
        spec: Model specification.
        model_names: Class names reported by the loaded model.

    Returns:
        ``(missing, unexpected)``: spec classes absent from the model, and
        model classes the spec does not know (both lower case, sorted).
    """
    expected = {n.lower() for n in spec.class_names()}
    actual = {n.lower() for n in model_names}
    return sorted(expected - actual), sorted(actual - expected)


def parse_infractions(values: Optional[List[str]], spec: PPEModelSpec) -> List[str]:
    """Validate infraction labels against a model (``None`` means all supported).

    Args:
        values: Labels given by the user, or ``None``.
        spec: Selected PPE model.

    Returns:
        The validated lower-case labels.

    Raises:
        ValueError: If a label is not supported by the model.
    """
    supported = spec.infraction_labels()
    if not values:
        return list(supported)
    unknown = [v for v in values if v.lower() not in supported]
    if unknown:
        raise ValueError(
            f"Infraction(s) {unknown} not supported by PPE model '{spec.key}'; "
            f"choose from {list(supported)}"
        )
    return [v.lower() for v in values]
