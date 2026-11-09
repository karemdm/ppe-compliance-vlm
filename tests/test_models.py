"""Tests for the PPE model registry (standard library only)."""

from ppe_compliance.models import (
    DEFAULT_PPE_MODEL,
    PPE_ITEMS,
    PPE_MODELS,
    get_model_spec,
    parse_infractions,
    unmatched_class_names,
)

SAFETYVISION_CLASSES = [
    "Fall-Detected", "Gloves", "Goggles", "Hardhat", "Mask", "NO-Gloves", "NO-Goggles",
    "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "No_Harness", "Person", "Safety Vest",
]
CHD_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone",
    "Safety Vest", "machinery", "utility pole", "vehicle",
]


def test_registry_keys_and_default() -> None:
    assert list(PPE_MODELS) == ["safetyvision", "chd"]
    assert DEFAULT_PPE_MODEL == "safetyvision"
    assert get_model_spec("CHD").key == "chd"
    try:
        get_model_spec("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_class_names_unique_and_items_known() -> None:
    for spec in PPE_MODELS.values():
        lower = [n.lower() for n in spec.class_names()]
        assert len(lower) == len(set(lower)), spec.key
        assert all(p.item in PPE_ITEMS for p in spec.pairs), spec.key
        assert not spec.positive_classes() & spec.infraction_classes(), spec.key
        assert spec.weights.startswith("weights/") and spec.weights.endswith(".pt")


def test_specs_match_published_class_names() -> None:
    assert unmatched_class_names(PPE_MODELS["safetyvision"], SAFETYVISION_CLASSES) == ([], [])
    assert unmatched_class_names(PPE_MODELS["chd"], CHD_CLASSES) == ([], [])
    assert unmatched_class_names(PPE_MODELS["chd"], ["hardhat", "dog"])[1] == ["dog"]


def test_safetyvision_infractions() -> None:
    spec = PPE_MODELS["safetyvision"]
    assert list(spec.infraction_labels()) == [
        "no-hardhat", "no-safety vest", "no-gloves", "no-goggles", "no-mask",
        "no_harness", "fall-detected",
    ]
    assert spec.infraction_labels()["no_harness"] == "Harness (missing)"
    assert spec.positive_classes() == {"hardhat", "safety vest", "gloves", "goggles", "mask"}
    assert spec.imgsz == 896


def test_chd_infractions() -> None:
    spec = PPE_MODELS["chd"]
    assert list(spec.infraction_labels()) == ["no-hardhat", "no-safety vest", "no-mask"]
    assert "gloves" not in spec.items() and "no-gloves" not in spec.infraction_classes()
    assert "fall-detected" not in spec.infraction_classes()
    assert "safety cone" not in spec.positive_classes()
    assert spec.imgsz == 640


def test_parse_infractions() -> None:
    sv, chd = PPE_MODELS["safetyvision"], PPE_MODELS["chd"]
    assert parse_infractions(None, sv) == list(sv.infraction_labels())
    assert parse_infractions(["No-Hardhat"], chd) == ["no-hardhat"]
    for values, spec in ((["no-gloves"], chd), (["no-shoes"], sv)):
        try:
            parse_infractions(values, spec)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {values}")
