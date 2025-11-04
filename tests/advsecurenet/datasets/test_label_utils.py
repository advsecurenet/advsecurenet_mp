import pytest
from unittest.mock import MagicMock

from advsecurenet.datasets.COCO.coco_utils import (
    COCO_INSTANCE_CATEGORY_NAMES,
    ID_TO_CONTIGUOUS,
)
from advsecurenet.datasets.PascalVOC.pascalvoc_utils import PASCAL_VOC_CATEGORY_NAMES
from advsecurenet.datasets.label_utils import (
    coco_label_id_to_pascal,
    coco_label_ids_to_pascal,
    get_dataset_classes_count,
    get_dataset_labels,
    get_model_label_names,
    resolve_label_names,
)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_dataset_labels_known_names_case_insensitive():
    coco = get_dataset_labels("coco")
    pascal = get_dataset_labels("PaScAl_VoC")
    assert isinstance(coco, list) and len(coco) > 0
    assert isinstance(pascal, list) and len(pascal) > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_dataset_labels_unknown_returns_empty():
    assert get_dataset_labels("unknown_dataset") == []
    assert get_dataset_labels("") == []
    assert get_dataset_labels(None) == []


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_model_label_names_hf_id2label_full():
    # Simulate HF model with config.id2label
    cfg = MagicMock()
    cfg.id2label = {0: "background", 1: "cat", 2: "dog"}
    model = MagicMock()
    model.config = cfg
    names = get_model_label_names(model)
    assert names == ["background", "cat", "dog"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_model_label_names_hf_id2label_missing_values():
    # Keys are numeric but not all provided; function should fill with stringified index
    cfg = MagicMock()
    cfg.id2label = {0: "zero", 2: "two"}  # missing 1 -> expect "1"
    model = MagicMock()
    model.config = cfg
    names = get_model_label_names(model)
    assert names == ["zero", "1", "two"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_model_label_names_yolo_names_on_model():
    model = MagicMock()
    model.names = ["__background__", "person", "car"]
    names = get_model_label_names(model)
    assert names == ["__background__", "person", "car"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_model_label_names_yolo_names_on_nested_model():
    inner = MagicMock()
    inner.names = ["a", "b"]
    model = MagicMock()
    model.model = inner
    names = get_model_label_names(model)
    assert names == []


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_model_label_names_none_or_invalid_returns_empty():
    model = MagicMock()
    # No config / names
    assert get_model_label_names(model) == []
    # config.id2label not a dict
    cfg = MagicMock()
    cfg.id2label = ["x", "y"]
    model = MagicMock()
    model.config = cfg
    assert get_model_label_names(model) == []


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_resolve_label_names_prefer_model_over_dataset():
    # Model provides names -> preferred
    model = MagicMock()
    model.names = ["m0", "m1"]
    out = resolve_label_names(dataset_name="COCO", model=model)
    assert out == ["m0", "m1"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_resolve_label_names_fallback_to_dataset():
    # Model yields none -> dataset fallback
    model = MagicMock()
    out = resolve_label_names(dataset_name="COCO", model=model)
    assert isinstance(out, list) and len(out) > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_resolve_label_names_both_missing_empty():
    assert resolve_label_names(dataset_name=None, model=None) == []


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_dataset_classes_count_known_unknown():
    assert get_dataset_classes_count("coco") == len(COCO_INSTANCE_CATEGORY_NAMES)
    assert get_dataset_classes_count("mystery") == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_coco_label_id_to_pascal_synonyms_and_none():
    airplane_idx = COCO_INSTANCE_CATEGORY_NAMES.index("airplane")
    pascal_idx = PASCAL_VOC_CATEGORY_NAMES.index("aeroplane")
    assert coco_label_id_to_pascal(airplane_idx) == pascal_idx

    raw_airplane = next(k for k, v in ID_TO_CONTIGUOUS.items() if v == airplane_idx)
    assert coco_label_id_to_pascal(raw_airplane, assume_contiguous=False) == pascal_idx

    bad_raw = max(ID_TO_CONTIGUOUS.keys()) + 123
    assert coco_label_id_to_pascal(bad_raw, assume_contiguous=False) == -1
    assert coco_label_id_to_pascal(None) == -1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_coco_label_ids_to_pascal_drop_and_keep_unmapped():
    airplane_idx = COCO_INSTANCE_CATEGORY_NAMES.index("airplane")
    pascal_idx = PASCAL_VOC_CATEGORY_NAMES.index("aeroplane")
    unmapped = 9999

    keep_all = coco_label_ids_to_pascal([airplane_idx, unmapped])
    assert keep_all == [pascal_idx, -1]

    dropped = coco_label_ids_to_pascal([airplane_idx, unmapped], drop_unmapped=True)
    assert dropped == [pascal_idx]

    raw_id = next(k for k, v in ID_TO_CONTIGUOUS.items() if v == airplane_idx)
    mixed = coco_label_ids_to_pascal([raw_id], assume_contiguous=False)
    assert mixed == [pascal_idx]
