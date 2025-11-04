import pytest

from advsecurenet.computer_vision.object_detection.utils import extract_predictions


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_predictions_with_label_names_key_filters_and_orders():
    preds = {
        "label_names": ["cat", "dog", "bird"],
        "boxes": [(0, 1, 2, 3), (10, 11, 12, 13), (20, 21, 22, 23)],
        "scores": [0.2, 0.9, 0.6],
    }
    classes, boxes, scores = extract_predictions(preds, conf_thresh=0.5)
    assert classes == ["dog", "bird"]
    assert boxes == [
        [(10, 11), (12, 13)],
        [(20, 21), (22, 23)],
    ]
    assert scores == [0.9, 0.6]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_predictions_with_labels_and_mapping_out_of_bounds_maps_to_str_index():
    preds = {
        "labels": [0, 3, 1],  # 3 is OOB for provided label_names
        "boxes": [(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)],
        "scores": [0.7, 0.8, 0.9],
    }
    label_names = ["cat", "dog"]
    classes, boxes, scores = extract_predictions(
        preds, conf_thresh=0.6, label_names=label_names
    )
    # Labels map to ["cat", "3", "dog"] then filtered (>0.6) -> keep all
    assert classes == ["cat", "3", "dog"]
    assert boxes == [[(0, 0), (1, 1)], [(1, 1), (2, 2)], [(2, 2), (3, 3)]]
    assert scores == [0.7, 0.8, 0.9]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_predictions_returns_empty_when_no_classes():
    preds = {
        "labels": [],
        "boxes": [],
        "scores": [],
    }
    classes, boxes, scores = extract_predictions(preds, conf_thresh=0.1)
    assert classes == [] and boxes == [] and scores == []


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_predictions_threshold_strict_greater():
    preds = {
        "label_names": ["a", "b", "c"],
        "boxes": [(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)],
        "scores": [0.5, 0.5, 0.51],
    }
    classes, boxes, scores = extract_predictions(preds, conf_thresh=0.5)
    # Only scores strictly greater than threshold kept -> index 2 only
    assert classes == ["c"]
    assert boxes == [[(2, 2), (3, 3)]]
    assert scores == [0.51]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_predictions_no_scores_above_threshold_returns_empty():
    preds = {
        "label_names": ["x", "y"],
        "boxes": [(0, 0, 1, 1), (1, 1, 2, 2)],
        "scores": [0.1, 0.2],
    }
    classes, boxes, scores = extract_predictions(preds, conf_thresh=0.5)
    assert classes == [] and boxes == [] and scores == []
