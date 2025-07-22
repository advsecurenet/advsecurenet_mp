import pytest
import torch
from unittest.mock import MagicMock

from advsecurenet.evaluation.od_adversarial_evaluator import (
    ObjectDetectorAdversarialEvaluator,
)


class DummyModel:
    def __init__(self):
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True
        return torch.zeros((1, 3, 10, 10))


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def dummy_images():
    return torch.zeros((2, 3, 10, 10))


@pytest.fixture
def dummy_targets():
    return [
        {
            "boxes": torch.zeros((1, 4)),
            "labels": torch.zeros((1,)),
            "scores": torch.ones((1,)),
        },
        {
            "boxes": torch.ones((2, 4)),
            "labels": torch.ones((2,)),
            "scores": torch.ones((2,)),
        },
    ]


@pytest.fixture
def evaluator():
    # Patch the mean_average_precision evaluator
    evaluator = ObjectDetectorAdversarialEvaluator(
        evaluators=["mean_average_precision"]
    )
    evaluator.evaluators["mean_average_precision"] = MagicMock()
    return evaluator


def test_update_calls_mean_average_precision(
    evaluator, dummy_model, dummy_images, dummy_targets
):
    adv_images = torch.ones_like(dummy_images)
    evaluator.update(dummy_model, dummy_images, adv_images, dummy_targets)
    evaluator.evaluators["mean_average_precision"].update.assert_called_once_with(
        dummy_model, dummy_images, adv_images, dummy_targets
    )


def test_update_skips_if_not_selected(dummy_model, dummy_images, dummy_targets):
    with pytest.raises(KeyError):
        ObjectDetectorAdversarialEvaluator(evaluators=["not_map"])


def test_get_results_returns_map(evaluator):
    evaluator.evaluators["mean_average_precision"].get_results.return_value = {
        "mAP": 0.5
    }
    results = evaluator.get_results()
    assert "mean_average_precision" in results
    assert results["mean_average_precision"] == {"mAP": 0.5}


def test_init_defaults_to_map():
    evaluator = ObjectDetectorAdversarialEvaluator()
    assert "mean_average_precision" in evaluator.selected_evaluators
