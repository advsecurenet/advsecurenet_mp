import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np

from advsecurenet.evaluation.evaluators.mean_average_precision_evaluator import MeanAveragePrecisionEvaluator
from advsecurenet.models.base_model import BaseModel

@pytest.fixture
def evaluator():
    with patch("advsecurenet.evaluation.evaluators.mean_average_precision_evaluator.MetricBuilder") as mock_metric_builder:
        # Mock the metric builder to avoid dependency on the real implementation
        mock_metric = MagicMock()
        mock_metric.value.return_value = {"mAP": 0.5}
        mock_metric.reset = MagicMock()
        mock_metric.add = MagicMock()
        mock_metric_builder.build_evaluation_metric.return_value = mock_metric
        yield MeanAveragePrecisionEvaluator(num_classes=3)

@pytest.fixture
def mock_model():
    model = MagicMock(spec=BaseModel)
    # Simulate model output: detections with pred attribute
    class DummyDetections:
        def __init__(self, batch_size=2):
            self.pred = [torch.tensor([[0, 0, 10, 10, 0.9, 1], [5, 5, 15, 15, 0.8, 2]]) for _ in range(batch_size)]
    model.side_effect = lambda imgs: DummyDetections(batch_size=len(imgs))
    return model

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialization(evaluator):
    assert evaluator.num_classes == 3
    assert hasattr(evaluator, "clean_metric")
    assert hasattr(evaluator, "adv_metric")

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.clean_metric.reset = MagicMock()
    evaluator.adv_metric.reset = MagicMock()
    evaluator.reset()
    # The reset method may be called more than once (e.g., in __init__ and here), so check at least once
    assert evaluator.clean_metric.reset.call_count >= 1
    assert evaluator.adv_metric.reset.call_count >= 1

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_detections_to_dicts(evaluator):
    class DummyDetections:
        def __init__(self):
            self.pred = [torch.tensor([[0, 0, 10, 10, 0.9, 1], [5, 5, 15, 15, 0.8, 2]])]
    dets = DummyDetections()
    results = evaluator.detections_to_dicts(dets)
    assert isinstance(results, list)
    assert set(results[0].keys()) == {"boxes", "labels", "scores"}
    assert results[0]["boxes"].shape[1] == 4

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_tensor_to_numpy_images(evaluator):
    images = torch.rand((2, 3, 16, 16))
    imgs = evaluator.tensor_to_numpy_images(images)
    assert isinstance(imgs, list)
    assert imgs[0].shape == (16, 16, 3)
    assert imgs[0].dtype == np.uint8

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_and_get_results(evaluator, mock_model):
    # Prepare dummy data
    batch_size = 2
    original_images = torch.rand((batch_size, 3, 16, 16))
    adversarial_images = torch.rand((batch_size, 3, 16, 16))
    targets = [
        {"boxes": [[0, 0, 10, 10], [5, 5, 15, 15]], "labels": [1, 2]},
        {"boxes": [[1, 1, 11, 11]], "labels": [1]},
    ]
    # Patch _process_and_update to just count calls
    with patch.object(evaluator, "_process_and_update", wraps=evaluator._process_and_update) as mock_proc_update:
        evaluator.update(mock_model, original_images, adversarial_images, targets)
        assert mock_proc_update.call_count == 2
    # get_results should return the mocked mAP values
    results = evaluator.get_results()
    assert set(results.keys()) == {"clean_mAP", "adversarial_mAP", "mAP_gap"}
    assert results["clean_mAP"] == 0.5
    assert results["adversarial_mAP"] == 0.5
    assert results["mAP_gap"] == 0.0

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_and_update_handles_exceptions(evaluator):
    # Should not raise even if input is malformed
    metric = MagicMock()
    bad_predictions = [{"boxes": [], "labels": [], "scores": []}]
    bad_targets = [{"boxes": [], "labels": []}]
    # Should print error but not raise
    evaluator._process_and_update(metric, bad_predictions, bad_targets)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_with_empty_targets(evaluator, mock_model):
    original_images = torch.rand((1, 3, 16, 16))
    adversarial_images = torch.rand((1, 3, 16, 16))
    targets = [{"boxes": [], "labels": []}]
    # Should not raise
    evaluator.update(mock_model, original_images, adversarial_images, targets)
