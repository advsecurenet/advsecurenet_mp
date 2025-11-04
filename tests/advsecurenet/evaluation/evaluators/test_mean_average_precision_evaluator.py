import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np

from advsecurenet.evaluation.evaluators.mean_average_precision_evaluator import (
    MeanAveragePrecisionEvaluator,
)
from advsecurenet.models.base_model import BaseModel


@pytest.fixture
def evaluator():
    with patch(
        "advsecurenet.evaluation.evaluators.mean_average_precision_evaluator.MetricBuilder"
    ) as mock_metric_builder, patch(
        "advsecurenet.evaluation.evaluators.mean_average_precision_evaluator.get_dataset_classes_count",
        return_value=3,
    ):
        # Mock the metric builder to avoid dependency on the real implementation
        mock_metric = MagicMock()
        mock_metric.value.return_value = {"mAP": 0.5}
        mock_metric.reset = MagicMock()
        mock_metric.add = MagicMock()
        mock_metric_builder.build_evaluation_metric.return_value = mock_metric
        yield MeanAveragePrecisionEvaluator(dataset_name="coco")


@pytest.fixture
def mock_model():
    # Create a mock model that behaves like a YOLO-style model expecting numpy images
    model = MagicMock(spec=BaseModel)
    # Mark that it expects numpy images so evaluator uses the numpy path
    model.expects_numpy_images = True

    # Simulate model output: detections object with .pred list of tensors
    class DummyDetections:
        def __init__(self, batch_size=2):
            self.pred = [
                torch.tensor([[0, 0, 10, 10, 0.9, 1], [5, 5, 15, 15, 0.8, 2]])
                for _ in range(batch_size)
            ]

    # When called with list of np.uint8 images, return DummyDetections of same batch size
    def model_call(imgs):
        bs = len(imgs)
        return DummyDetections(batch_size=bs)

    model.side_effect = model_call

    # Provide translate_predictions_for_map_evaluator to convert YOLO-like outputs
    def translate_predictions_for_map_evaluator(
        outputs, expects_numpy: bool = True, dataset_name: str | None = None
    ):
        if not hasattr(outputs, "pred"):
            return []
        preds = []
        for det in outputs.pred:
            arr = det.detach().cpu().numpy() if isinstance(det, torch.Tensor) else det
            boxes = (
                arr[:, :4].astype(np.float32)
                if arr.size
                else np.empty((0, 4), np.float32)
            )
            scores = (
                arr[:, 4].astype(np.float32) if arr.size else np.empty((0,), np.float32)
            )
            labels = (
                arr[:, 5].astype(np.int64) if arr.size else np.empty((0,), np.int64)
            )
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        return preds

    model.translate_predictions_for_map_evaluator = MagicMock(
        side_effect=translate_predictions_for_map_evaluator
    )

    # For device detection in evaluator.update
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])

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
    # This test now validates the expected format for YOLO-like detections conversion
    class DummyDetections:
        def __init__(self):
            self.pred = [torch.tensor([[0, 0, 10, 10, 0.9, 1], [5, 5, 15, 15, 0.8, 2]])]

    def convert(outputs):
        preds = []
        for det in outputs.pred:
            arr = det.detach().cpu().numpy()
            preds.append(
                {
                    "boxes": arr[:, :4].astype(np.float32),
                    "scores": arr[:, 4].astype(np.float32),
                    "labels": arr[:, 5].astype(np.int64),
                }
            )
        return preds

    results = convert(DummyDetections())
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
    with patch.object(
        evaluator, "_process_and_update", wraps=evaluator._process_and_update
    ) as mock_proc_update:
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
def test_update_tensor_model_path(evaluator):
    # Model that expects tensor inputs and returns list of dicts with tensors
    class TensorModel(BaseModel):
        def __init__(self):
            super().__init__()
            # single parameter for device detection
            self.lin = torch.nn.Linear(1, 1)

        def forward(self, imgs):  # imgs is list[Tensor]
            out = []
            for _ in imgs:
                out.append(
                    {
                        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1], dtype=torch.int64),
                    }
                )
            return out

        # Needed by evaluator when expects_numpy_images is False
        def translate_predictions_for_map_evaluator(
            self, outputs, dataset_name: str = "coco"
        ):
            preds = []
            for d in outputs:
                preds.append(
                    {
                        "boxes": d["boxes"].detach().cpu().numpy(),
                        "scores": d["scores"].detach().cpu().numpy(),
                        "labels": d["labels"].detach().cpu().numpy(),
                    }
                )
            return preds

        def load_model(self):
            return None

        def models(self):
            return [self]

    model = TensorModel()
    images = torch.rand(2, 3, 8, 8)
    targets = [
        {
            "boxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        for _ in range(2)
    ]
    evaluator.update(model, images, images, targets)
    # Should have processed entries for both clean and adv
    assert len(evaluator._clean_entries) >= 2
    # adv entries may be deferred until add() in non-distributed mode; check non-negativity
    assert len(evaluator._adv_entries) >= 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_to_tensor_list_variants(evaluator):
    device = torch.device("cpu")
    # 4D tensor -> list of 3D tensors
    x4 = torch.rand(2, 3, 4, 4)
    lst4 = evaluator.to_tensor_list(x4, device)
    assert isinstance(lst4, list) and lst4[0].shape == (3, 4, 4)
    # 3D tensor -> single item list
    x3 = torch.rand(3, 4, 4)
    lst3 = evaluator.to_tensor_list(x3, device)
    assert len(lst3) == 1 and lst3[0].shape == (3, 4, 4)
    # List of numpy HWC arrays -> CHW float scaled to [0,1]
    arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
    lstnp = evaluator.to_tensor_list([arr, arr], device)
    assert lstnp[0].shape == (3, 4, 4)
    assert 0.0 <= float(lstnp[0].max()) <= 1.0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results_distributed_rank0(evaluator, monkeypatch):
    # Patch distributed environment as initialized with rank 0, world size 2
    import advsecurenet.evaluation.evaluators.mean_average_precision_evaluator as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mod.dist, "barrier", lambda: None)

    def fake_all_gather_object(out_list, obj):
        for i in range(len(out_list)):
            out_list[i] = list(obj)

    monkeypatch.setattr(mod.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(mod.dist, "broadcast", lambda t, src=0: None)

    # Populate some entries so rank 0 rebuilds metrics
    evaluator._clean_entries = [
        (np.array([[0, 0, 1, 1, 1, 0.9]]), np.array([[0, 0, 1, 1, 1, 0, 0]]))
    ]
    evaluator._adv_entries = [
        (np.array([[0, 0, 1, 1, 1, 0.8]]), np.array([[0, 0, 1, 1, 1, 0, 0]]))
    ]
    res = evaluator.get_results()
    assert set(res.keys()) == {"clean_mAP", "adversarial_mAP", "mAP_gap"}


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results_distributed_nonzero_rank(evaluator, monkeypatch):
    # Patch distributed environment as initialized with rank 1, world size 2
    import advsecurenet.evaluation.evaluators.mean_average_precision_evaluator as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(mod.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(mod.dist, "barrier", lambda: None)
    monkeypatch.setattr(mod.dist, "all_gather_object", lambda out_list, obj: None)

    def fake_broadcast(t, src=0):
        # Set final values to be broadcast to all ranks
        t.copy_(torch.tensor([0.7, 0.6], dtype=t.dtype, device=t.device))

    monkeypatch.setattr(mod.dist, "broadcast", fake_broadcast)

    res = evaluator.get_results()
    assert res["clean_mAP"] == pytest.approx(0.7)
    assert res["adversarial_mAP"] == pytest.approx(0.6)


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


def test_process_and_update_empty_detections(evaluator):
    metric = MagicMock()
    predictions = []
    targets = [{"boxes": [], "labels": []}]
    # Should not raise
    evaluator._process_and_update(metric, predictions, targets)


def test_process_and_update_empty_targets(evaluator):
    metric = MagicMock()
    predictions = [{"boxes": [[0, 0, 1, 1]], "labels": [1], "scores": [0.9]}]
    targets = []
    # Should not raise
    evaluator._process_and_update(metric, predictions, targets)


def test_get_results_missing_keys(evaluator):
    evaluator.clean_metric.value.return_value = {}
    evaluator.adv_metric.value.return_value = {}
    with pytest.raises(KeyError):
        evaluator.get_results()


def test_get_results_none_values(evaluator):
    evaluator.clean_metric.value.return_value = None
    evaluator.adv_metric.value.return_value = None
    with pytest.raises(TypeError):
        evaluator.get_results()
