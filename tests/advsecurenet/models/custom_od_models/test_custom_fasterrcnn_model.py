import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import importlib

from advsecurenet.models.CustomODModels.CustomFasterRCNNModel import (
    CustomFasterRCNNModel,
)


class DummyTorchVisionDetModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._eval = False
        self._loaded_sd = None

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        self._param = self._param.to(device)
        return self

    def eval(self):
        self._eval = True
        return self

    def train(self, mode=True):
        self._eval = not mode
        return self

    def __call__(self, x, targets=None):
        # If list of tensors and targets provided -> return loss dict
        if isinstance(x, list) and targets is not None:
            return {"loss_cls": torch.tensor(1.0), "loss_box_reg": torch.tensor(2.0)}
        # Else return predictions list
        outs = []
        b = len(x) if isinstance(x, list) else 1
        for i in range(b):
            outs.append(
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
            )
        return outs

    # Provide modules() to satisfy source loop over batchnorm modules
    def modules(self):
        # no BatchNorm modules inside dummy; just yield self
        yield self

    # Minimal state dict interface for weight loading
    def state_dict(self):
        return {"a": torch.tensor(1.0)}

    def load_state_dict(self, sd, strict=False):
        self._loaded_sd = sd
        return type("_LD", (), {"missing_keys": [], "unexpected_keys": []})()


@pytest.fixture
def patched_frcnn():
    frcnn_mod = importlib.import_module(
        "advsecurenet.models.CustomODModels.CustomFasterRCNNModel"
    )
    with patch.object(
        frcnn_mod,
        "fasterrcnn_resnet50_fpn_v2",
        autospec=True,
    ) as mock_ctor, patch.object(
        frcnn_mod,
        "FasterRCNN_ResNet50_FPN_V2_Weights",
        autospec=True,
    ) as mock_w:
        dummy = DummyTorchVisionDetModel()
        mock_ctor.return_value = dummy
        mock_w.DEFAULT = MagicMock()
        mock_w.DEFAULT.meta = {"categories": ["__background__", "person", "car"]}
        yield mock_ctor, mock_w


@pytest.mark.advsecurenet
def test_init_pretrained_with_backbone(patched_frcnn):
    model = CustomFasterRCNNModel(
        num_classes=91, pretrained=True, pretrained_backbone=True, device="cpu"
    )
    assert model.expects_numpy_images is False
    assert model.num_classes == 91
    assert model._model is not None
    assert model.device.type in ("cpu", "cuda")
    assert isinstance(model.categories, list)


@pytest.mark.advsecurenet
def test_init_pretrained_without_backbone(patched_frcnn):
    # Should still construct successfully
    model = CustomFasterRCNNModel(
        num_classes=42, pretrained=True, pretrained_backbone=False, device="cpu"
    )
    assert model.num_classes == 42


@pytest.mark.advsecurenet
def test_init_non_pretrained(patched_frcnn):
    model = CustomFasterRCNNModel(num_classes=5, pretrained=False, device="cpu")
    assert model.num_classes == 5


@pytest.mark.advsecurenet
def test_forward_eval_and_train_paths(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    # eval path -> predictions list
    model.eval()
    x = torch.zeros((2, 3, 8, 8))
    outs = model(x)
    assert isinstance(outs, list) and isinstance(outs[0], dict)
    # train path -> loss dict augmented with loss_total
    model.train()
    outs_train = model(
        x,
        targets=[
            {
                "boxes": torch.zeros((1, 4)),
                "labels": torch.zeros((1,), dtype=torch.long),
            }
        ]
        * 2,
    )
    assert isinstance(outs_train, dict)
    assert "loss_total" in outs_train
    assert outs_train["loss_total"] == sum(
        v for k, v in outs_train.items() if k != "loss_total"
    )


@pytest.mark.advsecurenet
def test_predict_and_predict_raw(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    x = torch.zeros((2, 3, 8, 8))
    # predict should call eval internally for not training
    preds = model.predict(x, training=False)
    assert isinstance(preds, list) and isinstance(preds[0], dict)
    # predict_raw calls underlying model directly
    raw = model.predict_raw([x[0]])
    assert isinstance(raw, list) and isinstance(raw[0], dict)

    # training=True path should not force eval, still returns forward outputs
    preds_train_flag = model.predict(x, training=True)
    assert isinstance(preds_train_flag, list) and isinstance(preds_train_flag[0], dict)


@pytest.mark.advsecurenet
def test_predict_per_batch_and_initialize_inference_model(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    # initialize_inference_model should .to(device) and .eval()
    inf = model.initialize_inference_model(model._model, device=model.device)
    assert inf is not None
    # predict_per_batch should accept torch batch and use inference model
    imgs = torch.zeros((2, 3, 8, 8))
    preds = model.predict_per_batch(imgs, inf, clip_values=(0, 255))
    assert isinstance(preds, list) and isinstance(preds[0], dict)


@pytest.mark.advsecurenet
def test_prepare_training_inputs(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    images = torch.zeros((2, 3, 8, 8))
    # prepare_training_inputs expects a dict-of-lists {'boxes': [..], 'labels': [..]}
    targets = {
        "boxes": [torch.tensor([[0, 0, 1, 1]]), torch.tensor([[0, 0, 1, 1]])],
        "labels": [torch.tensor([1]), torch.tensor([1])],
    }
    imgs_out, targets_out = model.prepare_training_inputs(images, targets)
    assert isinstance(imgs_out, list) and len(imgs_out) == 2
    assert all(isinstance(t, torch.Tensor) and t.requires_grad for t in imgs_out)
    # Should be a list of two dicts (aligned per image)
    assert isinstance(targets_out, list) and len(targets_out) == 2


@pytest.mark.advsecurenet
def test_preprocess_single_image_tensor_and_normalization(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    # 3D tensor single image with values > 1 should be normalized to <=1
    img = (torch.ones((3, 8, 8)) * 255).to(torch.float32)
    out = model.preprocess_x_for_loss_calculation(img, requires_grad=True)
    assert isinstance(out, torch.Tensor) and out.ndim == 4
    assert out.max() <= 1.0 + 1e-6
    assert out.requires_grad


@pytest.mark.advsecurenet
def test_calculate_loss_scores_and_logits_paths(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    # scores path
    preds_scores = [{"scores": torch.tensor([0.5, 0.25])}]
    loss_scores = model.calculate_loss(preds_scores, target_val=1.0)
    assert isinstance(loss_scores, torch.Tensor)
    # logits path
    preds_logits = [{"logits": torch.randn(3, 4)}]
    loss_logits = model.calculate_loss(preds_logits, target_val=0.0)
    assert isinstance(loss_logits, torch.Tensor)


@pytest.mark.advsecurenet
def test_preprocess_x_for_loss_calculation_numpy_and_torch(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    arr = (np.zeros((2, 3, 8, 8)) * 255).astype(np.uint8)
    out_np = model.preprocess_x_for_loss_calculation(arr, requires_grad=True)
    assert (
        isinstance(out_np, torch.Tensor) and out_np.requires_grad and out_np.dim() == 4
    )
    tx = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    out_t = model.preprocess_x_for_loss_calculation(tx, requires_grad=False)
    assert (
        isinstance(out_t, torch.Tensor) and out_t.dim() == 4 and not out_t.requires_grad
    )


@pytest.mark.advsecurenet
def test_translate_labels_and_align_targets(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    labs = [
        {"boxes": np.array([[0, 0, 1, 1]]), "labels": np.array([1])},
        {"boxes": np.array([[0.5, 0.5, 0.6, 0.6]]), "labels": np.array([2])},
    ]
    out = model.translate_labels(labs, batch_size=2)
    assert isinstance(out, list) and isinstance(out[0]["boxes"], torch.Tensor)
    # padding
    out2 = model._align_targets_to_batch(labs, batch_size=3)
    assert len(out2) == 3 and out2[-1]["boxes"].shape[0] == 0
    # truncation
    out3 = model._align_targets_to_batch(labs * 2, batch_size=2)
    assert len(out3) == 2

    # None input -> padded empty targets
    out_none = model._align_targets_to_batch(None, batch_size=2)
    assert len(out_none) == 2 and out_none[0]["boxes"].shape == (0, 4)

    # numpy inputs conversion path
    labs_np = [
        {
            "boxes": np.array([[1, 2, 3, 4]], dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
        }
    ]
    out_np = model.translate_labels(labs_np, batch_size=1)
    assert isinstance(out_np, list) and isinstance(out_np[0]["boxes"], torch.Tensor)


@pytest.mark.advsecurenet
def test_translate_predictions(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    outs = [
        {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "scores": torch.tensor([0.7]),
            "labels": torch.tensor([1]),
        }
    ]
    preds = model._translate_predictions(outs)
    assert isinstance(preds, list) and preds[0]["boxes"].shape == (1, 4)
    assert "label_names" in preds[0]


@pytest.mark.advsecurenet
def test_forward_with_list_input_inference_and_training(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    # inference with list input
    imgs_list = [torch.zeros((3, 8, 8)), torch.zeros((3, 8, 8))]
    outs_inf = model.forward(imgs_list)
    assert isinstance(outs_inf, list) and isinstance(outs_inf[0], dict)
    # training with targets
    model.train()
    outs_train = model.forward(
        imgs_list,
        targets=[
            {
                "boxes": torch.zeros((1, 4)),
                "labels": torch.zeros((1,), dtype=torch.long),
            }
        ]
        * 2,
    )
    assert isinstance(outs_train, dict) and "loss_total" in outs_train


@pytest.mark.advsecurenet
def test_predict_per_batch_with_list_input_and_mock_inference(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")

    # Mock an inference model that returns list of predictions dicts
    def fake_infer(imgs):
        # imgs may be list of tensors
        b = len(imgs)
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.5]),
                "labels": torch.tensor([1]),
            }
            for _ in range(b)
        ]

    imgs = [torch.zeros((3, 8, 8)), torch.zeros((3, 8, 8))]
    preds = model.predict_per_batch(
        imgs, inference_model=fake_infer, clip_values=(0, 255)
    )
    assert (
        isinstance(preds, list)
        and len(preds) == 2
        and isinstance(preds[0]["boxes"], np.ndarray)
    )


@pytest.mark.advsecurenet
def test_initialize_inference_model_calls_to_and_eval(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    m = MagicMock()
    m.to.return_value = m
    m.eval.return_value = m
    out = model.initialize_inference_model(m, device=model.device)
    assert out is m
    m.to.assert_called_once()
    m.eval.assert_called_once()


@pytest.mark.advsecurenet
def test_load_model_weights_pth_changes_and_message(
    tmp_path, patched_frcnn, monkeypatch, capsys
):
    # Prepare a dummy .pth file with nested state_dict structure
    pth = tmp_path / "weights.pth"
    torch.save({"state_dict": {"some.key": torch.tensor(3.14)}}, pth)

    # Force parameter hash to change between before/after
    from advsecurenet.models.CustomODModels.CustomFasterRCNNModel import (
        CustomFasterRCNNModel,
    )

    seq = iter(["HASH_BEFORE", "HASH_AFTER"])  # different
    monkeypatch.setattr(
        CustomFasterRCNNModel, "_parameters_sha256", lambda self: next(seq)
    )

    model = CustomFasterRCNNModel(
        pretrained=False, device="cpu", model_weights_path=str(pth)
    )
    out = capsys.readouterr().out
    assert "Model parameters changed" in out


@pytest.mark.advsecurenet
def test_load_model_weights_pth_unchanged_message(
    tmp_path, patched_frcnn, monkeypatch, capsys
):
    pth = tmp_path / "weights2.pth"
    torch.save({"model": {"another.key": torch.tensor(2.72)}}, pth)

    from advsecurenet.models.CustomODModels.CustomFasterRCNNModel import (
        CustomFasterRCNNModel,
    )

    seq = iter(["SAME", "SAME"])  # unchanged
    monkeypatch.setattr(
        CustomFasterRCNNModel, "_parameters_sha256", lambda self: next(seq)
    )

    _ = CustomFasterRCNNModel(
        pretrained=False, device="cpu", model_weights_path=str(pth)
    )
    out = capsys.readouterr().out
    assert "did not change model parameters" in out


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator_coco_mapping(
    monkeypatch, patched_frcnn
):
    # Patch mapping so label 1 -> 0 contiguous, 99 filtered out
    import advsecurenet.datasets.COCO.coco_utils as coco_mod

    monkeypatch.setattr(coco_mod, "ID_TO_CONTIGUOUS", {1: 0, 5: 4})

    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    # two detections, one with unknown label (99) to be dropped
    outs = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([1, 99]),
        }
    ]
    # Call the internal helper directly to avoid getattr issues on some torch versions
    preds = model._translate_predictions_for_map_evaluator_coco(outs)
    assert len(preds) == 1
    assert preds[0]["boxes"].shape[0] == 1  # one kept
    assert preds[0]["labels"].ndim == 1


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator_non_coco(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")
    outs = [
        {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "scores": torch.tensor([0.7]),
            "labels": torch.tensor([2]),
        }
    ]
    preds = model.translate_predictions_for_map_evaluator(outs, dataset_name="voc")
    assert isinstance(preds, list) and preds[0]["labels"].shape == (1,)


@pytest.mark.advsecurenet
def test_predict_per_batch_with_tensor_input(patched_frcnn):
    model = CustomFasterRCNNModel(pretrained=False, device="cpu")

    # inference model returns list[dict] like torchvision
    def fake_infer(imgs):
        b = len(imgs)
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.6]),
                "labels": torch.tensor([1]),
            }
            for _ in range(b)
        ]

    imgs = torch.zeros((2, 3, 8, 8))
    preds = model.predict_per_batch(
        imgs, inference_model=fake_infer, clip_values=(0, 255)
    )
    assert (
        isinstance(preds, list)
        and len(preds) == 2
        and isinstance(preds[0]["boxes"], np.ndarray)
    )
