import pytest
import torch
import numpy as np
import importlib
import sys
from unittest.mock import patch, MagicMock

# NOTE: Do NOT import CustomRTDetrModel here. We patch first, then import inside fixture.


class DummyHFModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.config = MagicMock()
        self.config.id2label = {0: "__background__", 1: "person", 2: "car"}
        self.config.label2id = {v: k for k, v in self.config.id2label.items()}

    def to(self, device):
        self._param = self._param.to(device)
        return self

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        if "labels" in kwargs and kwargs["labels"] is not None:
            out = MagicMock()
            out.loss = torch.tensor(1.0)
            out.loss_dict = {"ce": torch.tensor(0.1)}
            return out
        out = MagicMock()
        out.logits = torch.randn(2, 5, 3)
        out.pred_boxes = torch.rand(2, 5, 4)
        return out


class DummyProcessor:
    def __init__(self):
        pass

    def to(self, device):
        return self

    @staticmethod
    def from_pretrained(*args, **kwargs):
        return DummyProcessor()

    def post_process_object_detection(self, outputs, target_sizes, threshold=0.7):
        res = []
        for _ in range(target_sizes.shape[0]):
            res.append(
                {
                    "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1], dtype=torch.int64),
                }
            )
        return res

    def __call__(self, images=None, return_tensors=None, do_rescale=False):
        return {
            "pixel_values": torch.rand(len(images), 3, images[0].shape[1], images[0].shape[2]),
            "pixel_mask": torch.ones(len(images), images[0].shape[1], images[0].shape[2]),
        }


@pytest.fixture
def patched_module():
    """
    Patch before importing module to avoid real HF network/model loading.
    """
    with patch(
        "advsecurenet.models.huggingface_model.HuggingFaceModel"
    ) as mock_hf_wrapper, patch(
        "transformers.RTDetrImageProcessor"
    ) as mock_proc_cls:
        mock_proc_instance = DummyProcessor()
        mock_proc_cls.from_pretrained.return_value = mock_proc_instance
        wrapper_instance = mock_hf_wrapper.return_value
        wrapper_instance.model = DummyHFModel()
        wrapper_instance.to.return_value = wrapper_instance
        mod_name = "advsecurenet.models.CustomODModels.CustomRTDetrModel"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        custom_module = importlib.import_module(mod_name)
        yield custom_module


@pytest.mark.advsecurenet
def test_init_constructor(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    assert model._model_name == "CustomRTDetrModel"
    assert isinstance(model.categories, list)
    assert model.num_classes == len(model.categories)
    assert model.device.type in ("cpu", "cuda")


@pytest.mark.advsecurenet
def test_forward_eval_and_train(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    x = torch.rand(2, 3, 16, 16)
    model.eval()
    out = model.forward(x)
    assert hasattr(out, "logits") or isinstance(out, dict)
    model.train()
    targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "class_labels": torch.tensor([1])},
        {"boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]]), "class_labels": torch.tensor([2])},
    ]
    loss_dict = model.forward(x, targets=targets)
    assert isinstance(loss_dict, dict) and "loss_total" in loss_dict


@pytest.mark.advsecurenet
def test_predict_and_predict_raw(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    x = torch.rand(2, 3, 16, 16)
    preds = model.predict(x, training=False)
    assert hasattr(preds, "logits") or isinstance(preds, dict)
    raw = model.predict_raw(x)
    assert hasattr(raw, "logits") or isinstance(raw, dict)


@pytest.mark.advsecurenet
def test_initialize_inference_model_and_predict_per_batch_modeloutput(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    RTDetrEvalAdapter = patched_module.RTDetrEvalAdapter
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    adapter = model.initialize_inference_model(model._model, device=model.device, conf_thresh=0.6)
    assert isinstance(adapter, RTDetrEvalAdapter)
    imgs = torch.zeros((2, 3, 16, 16))
    preds = model.predict_per_batch(imgs, adapter, clip_values=(0, 255))
    assert isinstance(preds, list) and all(isinstance(d, dict) for d in preds)


@pytest.mark.advsecurenet
def test_predict_per_batch_list_path(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")

    class ListModel:
        def __call__(self, pixel_values=None, pixel_mask=None):
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                    "scores": torch.tensor([0.8]),
                    "labels": torch.tensor([1], dtype=torch.int64),
                }
            ]

    imgs = torch.zeros((1, 3, 8, 8))
    preds = model.predict_per_batch(imgs, ListModel(), clip_values=(0, 255))
    assert isinstance(preds, list) and isinstance(preds[0], dict)


@pytest.mark.advsecurenet
def test_adapter_forward_with_pixel_values(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    adapter = model.initialize_inference_model(model._model, device=model.device)
    pv = torch.rand(1, 3, 16, 16)
    out = adapter(pixel_values=pv)
    assert out is not None


@pytest.mark.advsecurenet
def test_adapter_forward_with_x_numpy_list(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    adapter = model.initialize_inference_model(model._model, device=model.device)
    x = [np.zeros((3, 16, 16), dtype=np.float32)]
    out = adapter(x=x)
    assert isinstance(out, list) and isinstance(out[0], dict)


@pytest.mark.advsecurenet
def test_prepare_training_inputs_and_translate_labels(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    model.input_shape = (3, 32, 32)
    model.channels_first = True
    images = torch.zeros((2, 3, 32, 32))
    labels = [
        {"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([1])},
        {"boxes": np.array([[5, 5, 15, 15]]), "labels": np.array([2])},
    ]
    imgs_out, y_out = model.prepare_training_inputs(images, labels)
    assert isinstance(imgs_out, torch.Tensor) and imgs_out.requires_grad
    assert isinstance(y_out, list) and "class_labels" in y_out[0]


@pytest.mark.advsecurenet
def test_calculate_loss_logits_and_scores(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    class Out:
        pass
    out = Out()
    out.logits = torch.randn(1, 3, 4)
    loss1 = model.calculate_loss(out, target_val=1.0)
    assert isinstance(loss1, torch.Tensor)
    loss2 = model.calculate_loss([{"scores": np.array([0.5, 0.25])}], target_val=0.0)
    assert isinstance(loss2, torch.Tensor)


@pytest.mark.advsecurenet
def test_preprocess_x_and_translate_labels_padding_truncation(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    arr = (np.zeros((1, 3, 10, 10)) * 255).astype(np.uint8)
    t = model.preprocess_x_for_loss_calculation(arr, requires_grad=True)
    assert isinstance(t, torch.Tensor) and t.requires_grad
    model.input_shape = (10, 10, 3)
    model.channels_first = False
    labs = [{"boxes": np.array([[0, 0, 2, 2]]), "labels": np.array([1])}]
    out2 = model.translate_labels(labs, batch_size=3)
    assert len(out2) == 3 and out2[0]["boxes"].shape[1] == 4


@pytest.mark.advsecurenet
def test__to_image_list_and_errors(patched_module):
    CustomRTDetrModel = patched_module.CustomRTDetrModel
    model = CustomRTDetrModel(model_name="dummy", device="cpu")
    with pytest.raises(TypeError):
        model._to_image_list(torch.zeros(3, 16, 16))
    lst = model._to_image_list(torch.zeros(2, 3, 8, 8))
    assert isinstance(lst, list) and lst[0].shape == (3, 8, 8)