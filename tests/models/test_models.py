import pytest
import enum
from unittest.mock import patch
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.models.custom_model import CustomModel
from advsecurenet.models.standard_model import StandardModel
from advsecurenet.shared.types import ModelType


def test_infer_model_type_standard():
    model_type = ModelFactory.infer_model_type('resnet18')
    assert model_type == ModelType.STANDARD


def test_infer_model_type_custom():
    model_type = ModelFactory.infer_model_type('CustomMnistModel')
    assert model_type == ModelType.CUSTOM


def test_infer_model_type_unsupported():
    with pytest.raises(ValueError):
        ModelFactory.infer_model_type('unsupported_model')


def test_get_model_standard():
    model = ModelFactory.get_model('resnet18', num_classes=10)
    assert model is not None


def test_get_model_custom():
    model = ModelFactory.get_model('CustomMnistModel', num_classes=10)
    assert model is not None


def test_get_model_unsupported():
    with pytest.raises(ValueError):
        ModelFactory.get_model('unsupported_model', num_classes=10)


def test_available_models():
    models = ModelFactory.available_models()
    assert len(models) > 0


def test_get_available_standard_models():
    models = ModelFactory.available_standard_models()
    assert len(models) > 0


def test_get_available_custom_models():
    models = ModelFactory.available_custom_models()
    assert len(models) > 0

def test_available_model_weights():
    weights = ModelFactory.available_weights("resnet18")
    assert type(weights) == enum.EnumMeta
    assert len(weights) > 0

def test_available_model_weights_unsupported():
    with pytest.raises(ValueError):
        ModelFactory.available_weights("unsupported_model")

def test_available_model_weights_custom():
    with patch('advsecurenet.models.model_factory.ModelFactory.infer_model_type', return_value=ModelType.CUSTOM):
        # expect value error
        with pytest.raises(ValueError, match="Custom models do not support pretrained weights."):
            ModelFactory.available_weights("CustomMnistModel")

def test_custom_model_weights():
    with pytest.raises(NotImplementedError):
        CustomModel.available_weights("CustomMnistModel")

def test_standard_model_weights():
    weights = StandardModel.available_weights("resnet18")
    assert type(weights) == enum.EnumMeta
    assert len(weights) > 0
