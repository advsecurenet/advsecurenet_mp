from unittest.mock import patch
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from advsecurenet.models.base_model import BaseModel, check_model_loaded


class MockBaseModel(BaseModel):
    def load_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 10),
        )

    def models(self):
        return ["mock_model"]


@pytest.fixture
def mock_base_model():
    return MockBaseModel()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward(mock_base_model):
    x = torch.randn(1, 3, 32, 32)
    output = mock_base_model.forward(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_predict(mock_base_model):
    x = torch.randn(1, 3, 32, 32)
    predicted_classes, max_probabilities = mock_base_model.predict(x)
    assert isinstance(predicted_classes, torch.Tensor)
    assert isinstance(max_probabilities, torch.Tensor)
    assert predicted_classes.shape == (1,)
    assert max_probabilities.shape == (1,)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model(mock_base_model, tmp_path):
    path = tmp_path / "model.pth"
    mock_base_model.save_model(path)
    assert path.exists()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_layer_names(mock_base_model):
    with patch(
        "advsecurenet.models.base_model.get_graph_node_names",
        return_value=([], ["layer1", "layer2"]),
    ):
        layer_names = mock_base_model.get_layer_names()
        assert isinstance(layer_names, list)
        assert "layer1" in layer_names
        assert "layer2" in layer_names


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_layer(mock_base_model):
    layer = mock_base_model.get_layer("0")
    assert isinstance(layer, nn.Module)
    assert isinstance(layer, nn.Conv2d)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_set_layer(mock_base_model):
    new_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    mock_base_model.set_layer("0", new_layer)
    assert isinstance(mock_base_model.model[0], nn.Conv2d)
    assert mock_base_model.model[0].out_channels == 32


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer(mock_base_model):
    new_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    mock_base_model.add_layer(new_layer)
    assert isinstance(mock_base_model.model[-1], nn.Conv2d)
    assert mock_base_model.model[-1].out_channels == 32


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_parent_module_and_name(mock_base_model):
    parent, name = mock_base_model._get_parent_module_and_name("0")
    assert isinstance(parent, nn.Sequential)
    assert name == "0"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_models(mock_base_model):
    models = mock_base_model.models()
    assert isinstance(models, list)
    assert "mock_model" in models


# Mock class to test the decorator


class MockModelClass:
    def __init__(self, model=None):
        self.model = model

    @check_model_loaded
    def some_method(self):
        return "Method called"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_with_model_loaded():
    mock_instance = MockModelClass(model="Dummy Model")
    result = mock_instance.some_method()
    assert result == "Method called"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_without_model_loaded():
    mock_instance = MockModelClass(model=None)
    with pytest.raises(ValueError, match="Model is not loaded."):
        mock_instance.some_method()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_with_custom_model():
    mock_instance = MockModelClass(model="Custom Model")
    result = mock_instance.some_method()
    assert result == "Method called"

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_with_custom_model():
    mock_instance = MockModelClass(model="Custom Model")
    result = mock_instance.some_method()
    assert result == "Method called"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_num_classes_huggingface_style(mock_base_model):
    # Create a dummy class that IS a torch.nn.Module to satisfy PyTorch
    class DummyHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Attach a mock config object to the instance
            self.config = MagicMock()
            self.config.num_labels = 100
        
        def forward(self, x):
            return x

    # Assign an instance of our valid dummy module
    mock_base_model.model = DummyHFModel()
    assert mock_base_model.infer_num_classes() == 100


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_num_classes_classifier_linear(mock_base_model):
    # Mock a model with a single linear classifier layer
    mock_model_with_classifier = nn.Module()
    mock_model_with_classifier.classifier = nn.Linear(10, 50)
    mock_base_model.model = mock_model_with_classifier
    assert mock_base_model.infer_num_classes() == 50


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_num_classes_classifier_sequential(mock_base_model):
    # Mock a model with a sequential classifier
    mock_model_with_seq_classifier = nn.Module()
    mock_model_with_seq_classifier.classifier = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30)
    )
    mock_base_model.model = mock_model_with_seq_classifier
    assert mock_base_model.infer_num_classes() == 30

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_layer_names_fallback(mock_base_model):
    # Test the fallback mechanism when get_graph_node_names fails
    with patch(
        "advsecurenet.models.base_model.get_graph_node_names",
        side_effect=torch.fx.proxy.TraceError,
    ):
        layer_names = mock_base_model.get_layer_names()
        # The mock_base_model has 4 layers in its sequential model
        # named '0', '1', '2', '3'
        assert "0" in layer_names
        assert "1" in layer_names
        assert len(layer_names) == 4


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_layer_non_existent(mock_base_model):
    # Test getting a layer that does not exist
    layer = mock_base_model.get_layer("non_existent_layer")
    assert layer is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_not_sequential_model(mock_base_model):
    # Test adding a layer to a model that is not nn.Sequential
    mock_base_model.model = nn.Linear(10, 10)
    new_layer = nn.ReLU()
    mock_base_model.add_layer(new_layer)
    assert isinstance(mock_base_model.model, nn.Sequential)
    assert len(mock_base_model.model) == 2
    assert isinstance(mock_base_model.model[1], nn.ReLU)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_invalid_position(mock_base_model):
    # Test adding a layer at an invalid position
    with pytest.raises(ValueError, match="Invalid position: 10"):
        mock_base_model.add_layer(nn.ReLU(), position=10)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_not_inplace(mock_base_model):
    # Test adding a layer not in-place
    original_model_len = len(mock_base_model.model)
    new_model = mock_base_model.add_layer(nn.ReLU(), inplace=False)
    assert len(mock_base_model.model) == original_model_len
    assert new_model is not None
    assert isinstance(new_model, nn.Sequential)
    assert len(new_model) == original_model_len + 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_parent_module_and_name_nested(mock_base_model):
    # Test getting parent for a nested layer name
    mock_base_model.model = nn.Sequential(nn.Sequential(nn.Linear(10, 10)))
    parent, name = mock_base_model._get_parent_module_and_name("0.0")
    assert isinstance(parent, nn.Sequential)
    assert name == "0"
    # The direct child is the Linear layer
    assert isinstance(getattr(parent, name), nn.Linear)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_num_classes_fc_style(mock_base_model):
    # Mock a ResNet-style model with an 'fc' layer
    mock_resnet_style_model = nn.Module()
    mock_resnet_style_model.fc = nn.Linear(10, 10)
    mock_base_model.model = mock_resnet_style_model
    assert mock_base_model.infer_num_classes() == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_num_classes_no_match(mock_base_model):
    # Use a model where the number of classes cannot be inferred
    mock_base_model.model = nn.Sequential(nn.Conv2d(3, 16, 3))
    assert mock_base_model.infer_num_classes() is None

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_at_specific_position(mock_base_model):
    # Test adding a layer at a specific position (not the end)
    new_layer = nn.ReLU()
    mock_base_model.add_layer(new_layer, position=1)
    assert isinstance(mock_base_model.model[1], nn.ReLU)
    assert len(mock_base_model.model) == 5


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_set_layer_non_existent(mock_base_model):
    # Test setting a layer that does not exist, which adds it as a new attribute
    new_layer = nn.ReLU()
    mock_base_model.set_layer("new_layer", new_layer)
    assert hasattr(mock_base_model.model, "new_layer")
    assert getattr(mock_base_model.model, "new_layer") == new_layer

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_abstract_models_method_coverage():
    """
    Tests the abstract 'models' method directly on the base class to ensure
    the 'pass' statement is covered.
    """
    # We call the method directly on the class, passing a dummy 'self'.
    # This executes the method body without needing an instance.
    result = BaseModel.models(self=None)
    # A function with only 'pass' implicitly returns None.
    assert result is None