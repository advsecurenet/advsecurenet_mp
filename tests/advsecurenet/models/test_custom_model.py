import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.custom_model import CustomModel
from advsecurenet.shared.types.configs.model_config import CustomModelConfig


class MockCustomModel:
    def __init__(self, num_classes, num_input_channels, **kwargs):
        self.architecture = {
            "num_classes": num_classes,
            "num_input_channels": num_input_channels,
        }


@pytest.fixture
def custom_model_config():
    return CustomModelConfig(
        custom_models_path="CustomModels",
        model_name="MockCustomModel",
        architecture={"num_classes": 10, "num_input_channels": 3},
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("importlib.import_module")
def test_load_model(mock_import_module, custom_model_config):
    mock_module = MagicMock()
    mock_module.MockCustomModel = MockCustomModel
    mock_import_module.return_value = mock_module

    model = CustomModel(custom_model_config)
    model.load_model()

    assert isinstance(model.model, MockCustomModel)
    assert (
        model.model.architecture["num_classes"]
        == custom_model_config.architecture["num_classes"]
    )
    assert (
        model.model.architecture["num_input_channels"]
        == custom_model_config.architecture["num_input_channels"]
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model_class_not_found(custom_model_config):
    with patch("importlib.import_module") as mock_import_module:
        mock_module = MagicMock()
        del mock_module.MockCustomModel
        mock_import_module.return_value = mock_module

        with pytest.raises(
            ValueError,
            match="Model class MockCustomModel not found in module advsecurenet.models.CustomModels.MockCustomModel",
        ):
            model = CustomModel(custom_model_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.listdir")
@patch("os.path.isfile")
@patch("os.path.abspath")
def test_models(mock_abspath, mock_isfile, mock_listdir):
    fake_custom_model_path = (
        "your/path/to/advsecurenet/advsecurenet/models/CustomModels"
    )
    custom_model_python_file_dir = (
        "your/path/to/advsecurenet/advsecurenet/models/custom_model.py"
    )

    mock_abspath.return_value = custom_model_python_file_dir
    mock_listdir.return_value = [
        "MockCustomModel.py",
        "__init__.py",
        "some_other_file.txt",
    ]

    # Configure os.path.isfile mock using side_effect for conditional return
    # This simulates which of the listed items are actually files
    def isfile_side_effect(path):
        # Check if the path matches the files we want to simulate as existing files
        if path == os.path.join(fake_custom_model_path, "MockCustomModel.py"):
            return True
        if path == os.path.join(fake_custom_model_path, "__init__.py"):
            return True

        return False

    mock_isfile.side_effect = isfile_side_effect

    models = CustomModel.models()
    assert "MockCustomModel" in models
    assert "__init__" not in models
    assert "some_other_file" not in models
    assert len(models) == 1

    mock_listdir.assert_called_once_with(fake_custom_model_path)

    # Check that isfile was called for each item returned by listdir
    assert mock_isfile.call_count == len(mock_listdir.return_value)
    mock_isfile.assert_any_call(
        os.path.join(fake_custom_model_path, "MockCustomModel.py")
    )
    mock_isfile.assert_any_call(os.path.join(fake_custom_model_path, "__init__.py"))
    mock_isfile.assert_any_call(
        os.path.join(fake_custom_model_path, "some_other_file.txt")
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("importlib.import_module")
def test_available_weights_not_implemented(mock_import_module, custom_model_config):
    mock_module = MagicMock()
    mock_module.MockCustomModel = MockCustomModel
    mock_import_module.return_value = mock_module

    model = CustomModel(custom_model_config)
    with pytest.raises(
        NotImplementedError, match="This method is not applicable for custom models."
    ):
        model.available_weights("MockCustomModel")
