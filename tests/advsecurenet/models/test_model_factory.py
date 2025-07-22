from enum import EnumMeta
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from advsecurenet.models import CustomModel, StandardModel
from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.CustomModels.CustomMnistModel import CustomMnistModel
from advsecurenet.models.external_model import ExternalModel
from advsecurenet.models.huggingface_model import HuggingFaceModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.model_config import (
    CreateModelConfig,
    CustomModelConfig,
    ExternalModelConfig,
    StandardModelConfig,
    HuggingFaceInputConfig,
    HuggingFaceResolvedConfig,
)
from advsecurenet.shared.types.model import ModelType
from advsecurenet.utils.reproducibility_utils import set_seed

import logging
import dataclasses


@pytest.fixture
def create_model_config():
    return CreateModelConfig(
        model_name="resnet18",
        architecture={"num_classes": 10},
        pretrained=True,
        weights="IMAGENET1K_V1",
        random_seed=42,
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_standard():
    with patch.object(StandardModel, "models", return_value=["resnet18"]):
        model_type = ModelFactory.infer_model_type("resnet18")
        assert model_type == ModelType.STANDARD


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_custom():
    with patch.object(CustomModel, "models", return_value=["CustomMnistModel"]):
        model_type = ModelFactory.infer_model_type("CustomMnistModel")
        assert model_type == ModelType.CUSTOM


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_invalid():
    with patch.object(StandardModel, "models", return_value=[]), patch.object(
        CustomModel, "models", return_value=[]
    ):
        with pytest.raises(ValueError, match="Unsupported model"):
            ModelFactory.infer_model_type("invalid_model")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(StandardModel, "models", return_value=["resnet18"])
@patch("advsecurenet.models.StandardModel")
def test_create_model_standard(mock_standard_model, create_model_config):
    cfg = CreateModelConfig(
        model_name="resnet18",
        architecture={"num_classes": 10},
        pretrained=True,
        weights="IMAGENET1K_V1",
    )
    model = ModelFactory.create_model(config=cfg)
    assert isinstance(model, BaseModel)
    assert model._model_name == "resnet18"


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.models.ExternalModel")
@patch("os.path.exists", return_value=True)
@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
def test_create_model_external(
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_exists,
    mock_external_model,
):
    config = CreateModelConfig(
        model_name="MockExternalModel",
        architecture={"num_classes": 10},
        model_arch_path="/path/to/mock_model.py",
        pretrained=False,
        is_external=True,
    )

    # Mock the module loading
    spec_mock = MagicMock()
    mock_spec_from_file_location.return_value = spec_mock
    module_mock = MagicMock()
    mock_module_from_spec.return_value = module_mock

    # Mock the external model class in the module
    setattr(module_mock, "MockExternalModel", MagicMock(spec=BaseModel))

    # Mock the return value of the ExternalModel constructor
    model_instance = MagicMock(spec=BaseModel)
    mock_external_model.return_value = model_instance

    with patch(
        "advsecurenet.models.external_model.filter_kwargs_for_callable",
        lambda cls, kwargs: kwargs,
    ):
        created_model = ModelFactory.create_model(config=config)

        mock_exists.assert_called_with("/path/to/mock_model.py")
        mock_spec_from_file_location.assert_called_once_with(
            "MockExternalModel", "/path/to/mock_model.py"
        )
        mock_module_from_spec.assert_called_once_with(spec_mock)
        assert isinstance(created_model, BaseModel)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(CustomModel, "models", return_value=["CustomMnistModel"])
@patch("advsecurenet.models.CustomModel")
def test_create_model_custom(mock_custom_model, mock_models):
    config = CreateModelConfig(
        model_name="CustomMnistModel",
        architecture={"num_classes": 10, "num_input_channels": 1},
        pretrained=False,
    )
    model = CustomMnistModel()
    mock_custom_model.return_value = model

    created_model = ModelFactory.create_model(config=config)

    assert isinstance(created_model.model, CustomMnistModel)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_create_model_config():
    config = CreateModelConfig(
        model_name="resnet18",
        architecture={"num_classes": 10},
        pretrained=True,
        random_seed=42,
    )
    with pytest.raises(
        ValueError, match="Pretrained standard models do not support random seed"
    ):
        ModelFactory._validate_create_model_config(ModelType.STANDARD, config)

    config = CreateModelConfig(
        model_name="CustomMnistModel", architecture={"num_classes": 10}, pretrained=True
    )
    with pytest.raises(
        ValueError, match="Custom models do not support pretrained weights"
    ):
        ModelFactory._validate_create_model_config(ModelType.CUSTOM, config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    new_layer = nn.Linear(20, 10)
    updated_model = ModelFactory.add_layer(model, new_layer, 1)
    assert isinstance(updated_model, nn.Sequential)
    assert len(updated_model) == 3
    assert isinstance(updated_model[1], nn.Linear)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_models():
    with patch.object(StandardModel, "models", return_value=["resnet18"]), patch.object(
        CustomModel, "models", return_value=["CustomMnistModel"]
    ):
        available_models = ModelFactory.available_models()
        assert available_models == ["resnet18", "CustomMnistModel"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_standard_models():
    with patch.object(StandardModel, "models", return_value=["resnet18"]):
        available_models = ModelFactory.available_standard_models()
        assert available_models == ["resnet18"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_custom_models():
    with patch.object(CustomModel, "models", return_value=["CustomMnistModel"]):
        available_models = ModelFactory.available_custom_models()
        assert available_models == ["CustomMnistModel"]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_weights():
    with patch.object(StandardModel, "models", return_value=["resnet18"]), patch.object(
        StandardModel, "available_weights", return_value=MagicMock(spec=EnumMeta)
    ):
        weights = ModelFactory.available_weights("resnet18")
        assert isinstance(weights, EnumMeta)


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestResolveConfigAndWarn:

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_none_uses_kwargs(self, mock_logger):
        """Test Case 1: No valid config, creates from kwargs."""
        kwargs = {"model_name": "test_model_from_kwargs", "pretrained": True}
        resolved_config = ModelFactory._resolve_config_and_warn(None, kwargs)

        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "test_model_from_kwargs"
        assert resolved_config.pretrained is True
        mock_logger.debug.assert_any_call(
            "No valid model config provided, creating from kwargs."
        )

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_invalid_type_uses_kwargs(self, mock_logger):
        """Test Case 1 variant: Invalid config type, creates from kwargs."""
        invalid_config = {"some_key": "some_value"}  # Not a CreateModelConfig
        kwargs = {"model_name": "test_model_from_kwargs_2", "pretrained": False}
        resolved_config = ModelFactory._resolve_config_and_warn(invalid_config, kwargs)

        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "test_model_from_kwargs_2"
        assert resolved_config.pretrained is False
        mock_logger.debug.assert_any_call(
            "No valid model config provided, creating from kwargs."
        )

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_no_kwargs(self, mock_logger):
        """Test Case 2: Valid config, no kwargs, uses provided config."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=True)
        kwargs = {}
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)

        assert resolved_config is initial_config  # Should be the same object
        assert resolved_config.model_name == "initial_model"
        mock_logger.debug.assert_any_call(
            "Valid CreateModelConfig provided and kwargs is empty. Using provided config directly."
        )

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_no_relevant_overlap(self, mock_logger):
        """Test Case 3a: Valid config, kwargs present but no overlap on CreateModelConfig fields."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=True)
        kwargs = {"non_config_key": "some_value", "another_non_config_key": 123}
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)

        assert resolved_config is initial_config  # Should be the same object
        assert resolved_config.model_name == "initial_model"
        # The current implementation logs "Provided config used, no overlapping kwargs detected."
        # even if kwargs are present but don't overlap with config fields.
        mock_logger.debug.assert_any_call(
            "Provided config used, no overlapping kwargs detected."
        )

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_and_overlap_warning(self, mock_logger):
        """Test Case 3b: Valid config, kwargs with overlap, kwargs prioritized, warning issued."""
        initial_config = CreateModelConfig(
            model_name="initial_model", pretrained=False, revision="rev1"
        )
        kwargs = {
            "model_name": "kwarg_model",
            "pretrained": True,
            "weights": "IMAGENET1K_V2",
        }  # weights is new

        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)

        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "kwarg_model"  # Kwarg prioritized
        assert resolved_config.pretrained is True  # Kwarg prioritized
        assert resolved_config.revision == "rev1"  # From original config
        assert resolved_config.weights == "IMAGENET1K_V2"  # From kwargs (new field)

        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert (
            "Overlap detected between provided 'config' object and keyword arguments"
            in warning_call_args
        )
        assert "'model_name'" in warning_call_args
        assert "'pretrained'" in warning_call_args
        assert "Values from keyword arguments will be prioritized." in warning_call_args
        mock_logger.debug.assert_any_call(
            "Merging overlapping kwargs into provided config."
        )

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_overlap_same_values(self, mock_logger):
        """Test Case 3c: Valid config, kwargs with overlap but same values (no warning for these)."""
        initial_config = CreateModelConfig(model_name="same_model", pretrained=True)
        # Overlap with same values, plus one different value to trigger merge path
        kwargs = {
            "model_name": "same_model",
            "pretrained": True,
            "revision": "rev_from_kwarg",
        }

        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)

        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "same_model"
        assert resolved_config.pretrained is True
        assert resolved_config.revision == "rev_from_kwarg"

        # A warning IS still issued because 'revision' is different.
        # The code checks if *any* overlapping key has a different value.
        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert (
            "'revision'" in warning_call_args
        )  # Only revision should cause the warning message detail
        assert "'model_name'" not in warning_call_args  # Because values were same
        assert "'pretrained'" not in warning_call_args  # Because values were same
        mock_logger.debug.assert_any_call(
            "Merging overlapping kwargs into provided config."
        )


# --- Tests for create_model with Hugging Face ---

# Mock HuggingFaceModel and its dependencies for these tests
MOCK_HF_MODEL_INSTANCE = MagicMock(spec=HuggingFaceModel)


@pytest.fixture
def mock_hf_dependencies():
    with patch(
        "advsecurenet.models.model_factory.determine_identifier_and_soruce",
        return_value=("user/hf-model", MagicMock(name="MODEL_IDENTIFIER")),
    ) as mock_det_id_src, patch(
        "advsecurenet.models.model_factory.ModelFactory.infer_model_type",
        return_value=ModelType.HUGGINGFACE,
    ) as mock_infer_type, patch(
        "advsecurenet.models.model_factory.ModelFactory._validate_create_model_config"
    ) as mock_validate, patch(
        "advsecurenet.models.model_factory.set_seed"
    ) as mock_set_seed, patch(
        "advsecurenet.utils.huggingface_utils.huggingface_general_utils.process_hf_identifier",
        return_value="user/hf-model-processed",
    ) as mock_proc_id, patch(
        "advsecurenet.models.model_factory.HuggingFaceModel",
        return_value=MOCK_HF_MODEL_INSTANCE,
    ) as mock_hf_model_ctor, patch(
        "advsecurenet.utils.huggingface_utils.huggingface_model_utils.huggingface_model_hub_utils.verify_hf_model_identifier_exists",
        return_value=True,
    ) as mock_verify_exists:
        yield {
            "determine_identifier_and_soruce": mock_det_id_src,
            "infer_model_type": mock_infer_type,
            "_validate_create_model_config": mock_validate,
            "set_seed": mock_set_seed,
            "process_hf_identifier": mock_proc_id,
            "HuggingFaceModel_ctor": mock_hf_model_ctor,
            "verify_hf_model_identifier_exists": mock_verify_exists,
        }


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_config_object(mock_hf_dependencies):
    """Test creating a Hugging Face model using a CreateModelConfig object."""
    hf_create_config = CreateModelConfig(
        model_name="user/hf-model-name1",
        model_identifier="user/hf-model",
        architecture={"num_labels": 5},
        pretrained=True,
        revision="main",
        cache_dir="/tmp/hf",
        trust_remote_code=True,
        model_class_name="CustomBert",
        random_seed=123,
    )
    created_model = ModelFactory.create_model(config=hf_create_config)
    assert created_model == MOCK_HF_MODEL_INSTANCE

    mocked_hf_constructor = mock_hf_dependencies["HuggingFaceModel_ctor"]
    mocked_set_seed_func = mock_hf_dependencies["set_seed"]
    mocked_process_id_func = mock_hf_dependencies["process_hf_identifier"]

    mocked_hf_constructor.assert_called_once()
    call_args_config = mocked_hf_constructor.call_args[0][0]
    assert isinstance(call_args_config, HuggingFaceResolvedConfig)
    # This is where the first AssertionError occurs
    assert call_args_config.architecture == {"num_labels": 5}
    assert call_args_config.pretrained is True
    assert call_args_config.revision == "main"
    assert call_args_config.cache_dir == "/tmp/hf"
    assert call_args_config.trust_remote_code is True
    assert call_args_config.model_class_name == "CustomBert"
    assert call_args_config.model_name == "user/hf-model-name1"

    mocked_set_seed_func.assert_called_once_with(123)

    # Reset mocks for the second call within this test
    mocked_hf_constructor.reset_mock()
    mocked_set_seed_func.reset_mock()
    mocked_process_id_func.reset_mock()  # This resets call count etc.

    hf_create_config2 = CreateModelConfig(
        model_name="my_hf_model_explicit_name",
        model_identifier="user/hf-model2",  # Different identifier
        architecture={"num_labels": 10},
        pretrained=False,
        random_seed=456,
    )
    # IMPORTANT: Update the return_value of the mock for the new identifier
    mocked_process_id_func.return_value = "user/hf-model2-processed"

    ModelFactory.create_model(config=hf_create_config2)

    mocked_hf_constructor.assert_called_once()
    call_args_rerun_config = mocked_hf_constructor.call_args[0][0]
    assert isinstance(call_args_rerun_config, HuggingFaceResolvedConfig)
    # This would be the second place an AssertionError could occur if the first was fixed
    assert call_args_rerun_config.model_name == "my_hf_model_explicit_name"
    assert call_args_rerun_config.architecture == {"num_labels": 10}
    assert call_args_rerun_config.pretrained is False
    mocked_set_seed_func.assert_called_once_with(456)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_kwargs(mock_hf_dependencies):
    """Test creating a Hugging Face model using kwargs."""
    mocked_hf_constructor = mock_hf_dependencies["HuggingFaceModel_ctor"]
    mocked_process_id_func = mock_hf_dependencies["process_hf_identifier"]
    mocked_set_seed_func = mock_hf_dependencies["set_seed"]

    mocked_hf_constructor.reset_mock()
    # Reset the return_value to the fixture's default if it was changed by another test or part of a test
    mocked_process_id_func.return_value = (
        "user/hf-model-processed"  # Or whatever the default for the fixture is
    )
    mocked_process_id_func.reset_mock()  # Resets call count
    mocked_set_seed_func.reset_mock()

    # Patch the return value of the process_hf_identifier mock specifically for this test's scenario
    with patch(
        "advsecurenet.models.model_factory.determine_identifier_and_soruce",
        return_value=("kwarg_hf_model_name", MagicMock(name="MODEL_NAME")),
    ) as _, patch.object(
        mocked_process_id_func, "return_value", "kwarg_hf_model_name-processed"
    ):  # Temporarily change return_value

        created_model = ModelFactory.create_model(
            model_name="kwarg_hf_model_name",
            architecture={"num_labels": 10},
            pretrained=False,
            revision="dev",
        )

        assert created_model == MOCK_HF_MODEL_INSTANCE
        mocked_set_seed_func.assert_not_called()

        mocked_hf_constructor.assert_called_once()
        call_args_config = mocked_hf_constructor.call_args[0][0]
        assert isinstance(call_args_config, HuggingFaceResolvedConfig)
        # This is where the AssertionError occurs
        assert call_args_config.model_name == "kwarg_hf_model_name"
        assert call_args_config.architecture == {"num_labels": 10}
        assert call_args_config.pretrained is False
        assert call_args_config.revision == "dev"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_huggingface():
    """Test that infer_model_type returns HUGGINGFACE for valid HF models"""
    with patch.object(StandardModel, "models", return_value=[]), patch.object(
        CustomModel, "models", return_value=[]
    ), patch(
        "advsecurenet.utils.huggingface_utils.huggingface_model_utils.huggingface_model_hub_utils.verify_hf_model_identifier_exists",
        return_value=True,
    ):
        model_type = ModelFactory.infer_model_type("microsoft/resnet-50")
        assert model_type == ModelType.HUGGINGFACE


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_weights_custom_model_error():
    """Test that available_weights raises error for custom models"""
    with patch.object(
        CustomModel, "models", return_value=["CustomMnistModel"]
    ), patch.object(StandardModel, "models", return_value=[]), patch(
        "advsecurenet.utils.huggingface_utils.huggingface_model_utils.huggingface_model_hub_utils.verify_hf_model_identifier_exists",
        return_value=False,
    ):
        with pytest.raises(
            ValueError,
            match="Custom models do not support pretrained weights. Instead, you can load the weights after loading the model.",
        ):
            ModelFactory.available_weights("CustomMnistModel")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_non_sequential():
    """Test add_layer with non-Sequential model (converts to Sequential)"""
    model = nn.Linear(10, 5)  # Non-Sequential model
    new_layer = nn.ReLU()
    updated_model = ModelFactory.add_layer(model, new_layer)

    assert isinstance(updated_model, nn.Sequential)
    assert len(updated_model) == 2
    assert isinstance(updated_model[0], nn.Linear)
    assert isinstance(updated_model[1], nn.ReLU)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_position_out_of_bounds():
    """Test add_layer raises error for invalid positions"""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    new_layer = nn.Linear(20, 10)

    # Test position too negative
    with pytest.raises(ValueError, match="Position out of bounds."):
        ModelFactory.add_layer(model, new_layer, -2)

    # Test position too large
    with pytest.raises(ValueError, match="Position out of bounds."):
        ModelFactory.add_layer(model, new_layer, 3)  # model has only 2 layers


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer_insert_at_specific_position():
    """Test add_layer inserting at specific position (not -1 or end)"""
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))
    new_layer = nn.ReLU()
    updated_model = ModelFactory.add_layer(model, new_layer, 1)

    assert isinstance(updated_model, nn.Sequential)
    assert len(updated_model) == 3
    assert isinstance(updated_model[0], nn.Linear)  # Original first layer
    assert isinstance(updated_model[1], nn.ReLU)  # New layer inserted at position 1
    assert isinstance(
        updated_model[2], nn.Linear
    )  # Original second layer moved to position 2


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_exception_handling():
    """Test that create_model properly handles and wraps exceptions"""
    # Create a config that will cause an exception during model creation
    with patch.object(StandardModel, "models", return_value=["resnet18"]), patch(
        "advsecurenet.models.model_factory.set_seed",
        side_effect=RuntimeError("Random seed error"),
    ):
        config = CreateModelConfig(
            model_name="resnet18",
            architecture={"num_classes": 10},
            pretrained=False,  # No pretrained to allow random_seed
            random_seed=42,  # This will trigger set_seed which we patch to raise exception
        )
        with pytest.raises(
            ValueError,
            match="Error creating model. Please check the model_name and other arguments. Error: Random seed error",
        ):
            ModelFactory.create_model(config=config)
