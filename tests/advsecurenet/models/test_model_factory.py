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
    HuggingFaceResolvedConfig
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
        model_name="resnet18", architecture={"num_classes": 10}, pretrained=True, weights="IMAGENET1K_V1"
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
        model_name="resnet18", architecture={"num_classes": 10}, pretrained=True, random_seed=42
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
        mock_logger.debug.assert_any_call("No valid model config provided, creating from kwargs.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_invalid_type_uses_kwargs(self, mock_logger):
        """Test Case 1 variant: Invalid config type, creates from kwargs."""
        invalid_config = {"some_key": "some_value"} # Not a CreateModelConfig
        kwargs = {"model_name": "test_model_from_kwargs_2", "pretrained": False}
        resolved_config = ModelFactory._resolve_config_and_warn(invalid_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "test_model_from_kwargs_2"
        assert resolved_config.pretrained is False
        mock_logger.debug.assert_any_call("No valid model config provided, creating from kwargs.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_no_kwargs(self, mock_logger):
        """Test Case 2: Valid config, no kwargs, uses provided config."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=True)
        kwargs = {}
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert resolved_config is initial_config # Should be the same object
        assert resolved_config.model_name == "initial_model"
        mock_logger.debug.assert_any_call("Valid CreateModelConfig provided and kwargs is empty. Using provided config directly.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_no_relevant_overlap(self, mock_logger):
        """Test Case 3a: Valid config, kwargs present but no overlap on CreateModelConfig fields."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=True)
        kwargs = {"non_config_key": "some_value", "another_non_config_key": 123}
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert resolved_config is initial_config # Should be the same object
        assert resolved_config.model_name == "initial_model"
        # The current implementation logs "Provided config used, no overlapping kwargs detected."
        # even if kwargs are present but don't overlap with config fields.
        mock_logger.debug.assert_any_call("Provided config used, no overlapping kwargs detected.")


    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_and_overlap_warning(self, mock_logger):
        """Test Case 3b: Valid config, kwargs with overlap, kwargs prioritized, warning issued."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=False, revision="rev1")
        kwargs = {"model_name": "kwarg_model", "pretrained": True, "weights": "IMAGENET1K_V2"} # weights is new
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "kwarg_model" # Kwarg prioritized
        assert resolved_config.pretrained is True         # Kwarg prioritized
        assert resolved_config.revision == "rev1"         # From original config
        assert resolved_config.weights == "IMAGENET1K_V2" # From kwargs (new field)

        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "Overlap detected between provided 'config' object and keyword arguments" in warning_call_args
        assert "'model_name'" in warning_call_args
        assert "'pretrained'" in warning_call_args
        assert "Values from keyword arguments will be prioritized." in warning_call_args
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_overlap_same_values(self, mock_logger):
        """Test Case 3c: Valid config, kwargs with overlap but same values (no warning for these)."""
        initial_config = CreateModelConfig(model_name="same_model", pretrained=True)
        # Overlap with same values, plus one different value to trigger merge path
        kwargs = {"model_name": "same_model", "pretrained": True, "revision": "rev_from_kwarg"} 
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "same_model"
        assert resolved_config.pretrained is True
        assert resolved_config.revision == "rev_from_kwarg"

        # A warning IS still issued because 'revision' is different.
        # The code checks if *any* overlapping key has a different value.
        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "'revision'" in warning_call_args # Only revision should cause the warning message detail
        assert "'model_name'" not in warning_call_args # Because values were same
        assert "'pretrained'" not in warning_call_args # Because values were same
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_only_new_fields_no_warning(self, mock_logger):
        """Test Case: Valid config, kwargs provide only new fields not in original config (no overlap)."""
        initial_config = CreateModelConfig(model_name="initial_model")
        kwargs = {"pretrained": True, "weights": "IMAGENET1K_V2"} # New fields for the config
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "initial_model"
        assert resolved_config.pretrained is True
        assert resolved_config.weights == "IMAGENET1K_V2"

        # In this case, because kwargs are merged into a new config if they are valid fields,
        # it will take the "merging" path if the kwargs are actual config fields.
        # The warning is only if values *differ* for *existing* fields.
        mock_logger.warning.assert_not_called()
        # It will still go through the merge logic because kwargs are present and are config fields
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")


# --- Tests for create_model with Hugging Face ---

# Mock HuggingFaceModel and its dependencies for these tests
MOCK_HF_MODEL_INSTANCE = MagicMock(spec=HuggingFaceModel)

@pytest.fixture
def mock_hf_dependencies(): # Removed 'mocker' from arguments
    with patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("user/hf-model", MagicMock(name="MODEL_IDENTIFIER"))) as mock_det_id_src, \
         patch("advsecurenet.models.model_factory.ModelFactory.infer_model_type", return_value=ModelType.HUGGINGFACE) as mock_infer_type, \
         patch("advsecurenet.models.model_factory.ModelFactory._validate_create_model_config") as mock_validate, \
         patch("advsecurenet.models.model_factory.set_seed") as mock_set_seed, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", return_value="user/hf-model-processed") as mock_proc_id, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel", return_value=MOCK_HF_MODEL_INSTANCE) as mock_hf_model_class, \
         patch("advsecurenet.models.huggingface_model.HuggingFaceModel.verify_hf_identifier_exists", return_value=True) as mock_verify_exists:
        # Yielding a dictionary of mocks in case any test needs to access them directly.
        # If not, a simple 'yield' would suffice.
        yield {
            "determine_identifier_and_soruce": mock_det_id_src,
            "infer_model_type": mock_infer_type,
            "_validate_create_model_config": mock_validate,
            "set_seed": mock_set_seed,
            "process_hf_identifier": mock_proc_id,
            "HuggingFaceModel_class": mock_hf_model_class, # Renamed to avoid conflict if HuggingFaceModel is imported
            "verify_hf_identifier_exists": mock_verify_exists
        }


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_config_object(mock_hf_dependencies): # mock_hf_dependencies will now apply patches
    """Test creating a Hugging Face model using a CreateModelConfig object."""
    hf_create_config = CreateModelConfig(
        model_identifier="user/hf-model", # or model_name
        architecture={"num_labels": 5},
        pretrained=True,
        revision="main",
        cache_dir="/tmp/hf",
        trust_remote_code=True,
        model_class_name="CustomBert",
        random_seed=123
    )

    created_model = ModelFactory.create_model(config=hf_create_config)

    assert created_model == MOCK_HF_MODEL_INSTANCE
    
    # Check that HuggingFaceModel was called with the correctly resolved config
    # Access the mock for HuggingFaceModel class from the fixture if needed, or rely on ModelFactory.HuggingFaceModel
    call_args = ModelFactory.HuggingFaceModel.call_args[0][0]
    assert isinstance(call_args, HuggingFaceResolvedConfig)
    assert call_args.model_id == "user/hf-model-processed" 
    assert call_args.architecture == {"num_labels": 5}
    assert call_args.pretrained is True
    assert call_args.revision == "main"
    assert call_args.cache_dir == "/tmp/hf"
    assert call_args.trust_remote_code is True
    assert call_args.model_class_name == "CustomBert"
    
    hf_create_config_with_name = CreateModelConfig(
        model_name="my_hf_model_explicit_name", 
        model_identifier="user/hf-model", 
        architecture={"num_labels": 5},
        pretrained=True,
        revision="main",
        random_seed=123
    )
    ModelFactory.create_model(config=hf_create_config_with_name) 
    call_args_rerun = ModelFactory.HuggingFaceModel.call_args[0][0]
    assert call_args_rerun.model_name == "my_hf_model_explicit_name"

    ModelFactory.set_seed.assert_called_once_with(123) # This mock is from ModelFactory.set_seed
    ModelFactory.HuggingFaceModel.process_hf_identifier.assert_called_once_with("user/hf-model")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_kwargs(mock_hf_dependencies): # Removed 'mocker'
    """Test creating a Hugging Face model using kwargs."""
    # Use unittest.mock.patch as context managers for specific overrides in this test
    with patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("kwarg_hf_model_name", MagicMock(name="MODEL_NAME"))) as _, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", return_value="kwarg_hf_model_name-processed") as _:

        created_model = ModelFactory.create_model(
            model_name="kwarg_hf_model_name", 
            architecture={"num_labels": 10},
            pretrained=False,
            revision="dev"
        )

        assert created_model == MOCK_HF_MODEL_INSTANCE
        ModelFactory.set_seed.assert_not_called() # This mock is from ModelFactory.set_seed
        
        call_args = ModelFactory.HuggingFaceModel.call_args[0][0]
        assert isinstance(call_args, HuggingFaceResolvedConfig)
        assert call_args.model_id == "kwarg_hf_model_name-processed"
        assert call_args.model_name == "kwarg_hf_model_name" 
        assert call_args.architecture == {"num_labels": 10}
        assert call_args.pretrained is False
        assert call_args.revision == "dev"
        ModelFactory.HuggingFaceModel.process_hf_identifier.assert_called_with("kwarg_hf_model_name")


@pytest.mark.advsecurenet
@pytest.mark.essential
# Use @patch decorators instead of mocker for this test
@patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", side_effect=ValueError("Failed to process identifier"))
@patch("advsecurenet.models.huggingface_model.HuggingFaceModel.verify_hf_identifier_exists", return_value=True)
@patch("advsecurenet.models.model_factory.ModelFactory._validate_create_model_config")
@patch("advsecurenet.models.model_factory.ModelFactory.infer_model_type", return_value=ModelType.HUGGINGFACE)
@patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("bad-hf-url", MagicMock(name="MODEL_IDENTIFIER")))
def test_create_model_huggingface_processing_error(
    mock_determine_id, 
    mock_infer_type, 
    mock_validate_config, 
    mock_verify_exists, 
    mock_process_hf_id
): # Patches are passed as arguments
    """Test error handling when HuggingFaceModel.process_hf_identifier fails."""
    hf_config = CreateModelConfig(model_identifier="bad-hf-url")
    
    with pytest.raises(ValueError, match="Error creating model. Please check the model_name and other arguments. Error: Failed to process identifier"):
        ModelFactory.create_model(config=hf_config)
# ... (imports and existing code before TestResolveConfigAndWarn) ...

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
        mock_logger.debug.assert_any_call("No valid model config provided, creating from kwargs.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_invalid_type_uses_kwargs(self, mock_logger):
        """Test Case 1 variant: Invalid config type, creates from kwargs."""
        invalid_config = {"some_key": "some_value"} # Not a CreateModelConfig
        kwargs = {"model_name": "test_model_from_kwargs_2", "pretrained": False}
        resolved_config = ModelFactory._resolve_config_and_warn(invalid_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "test_model_from_kwargs_2"
        assert resolved_config.pretrained is False
        mock_logger.debug.assert_any_call("No valid model config provided, creating from kwargs.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_no_kwargs(self, mock_logger):
        """Test Case 2: Valid config, no kwargs, uses provided config."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=True)
        kwargs = {}
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert resolved_config is initial_config 
        assert resolved_config.model_name == "initial_model"
        mock_logger.debug.assert_any_call("Valid CreateModelConfig provided and kwargs is empty. Using provided config directly.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_no_relevant_overlap(self, mock_logger):
        """Test Case 3a: Valid config, kwargs present but no overlap on CreateModelConfig fields."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=True)
        kwargs = {"non_config_key": "some_value", "another_non_config_key": 123}
        # Since kwargs_to_merge will be empty, overlapping_keys will be empty.
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert resolved_config is initial_config 
        assert resolved_config.model_name == "initial_model"
        mock_logger.debug.assert_any_call("Provided config used, no overlapping kwargs detected.")


    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_and_overlap_warning(self, mock_logger):
        """Test Case 3b: Valid config, kwargs with overlap, kwargs prioritized, warning issued."""
        initial_config = CreateModelConfig(model_name="initial_model", pretrained=False, revision="rev1")
        kwargs = {"model_name": "kwarg_model", "pretrained": True, "weights": "IMAGENET1K_V2"} 
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "kwarg_model" 
        assert resolved_config.pretrained is True         
        assert resolved_config.revision == "rev1"         
        assert resolved_config.weights == "IMAGENET1K_V2" 

        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "Overlap detected between provided 'config' object and keyword arguments" in warning_call_args
        assert "'model_name'" in warning_call_args
        assert "'pretrained'" in warning_call_args
        # 'weights' is also different from default and will be in overlapping_keys
        assert "'weights'" in warning_call_args 
        assert "Values from keyword arguments will be prioritized." in warning_call_args
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_overlap_same_values(self, mock_logger):
        """Test Case 3c: Valid config, kwargs with overlap but same values (no warning for these specific keys)."""
        initial_config = CreateModelConfig(model_name="same_model", pretrained=True)
        kwargs = {"model_name": "same_model", "pretrained": True, "revision": "rev_from_kwarg"} 
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "same_model"
        assert resolved_config.pretrained is True
        assert resolved_config.revision == "rev_from_kwarg"

        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "'revision'" in warning_call_args 
        assert "'model_name'" not in warning_call_args 
        assert "'pretrained'" not in warning_call_args 
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_new_fields_differing_from_defaults_issues_warning(self, mock_logger):
        """Test: Valid config, kwargs provide fields that differ from config defaults, issues warning."""
        initial_config = CreateModelConfig(model_name="initial_model") # pretrained=False, weights="IMAGENET1K_V1" by default
        kwargs = {"pretrained": True, "weights": "IMAGENET1K_V2"} # These are different from defaults
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "initial_model"
        assert resolved_config.pretrained is True
        assert resolved_config.weights == "IMAGENET1K_V2"

        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "'pretrained'" in warning_call_args
        assert "'weights'" in warning_call_args
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")


# --- Tests for create_model with Hugging Face ---

MOCK_HF_MODEL_INSTANCE = MagicMock(spec=HuggingFaceModel)

@pytest.fixture
def mock_hf_dependencies():
    with patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("user/hf-model", MagicMock(name="MODEL_IDENTIFIER"))) as mock_det_id_src, \
         patch("advsecurenet.models.model_factory.ModelFactory.infer_model_type", return_value=ModelType.HUGGINGFACE) as mock_infer_type, \
         patch("advsecurenet.models.model_factory.ModelFactory._validate_create_model_config") as mock_validate, \
         patch("advsecurenet.models.model_factory.set_seed") as mock_set_seed, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", return_value="user/hf-model-processed") as mock_proc_id, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel", return_value=MOCK_HF_MODEL_INSTANCE) as mock_hf_model_class_ctor, \
         patch("advsecurenet.models.huggingface_model.HuggingFaceModel.verify_hf_identifier_exists", return_value=True) as mock_verify_exists:
        yield {
            "determine_identifier_and_soruce": mock_det_id_src,
            "infer_model_type": mock_infer_type,
            "_validate_create_model_config": mock_validate,
            "set_seed": mock_set_seed,
            "process_hf_identifier": mock_proc_id,
            "HuggingFaceModel_ctor": mock_hf_model_class_ctor, # Mock for the constructor
            "verify_hf_identifier_exists": mock_verify_exists
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
        random_seed=123
    )

    created_model = ModelFactory.create_model(config=hf_create_config)
    assert created_model == MOCK_HF_MODEL_INSTANCE
    
    # Access the mock for the HuggingFaceModel constructor from the fixture
    mocked_hf_constructor = mock_hf_dependencies["HuggingFaceModel_ctor"]
    mocked_hf_constructor.assert_called_once() # Ensure it was called
    call_args = mocked_hf_constructor.call_args[0][0] # Get the first positional arg (the config)

    assert isinstance(call_args, HuggingFaceResolvedConfig)
    assert call_args.model_id == "user/hf-model-processed" 
    assert call_args.architecture == {"num_labels": 5}
    assert call_args.pretrained is True
    assert call_args.revision == "main"
    assert call_args.cache_dir == "/tmp/hf"
    assert call_args.trust_remote_code is True
    assert call_args.model_class_name == "CustomBert"
    assert call_args.model_name == "user/hf-model-name1" 
    
    hf_create_config_with_name = CreateModelConfig(
        model_name="my_hf_model_explicit_name", 
        model_identifier="user/hf-model", 
        architecture={"num_labels": 5},
        pretrained=True,
        revision="main",
        random_seed=123
    )
    # Reset mock for the next call if checking specific calls for this instance
    mocked_hf_constructor.reset_mock() 
    mock_hf_dependencies["process_hf_identifier"].reset_mock()
    mock_hf_dependencies["set_seed"].reset_mock()

    ModelFactory.create_model(config=hf_create_config_with_name) 
    call_args_rerun = mocked_hf_constructor.call_args[0][0]
    assert call_args_rerun.model_name == "my_hf_model_explicit_name"

    mock_hf_dependencies["set_seed"].assert_called_once_with(123) 
    mock_hf_dependencies["process_hf_identifier"].assert_called_once_with("user/hf-model")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_kwargs(mock_hf_dependencies): 
    """Test creating a Hugging Face model using kwargs."""
    mocked_hf_constructor = mock_hf_dependencies["HuggingFaceModel_ctor"]
    mocked_process_id_func = mock_hf_dependencies["process_hf_identifier"]
    mocked_set_seed_func = mock_hf_dependencies["set_seed"]

    with patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("kwarg_hf_model_name", MagicMock(name="MODEL_NAME"))) as _, \
         patch.object(mocked_process_id_func, 'return_value', "kwarg_hf_model_name-processed"): # Patch the mock from fixture

        created_model = ModelFactory.create_model(
            model_name="kwarg_hf_model_name", 
            architecture={"num_labels": 10},
            pretrained=False,
            revision="dev"
        )

        assert created_model == MOCK_HF_MODEL_INSTANCE
        mocked_set_seed_func.assert_not_called() 
        
        mocked_hf_constructor.assert_called_once()
        call_args = mocked_hf_constructor.call_args[0][0]
        assert isinstance(call_args, HuggingFaceResolvedConfig)
        assert call_args.model_id == "kwarg_hf_model_name-processed"
        assert call_args.model_name == "kwarg_hf_model_name" 
        assert call_args.architecture == {"num_labels": 10}
        assert call_args.pretrained is False
        assert call_args.revision == "dev"
        mocked_process_id_func.assert_called_with("kwarg_hf_model_name")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", side_effect=ValueError("Failed to process identifier"))
@patch("advsecurenet.models.huggingface_model.HuggingFaceModel.verify_hf_identifier_exists", return_value=True)
@patch("advsecurenet.models.model_factory.ModelFactory._validate_create_model_config")
@patch("advsecurenet.models.model_factory.ModelFactory.infer_model_type", return_value=ModelType.HUGGINGFACE)
@patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("bad-hf-url", MagicMock(name="MODEL_IDENTIFIER")))
def test_create_model_huggingface_processing_error(
    mock_determine_id, 
    mock_infer_type, 
    mock_validate_config, 
    mock_verify_exists, 
    mock_process_hf_id
): 
    """Test error handling when HuggingFaceModel.process_hf_identifier fails."""
    hf_config = CreateModelConfig(model_name="bad-hf-url-name", model_identifier="bad-hf-url")
    
    with pytest.raises(ValueError, match="Error creating model. Please check the model_name and other arguments. Error: Failed to process identifier"):
        ModelFactory.create_model(config=hf_config)

# ... (rest of your test file)
    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_overlap_same_values(self, mock_logger):
        """Test Case 3c: Valid config, kwargs with overlap but same values (no warning for these)."""
        initial_config = CreateModelConfig(model_name="same_model", pretrained=True)
        # Overlap with same values, plus one different value to trigger merge path
        kwargs = {"model_name": "same_model", "pretrained": True, "revision": "rev_from_kwarg"} 
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "same_model"
        assert resolved_config.pretrained is True
        assert resolved_config.revision == "rev_from_kwarg"

        # A warning IS still issued because 'revision' is different.
        # The code checks if *any* overlapping key has a different value.
        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "'revision'" in warning_call_args # Only revision should cause the warning message detail
        assert "'model_name'" not in warning_call_args # Because values were same
        assert "'pretrained'" not in warning_call_args # Because values were same
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")

    @patch("advsecurenet.models.model_factory.logger")
    def test_resolve_config_valid_with_kwargs_only_new_fields_no_warning(self, mock_logger):
        """Test Case: Valid config, kwargs provide only new fields not in original config (no overlap)."""
        initial_config = CreateModelConfig(model_name="initial_model")
        kwargs = {"pretrained": True, "weights": "IMAGENET1K_V2"} # New fields for the config
        
        resolved_config = ModelFactory._resolve_config_and_warn(initial_config, kwargs)
        
        assert isinstance(resolved_config, CreateModelConfig)
        assert resolved_config.model_name == "initial_model"
        assert resolved_config.pretrained is True
        assert resolved_config.weights == "IMAGENET1K_V2"

        # In this case, because kwargs are merged into a new config if they are valid fields,
        # it will take the "merging" path if the kwargs are actual config fields.
        # The warning is only if values *differ* for *existing* fields.
        mock_logger.warning.assert_not_called() # This line was the subject of the first error in the previous turn
        # It will still go through the merge logic because kwargs are present and are config fields
        mock_logger.debug.assert_any_call("Merging overlapping kwargs into provided config.")


# --- Tests for create_model with Hugging Face ---

# Mock HuggingFaceModel and its dependencies for these tests
MOCK_HF_MODEL_INSTANCE = MagicMock(spec=HuggingFaceModel)

@pytest.fixture
def mock_hf_dependencies(): # Removed 'mocker' from arguments
    with patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("user/hf-model", MagicMock(name="MODEL_IDENTIFIER"))) as mock_det_id_src, \
         patch("advsecurenet.models.model_factory.ModelFactory.infer_model_type", return_value=ModelType.HUGGINGFACE) as mock_infer_type, \
         patch("advsecurenet.models.model_factory.ModelFactory._validate_create_model_config") as mock_validate, \
         patch("advsecurenet.models.model_factory.set_seed") as mock_set_seed, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", return_value="user/hf-model-processed") as mock_proc_id, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel", return_value=MOCK_HF_MODEL_INSTANCE) as mock_hf_model_class, \
         patch("advsecurenet.models.huggingface_model.HuggingFaceModel.verify_hf_identifier_exists", return_value=True) as mock_verify_exists:
        # Yielding a dictionary of mocks in case any test needs to access them directly.
        # If not, a simple 'yield' would suffice.
        yield {
            "determine_identifier_and_soruce": mock_det_id_src,
            "infer_model_type": mock_infer_type,
            "_validate_create_model_config": mock_validate,
            "set_seed": mock_set_seed,
            "process_hf_identifier": mock_proc_id,
            "HuggingFaceModel_class": mock_hf_model_class, # Renamed to avoid conflict if HuggingFaceModel is imported
            "verify_hf_identifier_exists": mock_verify_exists
        }


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_config_object(mock_hf_dependencies): # mock_hf_dependencies will now apply patches
    """Test creating a Hugging Face model using a CreateModelConfig object."""
    hf_create_config = CreateModelConfig(
        model_name="user/hf-model-name1", # ADDED model_name
        model_identifier="user/hf-model", 
        architecture={"num_labels": 5},
        pretrained=True,
        revision="main",
        cache_dir="/tmp/hf",
        trust_remote_code=True,
        model_class_name="CustomBert",
        random_seed=123
    )

    created_model = ModelFactory.create_model(config=hf_create_config)

    assert created_model == MOCK_HF_MODEL_INSTANCE
    
    call_args = ModelFactory.HuggingFaceModel.call_args[0][0]
    assert isinstance(call_args, HuggingFaceResolvedConfig)
    assert call_args.model_id == "user/hf-model-processed" 
    assert call_args.architecture == {"num_labels": 5}
    assert call_args.pretrained is True
    assert call_args.revision == "main"
    assert call_args.cache_dir == "/tmp/hf"
    assert call_args.trust_remote_code is True
    assert call_args.model_class_name == "CustomBert"
    assert call_args.model_name == "user/hf-model-name1" # Ensure model_name is passed
    
    hf_create_config_with_name = CreateModelConfig(
        model_name="my_hf_model_explicit_name", 
        model_identifier="user/hf-model", 
        architecture={"num_labels": 5},
        pretrained=True,
        revision="main",
        random_seed=123
    )
    ModelFactory.create_model(config=hf_create_config_with_name) 
    call_args_rerun = ModelFactory.HuggingFaceModel.call_args[0][0]
    assert call_args_rerun.model_name == "my_hf_model_explicit_name"

    ModelFactory.set_seed.assert_called_once_with(123) 
    ModelFactory.HuggingFaceModel.process_hf_identifier.assert_called_once_with("user/hf-model")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_model_huggingface_from_kwargs(mock_hf_dependencies): 
    """Test creating a Hugging Face model using kwargs."""
    with patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("kwarg_hf_model_name", MagicMock(name="MODEL_NAME"))) as _, \
         patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", return_value="kwarg_hf_model_name-processed") as _:

        created_model = ModelFactory.create_model(
            model_name="kwarg_hf_model_name", 
            architecture={"num_labels": 10},
            pretrained=False,
            revision="dev"
        )

        assert created_model == MOCK_HF_MODEL_INSTANCE
        ModelFactory.set_seed.assert_not_called() 
        
        call_args = ModelFactory.HuggingFaceModel.call_args[0][0]
        assert isinstance(call_args, HuggingFaceResolvedConfig)
        assert call_args.model_id == "kwarg_hf_model_name-processed"
        assert call_args.model_name == "kwarg_hf_model_name" 
        assert call_args.architecture == {"num_labels": 10}
        assert call_args.pretrained is False
        assert call_args.revision == "dev"
        ModelFactory.HuggingFaceModel.process_hf_identifier.assert_called_with("kwarg_hf_model_name")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.models.model_factory.HuggingFaceModel.process_hf_identifier", side_effect=ValueError("Failed to process identifier"))
@patch("advsecurenet.models.huggingface_model.HuggingFaceModel.verify_hf_identifier_exists", return_value=True)
@patch("advsecurenet.models.model_factory.ModelFactory._validate_create_model_config")
@patch("advsecurenet.models.model_factory.ModelFactory.infer_model_type", return_value=ModelType.HUGGINGFACE)
@patch("advsecurenet.models.model_factory.determine_identifier_and_soruce", return_value=("bad-hf-url", MagicMock(name="MODEL_IDENTIFIER")))
def test_create_model_huggingface_processing_error(
    mock_determine_id, 
    mock_infer_type, 
    mock_validate_config, 
    mock_verify_exists, 
    mock_process_hf_id
): 
    """Test error handling when HuggingFaceModel.process_hf_identifier fails."""
    hf_config = CreateModelConfig(model_name="bad-hf-url-name", model_identifier="bad-hf-url") # ADDED model_name
    
    with pytest.raises(ValueError, match="Error creating model. Please check the model_name and other arguments. Error: Failed to process identifier"):
        ModelFactory.create_model(config=hf_config)
