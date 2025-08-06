import logging
from enum import EnumMeta
from typing import Optional, Dict, Any
import dataclasses

from torch import nn

from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.custom_model import CustomModel
from advsecurenet.models.external_model import ExternalModel
from advsecurenet.models.standard_model import StandardModel
from advsecurenet.models.huggingface_model import HuggingFaceModel
from advsecurenet.shared.types.configs.model_config import (
    CreateModelConfig,
    CustomModelConfig,
    ExternalModelConfig,
    StandardModelConfig,
    HuggingFaceResolvedConfig,
    determine_identifier_and_source,
)
from advsecurenet.shared.types.model import ModelType
from advsecurenet.utils.reproducibility_utils import set_seed
from advsecurenet.utils.huggingface_utils import (
    huggingface_model_utils,
    huggingface_general_utils,
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    This class is a factory class for creating models. It provides a single interface for creating models. It supports both standard models and custom models.
    """

    @staticmethod
    def _resolve_config_and_warn(
        config: Optional[CreateModelConfig], kwargs: Dict[str, Any]
    ) -> CreateModelConfig:
        """
        Resolves the final CreateModelConfig by handling None config, merging kwargs,
        and issuing a warning on overlap. Prioritizes kwargs values.

        Args:
            config: The initially provided config object (or None).
            kwargs: Keyword arguments passed to create_model.

        Returns:
            The definitive CreateModelConfig instance.
        """
        if config is None or not isinstance(config, CreateModelConfig):
            # Case 1: No valid config provided, create purely from kwargs
            logger.debug("No valid model config provided, creating from kwargs.")
            return CreateModelConfig(**kwargs)
        elif not kwargs:
            # Case 2: Valid config provided, and kwargs is empty. Use the provided config directly.
            logger.debug(
                "Valid CreateModelConfig provided and kwargs is empty. Using provided config directly."
            )
            return config
        else:
            # Case 3: Config and kwargs are provided. Check for overlap and merge config and kwargs.
            overlapping_keys = []
            config_field_names = {f.name for f in dataclasses.fields(CreateModelConfig)}
            kwargs_to_merge = {}  # Only store kwargs that are actual config fields

            for key, kwarg_value in kwargs.items():
                if key in config_field_names:
                    kwargs_to_merge[key] = kwarg_value  # Prepare for merge
                    config_value = getattr(
                        config, key, None
                    )  # Safely get current value
                    if kwarg_value != config_value:
                        overlapping_keys.append(key)

            if overlapping_keys:
                keys_str = ", ".join(f"'{k}'" for k in overlapping_keys)
                warning_msg = (
                    f"Overlap detected between provided 'config' object and keyword arguments "
                    f"for keys: [{keys_str}]. Values from keyword arguments will be prioritized."
                )
                logger.warning(warning_msg)
                # Merge kwargs into the config (prioritizing kwargs)
                config_dict = dataclasses.asdict(config)
                config_dict.update(kwargs_to_merge)
                logger.debug("Merging overlapping kwargs into provided config.")
                return CreateModelConfig(**config_dict)
            else:
                # No overlap affecting config fields, return original config
                # (kwargs might still contain non-config extras, but they won't be used later)
                logger.debug("Provided config used, no overlapping kwargs detected.")
                return config

    @staticmethod
    def infer_model_type(model_name: str) -> ModelType:
        """
        This function infers the model type based on the model_name.

        Parameters
        ----------
        model_name: str
            The name of the model to be loaded. For example, 'resnet18' or 'CustomMnistModel'.

        Returns
        -------
        ModelType
            The model type of the model_name.

        Raises
        ------
        ValueError
            If the model_name is not supported by torchvision or is not a custom model.
        """
        if model_name in StandardModel.models():
            return ModelType.STANDARD

        if model_name in CustomModel.models():
            return ModelType.CUSTOM

        if huggingface_model_utils.verify_hf_model_identifier_exists(model_name):
            return ModelType.HUGGINGFACE

        raise ValueError(
            "Unsupported model. If you are trying to load an external model, please set is_external=True in the CreateModelConfig."
        )

    @staticmethod
    def create_model(config: Optional[CreateModelConfig] = None, **kwargs) -> BaseModel:
        """
        This function creates a model based on the CreateModelConfig.

        Args:
            config (Optional[CreateModelConfig]): The configuration for creating the model. If not provided, the model will be created with the passed keyword arguments.
            CreateModelConfig contains the following fields:
                - model_name: str
                - architecture: dict
                - pretrained: Optional[bool] = True
                - weights: Optional[str] = "IMAGENET1K_V1"
                - custom_models_path: Optional[str] = "CustomModels"
                - model_arch_path: Optional[str] = None
                - model_weights_path: Optional[str] = None
                - is_external: bool = False
                - random_seed: Optional[int] = None

            **kwargs: Additional keyword arguments to be passed to the model constructor.

        Returns:
            BaseModel: The created model.

        Note:
            If the model is a custom model, the model_name should be the name of the custom model class. For example, 'CustomMnistModel'.
            You can use your external model by setting is_external=True in the CreateModelConfig and providing the model_arch_path and model_weights_path.
        """
        try:
            resolved_config = ModelFactory._resolve_config_and_warn(config, kwargs)

            if resolved_config.is_external:
                cfg = ExternalModelConfig(
                    model_name=resolved_config.model_name,
                    model_arch_path=resolved_config.model_arch_path,
                    pretrained=resolved_config.pretrained,
                    model_weights_path=resolved_config.model_weights_path,
                    architecture=resolved_config.architecture,
                )
                return ExternalModel(cfg)

            identifier, _ = determine_identifier_and_source(resolved_config)

            inferred_type: ModelType = ModelFactory.infer_model_type(identifier)

            ModelFactory._validate_create_model_config(inferred_type, resolved_config)

            if resolved_config.random_seed is not None:
                set_seed(resolved_config.random_seed)

            if inferred_type == ModelType.STANDARD:
                cfg = StandardModelConfig(
                    model_name=resolved_config.model_name,
                    pretrained=resolved_config.pretrained,
                    weights=resolved_config.weights,
                    architecture=resolved_config.architecture,
                )
                return StandardModel(cfg)

            if inferred_type == ModelType.CUSTOM:
                # The custom model name would typically be without the 'Custom' prefix for the filename.
                # For example: 'MnistModel' for 'CustomMnistModel.py'. Adjust as necessary.
                cfg = CustomModelConfig(
                    model_name=resolved_config.model_name,
                    custom_models_path=resolved_config.custom_models_path,
                    pretrained=resolved_config.pretrained,
                    architecture=resolved_config.architecture,
                )
                return CustomModel(cfg)

            if inferred_type == ModelType.HUGGINGFACE:
                model_id = huggingface_general_utils.process_hf_identifier(identifier)

                cfg = HuggingFaceResolvedConfig(
                    model_name=resolved_config.model_name,
                    architecture=resolved_config.architecture,
                    pretrained=resolved_config.pretrained,
                    model_id=model_id,
                    revision=resolved_config.revision,
                    cache_dir=resolved_config.cache_dir,
                    trust_remote_code=resolved_config.trust_remote_code,
                    model_class_name=resolved_config.model_class_name,
                )
                return HuggingFaceModel(cfg)

        except Exception as e:
            err = f"Error creating model. Please check the model_name and other arguments. Error: {str(e)}"
            logger.error(err)
            raise ValueError(err) from e

    @staticmethod
    def _validate_create_model_config(
        inferred_type: ModelType, config: CreateModelConfig
    ):
        """
        This function validates the CreateModelConfig based on the inferred model type.
        """
        if inferred_type == ModelType.CUSTOM and config.pretrained:
            raise ValueError(
                "Custom models do not support pretrained weights. Instead, you can load the weights after loading the model."
            )

        if (
            inferred_type == ModelType.STANDARD
            and config.pretrained
            and config.random_seed is not None
        ):
            raise ValueError(
                "Pretrained standard models do not support random seed. They already have a fixed set of weights :)"
            )

    @staticmethod
    def available_models() -> list[str]:
        """
        Returns a list of all available models.
        """
        return StandardModel.models() + CustomModel.models()

    @staticmethod
    def available_standard_models() -> list[str]:
        """
        Returns a list of all available standard models that are supported by torchvision.
        """
        return StandardModel.models()

    @staticmethod
    def available_custom_models() -> list[str]:
        """
        Returns a list of all available custom models that are created by the user. These models are stored in the 'advsecurenet/models/CustomModels' directory.
        """
        return CustomModel.models()

    @staticmethod
    def available_weights(model_name: str) -> EnumMeta:
        """
        Returns a list of available weights for the given model_name.

        Args:
            model_name (str): The name of the model. You can get the list of available models using StandardModel.models().

        Returns:
            EnumMeta: A EnumMeta object containing the available weights for the given model_name.

        Raises:
            ValueError: If the model_name is not supported.

        Note:
            You can get the list of available weights for a model using list(StandardModel.available_weights(model_name)).
            This is only applicable for standard models.

        Raises:
            ValueError: If the model_name is not supported.
            ValueError: If the model_name is a custom model.

        Examples:
            >>> ModelFactory.available_weights("resnet18")
            <enum 'ResNet18Weights'>
            >>> list(ModelFactory.available_weights("resnet18"))
            [ResNet18_Weights.IMAGENET1K_V1]

        """
        inferred_type: ModelType = ModelFactory.infer_model_type(model_name)
        if inferred_type == ModelType.CUSTOM:
            raise ValueError(
                "Custom models do not support pretrained weights. Instead, you can load the weights after loading the model."
            )
        return StandardModel.available_weights(model_name)

    @staticmethod
    def add_layer(
        model: nn.Module, new_layer: nn.Module, position: int = -1
    ) -> nn.Module:
        """
        Inserts a new layer into an existing PyTorch model at the specified position. If the model is not a Sequential model,
        it will be converted into one.

        Args:
            model (nn.Module): The original model to which the new layer will be added.
            new_layer (nn.Module): The layer to be inserted into the model.
            position (int): The position at which to insert the new layer. If set to -1, the layer is added at the end.
                            Positions are zero-indexed.

        Returns:
            nn.Module: A new model with the layer added at the specified position.

        Raises:
            ValueError: If the specified position is out of bounds.
        """
        # Convert non-Sequential models to Sequential if necessary
        if not isinstance(model, nn.Sequential):
            model = nn.Sequential(model)

        # Prepare the list of existing layers
        layers = list(model.children())

        # Check position validity
        if position < -1 or position > len(layers):
            raise ValueError("Position out of bounds.")

        # Insert the new layer at the specified position or append at the end
        if position == -1 or position == len(layers):
            layers.append(new_layer)
        else:
            layers.insert(position, new_layer)

        # Create a new Sequential model with the updated list of layers
        updated_model = nn.Sequential(*layers)

        return updated_model
