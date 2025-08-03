from typing import Tuple

from advsecurenet.utils.huggingface_utils import huggingface_general_utils
import advsecurenet.shared.types.configs.model_config as model_config


def resolve_hf_identifiers(config: model_config.CreateModelConfig) -> Tuple[str, str]:
    """
    Determines the canonical Hugging Face model ID and model name from the config.

    Prioritizes `config.model_identifier`. If not present, uses `config.model_name`.
    Handles cases where identifiers are URLs or plain IDs.

    Args:
        config (CreateModelConfig): The configuration object.

    Returns:
        Str: A string containing the resolved identifier.

    Raises:
        ValueError: If an identifier is required but not found, or if a URL is provided but the ID cannot be extracted.
    """
    identifier_to_process, identifier_source_field_name = (
        model_config.determine_identifier_and_soruce(config)
    )

    final_model_id = huggingface_general_utils.process_hf_identifier(
        identifier_to_process
    )

    if final_model_id is None:
        # This means process_hf_identifier failed (likely due to URL extraction)
        raise ValueError(
            f"Could not extract model ID from URL in '{identifier_source_field_name.name.lower()}': {identifier_to_process}"
        )

    return final_model_id
