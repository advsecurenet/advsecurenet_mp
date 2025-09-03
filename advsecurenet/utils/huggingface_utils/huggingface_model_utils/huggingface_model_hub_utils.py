import warnings

from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

import advsecurenet.utils.huggingface_utils.huggingface_general_utils as huggingface_general_utils


def check_hub_for_model_id(model_id: str) -> bool:
    """
    Internal helper: Checks if a specific model ID exists on the Hub.
    Assumes model_id is already validated for the correct format (e.g., "user/repo").

    Returns:
        bool: True if the model exists, False if specifically not found.
    Raises:
        Exception: Propagates exceptions (network errors, etc.) from model_info.
    """
    try:
        model_info(model_id)
        return True
    except RepositoryNotFoundError:
        # Model ID specifically not found on the Hub
        return False


def verify_hf_model_identifier_exists(identifier: str) -> bool:
    """
    Verifies if a Hugging Face identifier (URL or model ID) corresponds
    to an existing model on the Hub. Validates format before checking.

    Args:
        identifier (str): The model identifier (e.g., "user/repo" or "https://huggingface.co/user/repo").

    Returns:
        bool: True if the identifier points to an existing model on the Hub, False otherwise.
                Returns False also if network errors occur during the check.
    """
    model_id_to_check = huggingface_general_utils.process_hf_identifier(identifier)

    if model_id_to_check is None:
        return False

    try:
        return check_hub_for_model_id(model_id_to_check)
    except Exception as e:
        # Treat Hub check errors (network, etc.) as "doesn't exist" for inference purposes
        warnings.warn(
            f"Could not verify Hugging Face identifier '{identifier}' due to Hub check error: {e}"
        )
        return False
