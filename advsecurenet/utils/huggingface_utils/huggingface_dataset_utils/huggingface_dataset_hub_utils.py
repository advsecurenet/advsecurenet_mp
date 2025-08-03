import warnings

from huggingface_hub import dataset_info

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    from huggingface_hub.utils._errors import RepositoryNotFoundError

import advsecurenet.utils.huggingface_utils.huggingface_general_utils as huggingface_general_utils


def check_hub_for_dataset_id(dataset_id: str) -> bool:
    """
    Checks if the given ID corresponds to a valid dataset on the Hugging Face Hub.

    Args:
        dataset_id: The Hugging Face ID to check (e.g., "squad", "glue", "username/my_dataset").

    Returns:
        True if the ID is a valid dataset on the Hub, False otherwise.
    """
    try:
        dataset_info(dataset_id)
        return True
    except RepositoryNotFoundError:
        # The ID is valid in format but does not correspond to any dataset on the hub
        return False


def verify_hf_dataset_identifier_exists(identifier: str) -> bool:
    """
    Verifies if a Hugging Face identifier (URL or dataset ID) corresponds
    to an existing dataset on the Hub. Validates format before checking.

    Args:
        identifier (str): The dataset identifier (e.g., "user/repo" or "https://huggingface.co/user/repo").

    Returns:
        bool: True if the identifier points to an existing dataset on the Hub, False otherwise.
                Returns False also if network errors occur during the check.
    """
    dataset_id_to_check = huggingface_general_utils.process_hf_identifier(identifier)

    if dataset_id_to_check is None:
        return False

    try:
        return check_hub_for_dataset_id(dataset_id_to_check)
    except Exception as e:
        # Treat Hub check errors (network, etc.) as "doesn't exist" for inference purposes
        warnings.warn(
            f"Could not verify Hugging Face identifier '{identifier}' due to Hub check error: {e}"
        )
        return False
