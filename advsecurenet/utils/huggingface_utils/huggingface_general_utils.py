import re
from typing import Optional


def is_huggingface_url(url: str) -> bool:
    """
    Check if a URL is a Hugging Face URL.

     Args:
        url (str): The URL to check.

    Returns:
         bool: True if the URL is a Hugging Face URL, False otherwise.
    """
    if not url:
        return False

    pattern = r"^(https?://) ?(www\.)?(huggingface\.co|hf\.co)/([^/]+/[^/]+).*$"
    return bool(re.match(pattern, url))


def is_huggingface_id(identifier: str) -> bool:
    """
    Checks if a string matches the typical Hugging Face ID format (e.g., 'user/repo').
    Uses regex for basic format validation, does not check Hub existence.
    """
    if not identifier:
        return False
    # Regex: Starts with allowed chars, has '/', ends with allowed chars.
    # Allowed chars: letters, numbers, dot, underscore, hyphen.
    pattern = r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"
    return bool(re.match(pattern, identifier))


def extract_id_from_url(url: str) -> Optional[str]:
    """
    Extract the ID from a Hugging Face URL.

    Args:
        url (str): The URL to extract the ID from.

    Returns:
        Optional[str]: The ID if the URL is a valid Hugging Face URL, None otherwise.
    """
    if not is_huggingface_url(url):
        return None

    pattern = r"^(https?://)?(www\.)?(huggingface\.co|hf\.co)/(?:datasets/)?([^/]+/[^/]+)(?:[/?#].*)?$"
    match = re.match(pattern, url)
    if match:
        return match.group(4)

    return None


def process_hf_identifier(identifier):
    # Process the chosen identifier
    if is_huggingface_url(identifier):
        extracted_id = extract_id_from_url(identifier)
        # If extraction fails, extracted_id will be None or empty, so return it directl
        return extracted_id
    elif is_huggingface_id(identifier):
        # If it's a valid ID, return it directly
        return identifier
    else:
        # If it's neither, return None
        return None
