import re
from typing import Optional
from urllib.parse import urlparse


def is_huggingface_url(url: str) -> bool:
    """
    Checks if a string is a valid Hugging Face URL format.

    This function validates whether the provided URL points to a Hugging Face resource
    by checking the domain and path structure. It supports both full URLs with schemes
    and domain-only URLs, and handles common variations like www prefixes.

    Args:
        url (str): The URL string to validate. Can include or omit the protocol scheme.
                  Examples: "huggingface.co/user/repo", "https://hf.co/datasets/user/repo"

    Returns:
        bool: True if the URL matches Hugging Face URL format, False otherwise.
              Returns False for empty strings or None inputs.

    Note:
        - Accepts both "huggingface.co" and "hf.co" domains
        - Automatically adds "https://" scheme if missing for parsing
        - Strips "www." prefix during validation
        - Requires at least two path segments (user/repo structure)
        - Does not verify if the URL actually exists on the Hub

    Examples:
        >>> is_huggingface_url("huggingface.co/microsoft/DialoGPT-medium")
        True
        >>> is_huggingface_url("https://hf.co/datasets/squad")
        True
        >>> is_huggingface_url("www.huggingface.co/user/repo")
        True
        >>> is_huggingface_url("github.com/user/repo")
        False
        >>> is_huggingface_url("")
        False
    """
    if not url:
        return False
    # Ensure a scheme for urlparse to work properly
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    o = urlparse(url, allow_fragments=True)
    # Normalize and strip “www.”
    host = o.netloc.lower().removeprefix("www.")
    if host not in {"huggingface.co", "hf.co"}:
        return False
    # Split path into non‑empty segments
    parts = [segment for segment in o.path.split("/") if segment]
    # Require at least two segments: user/model
    return len(parts) >= 2


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
    # 1. Early exit if not a valid HF URL
    if not is_huggingface_url(url):
        return None

    # 2. Normalize scheme and parse
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    o = urlparse(url, allow_fragments=True)

    # 3. Split path into non-empty segments
    parts = [seg for seg in o.path.split("/") if seg]

    # 4. Handle optional 'datasets/' prefix
    if parts and parts[0] == "datasets":
        parts = parts[1:]

    # 5. Return 'user/model' if present
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None


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
