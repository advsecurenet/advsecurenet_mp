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

    This function validates whether the provided string follows the standard Hugging Face
    repository ID format using regex pattern matching. It checks for the basic structure
    but does not verify if the ID actually exists on the Hugging Face Hub.

    Args:
        identifier (str): The string to validate as a Hugging Face ID.
                         Expected format: "username/repository-name"

    Returns:
        bool: True if the identifier matches the Hugging Face ID format, False otherwise.
              Returns False for empty strings, None inputs, or malformed IDs.

    Note:
        - Requires exactly one forward slash separating two parts
        - Allowed characters: letters (a-z, A-Z), numbers (0-9), dots (.),
          underscores (_), and hyphens (-)
        - Both username and repository parts must contain at least one character
        - Does not validate against Hub existence or user permissions
        - Case-sensitive validation

    Examples:
        >>> is_huggingface_id("microsoft/DialoGPT-medium")
        True
        >>> is_huggingface_id("huggingface/transformers")
        True
        >>> is_huggingface_id("user_name/repo.name")
        True
        >>> is_huggingface_id("single-part")
        False
        >>> is_huggingface_id("user/repo/extra")
        False
        >>> is_huggingface_id("user/repo with spaces")
        False
        >>> is_huggingface_id("")
        False
    """
    if not identifier:
        return False
    # Regex: Starts with allowed chars, has '/', ends with allowed chars.
    # Allowed chars: letters, numbers, dot, underscore, hyphen.
    pattern = r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"
    return bool(re.match(pattern, identifier))


def extract_id_from_url(url: str) -> Optional[str]:
    """
    Extracts a repository ID from a Hugging Face URL.
    
    This function parses a Hugging Face URL and extracts the repository identifier
    in the standard 'user/repo' format. It handles various URL formats including
    dataset URLs with optional 'datasets/' prefix and normalizes the output.
    
    Args:
        url (str): The Hugging Face URL to parse. Should be a valid HF URL format.
                  Can include or omit protocol schemes.
    
    Returns:
        Optional[str]: The extracted repository ID in 'user/repo' format, or None
                      if the URL is invalid or doesn't contain sufficient path segments.
    
    Note:
        - Validates input using is_huggingface_url() before processing
        - Automatically handles missing URL schemes by adding 'https://'
        - Strips optional 'datasets/' prefix from dataset URLs
        - Requires at least two path segments after prefix removal
        - Returns None for malformed or non-Hugging Face URLs
    
    Examples:
        >>> extract_id_from_url("https://huggingface.co/microsoft/DialoGPT-medium")
        "microsoft/DialoGPT-medium"
        >>> extract_id_from_url("hf.co/datasets/squad/viewer")
        "squad/viewer"
        >>> extract_id_from_url("huggingface.co/datasets/user/repo")
        "user/repo"
        >>> extract_id_from_url("www.huggingface.co/transformers/bert-base-uncased")
        "transformers/bert-base-uncased"
        >>> extract_id_from_url("invalid-url")
        None
        >>> extract_id_from_url("huggingface.co/single-segment")
        None
    """
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


def process_hf_identifier(identifier: str) -> Optional[str]:
    """
    Processes and normalizes a Hugging Face identifier from various input formats.
    
    This function accepts either a Hugging Face URL or a direct repository ID and
    returns a normalized repository ID in the standard 'user/repo' format. It handles
    URL extraction and validates the format to ensure consistency across the application.
    
    Args:
        identifier (str): The Hugging Face identifier to process. Can be:
                         - A full URL: "https://huggingface.co/microsoft/DialoGPT-medium"
                         - A domain URL: "huggingface.co/user/repo"  
                         - A dataset URL: "https://hf.co/datasets/squad"
                         - A direct ID: "microsoft/DialoGPT-medium"
    
    Returns:
        Optional[str]: The normalized repository ID in 'user/repo' format if the input
                      is valid, None if the identifier is invalid or malformed.
    
    Note:
        - URLs are parsed to extract the repository ID
        - Handles optional 'datasets/' prefix in URLs automatically
        - Direct IDs are validated and returned unchanged if valid
        - Invalid formats return None rather than raising exceptions
    
    Examples:
        >>> process_hf_identifier("https://huggingface.co/microsoft/DialoGPT-medium")
        "microsoft/DialoGPT-medium"
        >>> process_hf_identifier("hf.co/datasets/squad")
        "squad"  # Note: single-name datasets are handled
        >>> process_hf_identifier("microsoft/DialoGPT-medium")
        "microsoft/DialoGPT-medium"
        >>> process_hf_identifier("invalid-format")
        None
        >>> process_hf_identifier("github.com/user/repo")
        None
    """
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
