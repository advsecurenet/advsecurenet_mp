from typing import Dict, Iterable, Any

def pop_keys_from_dict(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """
    Returns a copy of the dictionary with specified keys removed.

    Args:
        d (dict): The original dictionary.
        keys (Iterable[str]): Keys to remove.

    Returns:
        dict: A new dictionary without the specified keys.
    """
    return {k: v for k, v in d.items() if k not in keys}