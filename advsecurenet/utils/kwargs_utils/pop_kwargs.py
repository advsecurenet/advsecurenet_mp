from typing import Dict, Iterable, Any
import copy


def pop_string_keys_from_dict(original_dictionary: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """
    Returns a deep copy of the dictionary with specified keys removed.
    
    Supports both top-level keys and nested key removal using dot notation.
    Uses recursion to handle unlimited nesting depth.

    Args:
        original_dictionary (Dict[str, Any]): The original dictionary.
        keys (Iterable[str]): Keys to remove. Supports dot notation for nested keys.
                             Examples: ["key1", "nested.subkey", "deep.nested.key"]

    Returns:
        Dict[str, Any]: A new deep copy of the dictionary without the specified keys.
                       All nested objects are recursively copied, ensuring complete
                       independence from the original dictionary.

    Note:
        - Keys are converted to a set for O(1) lookup performance
        - Supports dot notation for nested key removal (e.g., "config.model.name")
        - Uses recursion to handle unlimited nesting depth
        - If a nested key path doesn't exist, it's silently ignored

    Examples:
        >>> original = {
        ...     "a": 1, 
        ...     "config": {
        ...         "model": {"name": "bert", "deep": {"nested": {"key": "value"}}}
        ...     }
        ... }
        >>> result = pop_string_keys_from_dict(original, ["config.model.deep.nested.key"])
        >>> # Removes the deeply nested key regardless of depth
    """
    # Convert keys to set for O(1) lookup performance
    keys_to_remove = set(keys)
    
    # Deep copy the entire dictionary first
    result = copy.deepcopy(original_dictionary)
    
    # Separate top-level keys from nested keys
    top_level_keys = {key for key in keys_to_remove if '.' not in key}
    nested_keys = {key for key in keys_to_remove if '.' in key}
    
    # Remove top-level keys
    for key in top_level_keys:
        result.pop(key, None)
    
    # Remove nested keys using recursion
    for nested_key in nested_keys:
        _remove_nested_key_recursive(result, nested_key.split('.'))
    
    return result


def _remove_nested_key_recursive(dictionary: Dict[str, Any], key_path: list) -> None:
    """
    Recursively removes a nested key from a dictionary.
    
    Args:
        dictionary (Dict[str, Any]): The dictionary to modify in-place.
        key_path (list): List of keys representing the path to the target key.
    """
    if not key_path or not isinstance(dictionary, dict):
        return
    
    current_key = key_path[0]
    
    # Base case: we're at the target key
    if len(key_path) == 1:
        dictionary.pop(current_key, None)
        return
    
    # Recursive case: go deeper
    if current_key in dictionary and isinstance(dictionary[current_key], dict):
        _remove_nested_key_recursive(dictionary[current_key], key_path[1:])