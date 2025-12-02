from typing import Dict, Any, Union, Callable


def map_kwargs(
    kwargs: Dict[str, Any],
    mapping: Dict[str, Union[str, Callable[[Dict[str, Any]], Dict[str, Any]]]],
) -> Dict[str, Any]:
    """
    Processes a dictionary of kwargs based on a flexible mapping.

    The mapping can specify two types of operations:
    1.  **Simple Key Rename**: If the value in the mapping is a string, the key will be renamed.
        Example: `{'old_key': 'new_key'}`
    2.  **Custom Transformation**: If the value is a callable, it will be executed with the
        entire kwargs dictionary, and is expected to return the transformed dictionary.
        Example: `{'key_to_process': my_transform_function}`

    Args:
        kwargs (Dict[str, Any]): The original keyword arguments.
        mapping (Dict): The mapping rules for processing.

    Returns:
        Dict[str, Any]: The processed keyword arguments.
    """
    processed = dict(kwargs)  # Work on a copy

    for key_to_process, rule in mapping.items():
        if key_to_process in processed:
            if isinstance(rule, str):  # Simple rename
                value = processed.pop(key_to_process)
                processed[rule] = value
            elif callable(rule):  # Custom transformation function
                processed = rule(processed)

    return processed
