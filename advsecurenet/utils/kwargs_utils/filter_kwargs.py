import inspect
import warnings
from typing import Callable, Dict, Any

def filter_kwargs_for_callable(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters a kwargs dictionary to only include arguments accepted by the callable's signature.
    Warns for each ignored argument.

    Args:
        func (Callable): The function or class to inspect.
        kwargs (Dict[str, Any]): The keyword arguments to filter.

    Returns:
        Dict[str, Any]: Filtered kwargs that are valid for the callable.
    """
    sig = inspect.signature(func)
    valid_args = set(sig.parameters.keys())
    valid_args.discard('self')
    filtered = {}
    for k, v in kwargs.items():
        if k in valid_args:
            filtered[k] = v
        else:
            warnings.warn(f"Ignoring argument '{k}' as it is not accepted by {func.__name__}.")
    return filtered