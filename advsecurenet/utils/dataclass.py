from dataclasses import fields, is_dataclass
from typing import Optional, Type, TypeVar, Union, Dict, Any, get_args, get_origin

# This is needed to support recursive dataclass instantiation
T = TypeVar("T")


def flatten_dataclass(instance: object) -> dict:
    """
    Recursively flatten dataclass instances into a single dictionary. Recursion is used to flatten nested dataclasses.

    Args:
        instance (object): The dataclass instance to flatten.

    Returns:
        dict: The flattened dataclass instance.
    """
    if not is_dataclass(instance):
        return instance

    result = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if is_dataclass(value):
            result[field.name] = flatten_dataclass(value)
        else:
            result[field.name] = value
    return result


def filter_for_dataclass(
    data: Union[dict, object], dataclass_type: type, convert: Optional[bool] = False
) -> Union[dict, object]:
    """
    Filter a dictionary to only include keys that are valid fields of the given dataclass type.

    Args:
        data (Union[dict, dataclass]): The data to filter. If a dataclass instance is provided, it will be flattened first.
        dataclass_type (type): The dataclass type to filter for.
        convert (Optional[bool]): Whether to convert the filtered data back to a dataclass instance. Default is False.

    Returns:
        dict or object: The filtered data. If the convert flag is set to True, the filtered data will be converted to a dataclass instance.
    """
    if is_dataclass(data):
        data = flatten_dataclass(data)
    valid_keys = {field.name for field in fields(dataclass_type)}
    filtered_data = {key: value for key, value in data.items() if key in valid_keys}
    if convert:
        return recursive_dataclass_instantiation(dataclass_type, filtered_data)
    return filtered_data


def recursive_dataclass_instantiation(cls: Type[T], data: dict) -> T:
    """
    Recursively instantiate a dataclass from a dictionary. Recursion is used to instantiate nested dataclasses.

    Args:
        cls (Type[T]): The dataclass type to instantiate.
        data (dict): The dictionary to instantiate the dataclass from.

    Returns:
        T: The instantiated dataclass.
    """
    if not is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    new_data = {
        key: process_field(field_types[key], value)
        for key, value in data.items()
        if key in field_types
    }

    return cls(**new_data)


def process_field(field_type: Type, value):
    origin = get_origin(field_type)
    args = get_args(field_type)

    if is_optional_type(field_type):
        return process_optional_field(args, value)
    elif is_dataclass(field_type) and isinstance(value, dict):
        return recursive_dataclass_instantiation(field_type, value)
    elif is_list_of_dataclass(field_type, value):
        return [recursive_dataclass_instantiation(args[0], item) for item in value]
    elif is_dict_of_dataclass(field_type, value):
        dataclass_value_type = get_args(field_type)[1]
        return _instantiate_dict_of_dataclasses(dataclass_value_type, value)
    elif origin and args and is_dataclass(args[0]) and isinstance(value, dict):
        return process_generic_type(origin, args, value)
    elif is_dataclass(origin):
        return recursive_dataclass_instantiation(origin, value)
    else:
        return value


def is_optional_type(field_type: Type) -> bool:
    origin = get_origin(field_type)
    args = get_args(field_type)
    return origin is Union and type(None) in args


def process_optional_field(args, value):
    actual_type = next(arg for arg in args if arg is not type(None))
    return process_field(actual_type, value)


def is_list_of_dataclass(field_type: Type, value) -> bool:
    origin = get_origin(field_type)
    args = get_args(field_type)
    return origin is list and is_dataclass(args[0]) and isinstance(value, list)


def is_dict_of_dataclass(field_type: Type, value) -> bool:
    """
    Checks if a field type is a Dictionary of dataclasses and the value is a dictionary.
    e.g. Dict[str, MyDataclass]
    """
    origin = get_origin(field_type)
    if origin is not dict or not isinstance(value, dict):
        return False

    args = get_args(field_type)
    # A valid Dict hint must have two arguments, e.g., Dict[key_type, value_type]
    if len(args) != 2:
        return False

    value_type = args[1]
    return is_dataclass(value_type)


def process_generic_type(origin, args, value):
    additional_fields = {field.name: arg for field, arg in zip(fields(origin), args)}
    for additional_key, additional_type in additional_fields.items():
        if additional_key in value and isinstance(value[additional_key], dict):
            value[additional_key] = recursive_dataclass_instantiation(
                additional_type, value[additional_key]
            )
        else:
            value[additional_key] = value.get(additional_key)
    return recursive_dataclass_instantiation(origin, value)


def merge_dataclasses(*dataclasses: object) -> object:
    """
    Merge two dataclasses into a single dataclass. The fields
    of the second dataclass will overwrite the fields of the first dataclass.

    Args:
        dataclasses (object): The dataclasses to merge.
    Returns:
        object: The merged dataclass.

    """

    if len(dataclasses) == 1:
        return dataclasses[0]

    flattened_data = {}
    for current_dataclass in dataclasses:
        if not is_dataclass(current_dataclass):
            continue
        flattened_data.update(flatten_dataclass(current_dataclass))

    return recursive_dataclass_instantiation(type(dataclasses[0]), flattened_data)


def _instantiate_dict_of_dataclasses(
    dataclass_type: Type, value_dict: Dict[Any, Any]
) -> Dict[Any, Any]:
    """
    Instantiates values of a dictionary that are expected to be dataclasses.

    Args:
        dataclass_type: The dataclass type to which the dictionary values should be instantiated.
        value_dict: The dictionary containing raw data for the dataclasses.

    Returns:
        A new dictionary with its values instantiated as dataclasses.
    """
    return {
        key: recursive_dataclass_instantiation(dataclass_type, val)
        for key, val in value_dict.items()
    }
