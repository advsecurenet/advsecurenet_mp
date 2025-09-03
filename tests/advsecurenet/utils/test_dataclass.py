from dataclasses import dataclass
from typing import List, Optional, Union, Dict, TypeVar, Generic

import pytest

from advsecurenet.utils.dataclass import (
    filter_for_dataclass,
    flatten_dataclass,
    is_list_of_dataclass,
    is_optional_type,
    merge_dataclasses,
    process_field,
    process_generic_type,
    process_optional_field,
    recursive_dataclass_instantiation,
    is_dict_of_dataclass,
    _instantiate_dict_of_dataclasses,
)


@dataclass
class Sample:
    value: int


@dataclass
class NestedSample:
    sample: Sample


@dataclass
class ListSample:
    samples: List[Sample]


@dataclass
class DictSample:
    samples: Dict[str, Sample]


# Define a generic dataclass for testing


@dataclass
class GenericSample:
    field: Sample


T = TypeVar("T")


@dataclass
class RealGenericSample(Generic[T]):
    field: T


@dataclass
class Nested:
    value: int


@dataclass
class Example:
    a: int
    b: str
    c: Nested
    d: Optional[int] = None
    e: Optional[Nested] = None


@dataclass
class AnotherExample:
    a: int
    b: str
    f: float


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_flatten_dataclass():
    instance = Example(a=1, b="test", c=Nested(value=10))
    flattened = flatten_dataclass(instance)
    assert flattened == {"a": 1, "b": "test", "c": {"value": 10}, "d": None, "e": None}


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_filter_for_dataclass():
    data = {"a": 1, "b": "test", "c": {"value": 10}, "extra_field": "extra"}
    filtered = filter_for_dataclass(data, Example)
    assert filtered == {"a": 1, "b": "test", "c": {"value": 10}}


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_filter_for_dataclass_with_convert():
    data = {"a": 1, "b": "test", "c": {"value": 10}, "extra_field": "extra"}
    filtered = filter_for_dataclass(data, Example, convert=True)
    assert isinstance(filtered, Example)
    assert filtered.a == 1
    assert filtered.b == "test"
    assert filtered.c.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_recursive_dataclass_instantiation():
    data = {"a": 1, "b": "test", "c": {"value": 10}}
    instance = recursive_dataclass_instantiation(Example, data)
    assert isinstance(instance, Example)
    assert instance.a == 1
    assert instance.b == "test"
    assert instance.c.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_merge_dataclasses():
    dataclass1 = Example(a=1, b="test", c=Nested(value=10))
    dataclass2 = AnotherExample(a=2, b="updated", f=3.14)
    merged = merge_dataclasses(dataclass1, dataclass2)
    assert isinstance(merged, Example)
    assert merged.a == 2
    assert merged.b == "updated"
    assert merged.c.value == 10
    assert not hasattr(merged, "f")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_merge_dataclasses_with_optional():
    dataclass1 = Example(a=1, b="test", c=Nested(value=10), d=5)
    dataclass2 = Example(a=2, b="updated", c=Nested(value=20), e=Nested(value=30))
    merged = merge_dataclasses(dataclass1, dataclass2)
    assert isinstance(merged, Example)
    assert merged.a == 2
    assert merged.b == "updated"
    assert merged.c.value == 20
    assert merged.e.value == 30


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_is_optional_type():
    assert is_optional_type(Union[int, None]) is True
    assert is_optional_type(int) is False


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_optional_field():
    data = {"value": 10}
    result = process_optional_field((Sample, type(None)), data)
    assert isinstance(result, Sample)
    assert result.value == 10

    result = process_optional_field((int, type(None)), 5)
    assert result == 5


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_is_list_of_dataclass():
    assert is_list_of_dataclass(List[Sample], [Sample(1)]) is True
    assert is_list_of_dataclass(List[int], [1, 2, 3]) is False


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_field():
    # Test optional field
    data = {"value": 10}
    result = process_field(Union[Sample, None], data)
    assert isinstance(result, Sample)
    assert result.value == 10

    # Test dataclass field
    result = process_field(Sample, data)
    assert isinstance(result, Sample)
    assert result.value == 10

    # Test list of dataclass field
    data = [{"value": 10}, {"value": 20}]
    result = process_field(List[Sample], data)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, Sample) for item in result)

    # Test generic type
    data = {"field": {"value": 10}}
    result = process_field(GenericSample, data)
    assert isinstance(result, GenericSample)
    assert isinstance(result.field, Sample)
    assert result.field.value == 10

    data = {"a": {"value": 10}, "b": {"value": 20}}
    result = process_field(Dict[str, Sample], data)
    assert isinstance(result, dict)
    assert isinstance(result["a"], Sample)
    assert result["b"].value == 20


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_generic_type():
    data = {"field": {"value": 10}}
    result = process_generic_type(GenericSample, (Sample,), data)
    assert isinstance(result, GenericSample)
    assert isinstance(result.field, Sample)
    assert result.field.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_is_dict_of_dataclass():
    # Valid case
    assert is_dict_of_dataclass(Dict[str, Sample], {"key": {"value": 1}}) is True
    # Invalid: value is not a dict
    assert is_dict_of_dataclass(Dict[str, Sample], [{"value": 1}]) is False
    # Invalid: field_type is not a dict
    assert is_dict_of_dataclass(List[Sample], {"key": {"value": 1}}) is False
    # Invalid: value type in Dict is not a dataclass
    assert is_dict_of_dataclass(Dict[str, int], {"key": 1}) is False
    # Invalid: malformed Dict hint (use raw Dict to test the arg count check)
    assert is_dict_of_dataclass(Dict, {"key": "value"}) is False


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_instantiate_dict_of_dataclasses():
    data = {"a": {"value": 10}, "b": {"value": 20}}
    result = _instantiate_dict_of_dataclasses(Sample, data)
    assert isinstance(result, dict)
    assert "a" in result and "b" in result
    assert isinstance(result["a"], Sample)
    assert result["a"].value == 10
    assert isinstance(result["b"], Sample)
    assert result["b"].value == 20


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_flatten_dataclass_non_dataclass_input():
    """
    Covers: `flatten_dataclass` -> `return instance`
    """
    # Test with a non-dataclass input
    data = {"a": 1}
    result = flatten_dataclass(data)
    assert result == data


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_filter_for_dataclass_with_dataclass_input():
    """
    Covers: `filter_for_dataclass` -> `data = flatten_dataclass(data)`
    """
    # Test with a dataclass instance as input
    instance = Example(a=1, b="test", c=Nested(value=10))
    filtered = filter_for_dataclass(instance, AnotherExample)
    assert filtered == {"a": 1, "b": "test"}


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_recursive_instantiation_non_dataclass_cls():
    """
    Covers: `recursive_dataclass_instantiation` -> `return data`
    """
    # Test with a non-dataclass class
    data = {"a": 1}
    result = recursive_dataclass_instantiation(dict, data)
    assert result == data


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_field_dict_of_dataclass():
    """
    Covers: `process_field` -> `dataclass_value_type = ...` and `return _instantiate_dict_of_dataclasses(...)`
    """
    # Test processing a field that is a Dict of dataclasses
    field_type = Dict[str, Sample]
    value = {"a": {"value": 10}, "b": {"value": 20}}
    result = process_field(field_type, value)
    assert isinstance(result, dict)
    assert isinstance(result["a"], Sample)
    assert result["b"].value == 20


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_generic_type_with_none_value():
    """
    Covers: `process_generic_type` -> `value[additional_key] = value.get(additional_key)`
    """
    # Test with a value that is None for a field in the generic type
    data = {"field": None}
    result = process_generic_type(GenericSample, (Sample,), data)
    assert result.field is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_merge_dataclasses_single_item():
    """
    Covers: `merge_dataclasses` -> `return dataclasses[0]`
    """
    # Test merging a single dataclass
    instance = Example(a=1, b="test", c=Nested(value=10))
    result = merge_dataclasses(instance)
    assert result == instance


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_field_with_true_generic():
    """
    Covers: `process_field` -> `return process_generic_type(origin, args, value)`
    """
    # Test with a true generic dataclass to hit the generic processing path
    field_type = RealGenericSample[Sample]
    value = {"field": {"value": 10}}
    result = process_field(field_type, value)
    assert isinstance(result, RealGenericSample)
    assert isinstance(result.field, Sample)
    assert result.field.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_field_with_plain_dataclass():
    """
    Covers: `process_field` -> `return recursive_dataclass_instantiation(origin, value)`
    """
    # Test with a plain dataclass to hit the standard recursive instantiation
    field_type = Sample
    value = {"value": 10}
    result = process_field(field_type, value)
    assert isinstance(result, Sample)
    assert result.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_process_field_with_generic_dataclass_and_primitive_type():
    """
    Covers: `process_field` -> `elif is_dataclass(origin): ...`
    This test uses a generic dataclass with a non-dataclass type argument (int).
    This bypasses the earlier `is_dataclass(args[0])` check and falls through
    to the target `is_dataclass(origin)` check.
    """
    # The field_type is a generic alias of a dataclass with a primitive type
    field_type = RealGenericSample[int]
    value = {"field": 123}

    result = process_field(field_type, value)

    assert isinstance(result, RealGenericSample)
    assert result.field == 123


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_merge_dataclasses_with_non_dataclass_item():
    """
    Covers: `merge_dataclasses` -> `continue`
    """
    # Arrange
    dataclass1 = Example(a=1, b="test", c=Nested(value=10))
    # This non-dataclass item should be skipped by the 'continue' statement
    non_dataclass_item = {"b": "should be ignored"}
    dataclass2 = AnotherExample(a=2, b="updated", f=3.14)

    # Act
    merged = merge_dataclasses(dataclass1, non_dataclass_item, dataclass2)

    # Assert
    # The final type is based on the first argument
    assert isinstance(merged, Example)
    # The values from dataclass2 should overwrite dataclass1
    assert merged.a == 2
    assert merged.b == "updated"
    # The non_dataclass_item was correctly ignored
    assert merged.c.value == 10
