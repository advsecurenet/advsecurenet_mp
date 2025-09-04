"""
Tests for advsecurenet.utils.kwargs_utils.pop_kwargs module.
"""

import pytest
from advsecurenet.utils.kwargs_utils.pop_kwargs import (
    pop_string_keys_from_dict,
    _remove_nested_key_recursive,
)


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestPopStringKeysFromDict:
    """Test class for pop_string_keys_from_dict function."""

    def test_pop_simple_top_level_keys(self):
        """Test removing simple top-level keys."""
        original = {"a": 1, "b": 2, "c": 3}
        result = pop_string_keys_from_dict(original, ["a", "c"])
        
        expected = {"b": 2}
        assert result == expected
        # Verify original is unchanged
        assert original == {"a": 1, "b": 2, "c": 3}

    def test_pop_single_key(self):
        """Test removing a single key."""
        original = {"key1": "value1", "key2": "value2"}
        result = pop_string_keys_from_dict(original, ["key1"])
        
        expected = {"key2": "value2"}
        assert result == expected

    def test_pop_nonexistent_keys(self):
        """Test removing keys that don't exist (should be silently ignored)."""
        original = {"a": 1, "b": 2}
        result = pop_string_keys_from_dict(original, ["nonexistent", "also_missing"])
        
        expected = {"a": 1, "b": 2}
        assert result == expected

    def test_pop_empty_keys_list(self):
        """Test with empty keys list."""
        original = {"a": 1, "b": 2}
        result = pop_string_keys_from_dict(original, [])
        
        expected = {"a": 1, "b": 2}
        assert result == expected

    def test_pop_nested_keys_dot_notation(self):
        """Test removing nested keys using dot notation."""
        original = {
            "a": 1,
            "config": {
                "model": {"name": "bert", "type": "classifier"},
                "training": {"epochs": 10}
            }
        }
        result = pop_string_keys_from_dict(original, ["config.model.name", "a"])
        
        expected = {
            "config": {
                "model": {"type": "classifier"},
                "training": {"epochs": 10}
            }
        }
        assert result == expected

    def test_pop_deeply_nested_keys(self):
        """Test removing deeply nested keys."""
        original = {
            "deep": {
                "nested": {
                    "structure": {
                        "with": {
                            "many": {
                                "levels": "target_value",
                                "keep": "this_value"
                            }
                        }
                    }
                }
            },
            "top": "level"
        }
        result = pop_string_keys_from_dict(original, ["deep.nested.structure.with.many.levels"])
        
        expected = {
            "deep": {
                "nested": {
                    "structure": {
                        "with": {
                            "many": {
                                "keep": "this_value"
                            }
                        }
                    }
                }
            },
            "top": "level"
        }
        assert result == expected

    def test_pop_nested_keys_nonexistent_path(self):
        """Test removing nested keys with nonexistent paths."""
        original = {
            "config": {
                "model": {"name": "bert"}
            }
        }
        result = pop_string_keys_from_dict(original, ["config.nonexistent.key", "missing.path"])
        
        expected = {
            "config": {
                "model": {"name": "bert"}
            }
        }
        assert result == expected

    def test_pop_mixed_keys(self):
        """Test removing both top-level and nested keys together."""
        original = {
            "top_key": "value1",
            "config": {
                "model": {"name": "bert", "version": "1.0"},
                "data": {"path": "/data"}
            },
            "another_top": "value2"
        }
        result = pop_string_keys_from_dict(original, [
            "top_key", 
            "config.model.name", 
            "config.data.path"
        ])
        
        expected = {
            "config": {
                "model": {"version": "1.0"},
                "data": {}
            },
            "another_top": "value2"
        }
        assert result == expected

    def test_pop_entire_nested_structure(self):
        """Test removing an entire nested structure."""
        original = {
            "keep": "this",
            "remove": {
                "entire": {
                    "structure": "here"
                }
            }
        }
        result = pop_string_keys_from_dict(original, ["remove"])
        
        expected = {"keep": "this"}
        assert result == expected

    def test_deep_copy_independence(self):
        """Test that the result is a deep copy independent of the original."""
        original = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"inner": "value"}
            }
        }
        result = pop_string_keys_from_dict(original, ["nested.dict.inner"])
        
        # Modify the original
        original["nested"]["list"].append(4)
        original["nested"]["dict"]["new"] = "added"
        
        # Result should be unaffected
        expected = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {}
            }
        }
        assert result == expected

    def test_pop_with_complex_nested_data_types(self):
        """Test with complex nested data types."""
        original = {
            "data": {
                "list": [1, 2, {"nested_in_list": True}],
                "tuple": (1, 2, 3),
                "set_data": {"set_item1", "set_item2"},
                "remove_me": "gone"
            },
            "keep": "this"
        }
        result = pop_string_keys_from_dict(original, ["data.remove_me"])
        
        expected = {
            "data": {
                "list": [1, 2, {"nested_in_list": True}],
                "tuple": (1, 2, 3),
                "set_data": {"set_item1", "set_item2"}
            },
            "keep": "this"
        }
        assert result == expected


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestRemoveNestedKeyRecursive:
    """Test class for _remove_nested_key_recursive function."""

    def test_remove_simple_nested_key(self):
        """Test removing a simple nested key."""
        dictionary = {"config": {"model": {"name": "bert"}}}
        _remove_nested_key_recursive(dictionary, ["config", "model", "name"])
        
        expected = {"config": {"model": {}}}
        assert dictionary == expected

    def test_remove_key_from_single_level(self):
        """Test removing a key from a single level dictionary."""
        dictionary = {"key1": "value1", "key2": "value2"}
        _remove_nested_key_recursive(dictionary, ["key1"])
        
        expected = {"key2": "value2"}
        assert dictionary == expected

    def test_remove_key_nonexistent_path(self):
        """Test removing a key with nonexistent path."""
        dictionary = {"config": {"model": {"name": "bert"}}}
        original_dict = dictionary.copy()
        
        _remove_nested_key_recursive(dictionary, ["config", "nonexistent", "key"])
        
        # Dictionary should be unchanged
        assert dictionary == original_dict

    def test_remove_key_empty_path(self):
        """Test with empty key path."""
        dictionary = {"key": "value"}
        original_dict = dictionary.copy()
        
        _remove_nested_key_recursive(dictionary, [])
        
        # Dictionary should be unchanged
        assert dictionary == original_dict

    def test_remove_key_non_dict_target(self):
        """Test when target is not a dictionary."""
        dictionary = {"config": {"model": "string_value"}}
        original_dict = dictionary.copy()
        
        _remove_nested_key_recursive(dictionary, ["config", "model", "nonexistent"])
        
        # Dictionary should be unchanged since "string_value" is not a dict
        assert dictionary == original_dict

    def test_remove_key_partial_path_exists(self):
        """Test when only partial path exists."""
        dictionary = {"config": {"model": {"name": "bert"}}}
        original_dict = dictionary.copy()
        
        _remove_nested_key_recursive(dictionary, ["config", "training", "epochs"])
        
        # Dictionary should be unchanged since "training" doesn't exist
        assert dictionary == original_dict

    def test_remove_deeply_nested_key(self):
        """Test removing a deeply nested key."""
        dictionary = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "target": "remove_this",
                            "keep": "this"
                        }
                    }
                }
            }
        }
        
        _remove_nested_key_recursive(dictionary, ["level1", "level2", "level3", "level4", "target"])
        
        expected = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "keep": "this"
                        }
                    }
                }
            }
        }
        assert dictionary == expected

    def test_remove_key_with_non_dict_input(self):
        """Test function behavior with non-dictionary input."""
        # Test the isinstance check for non-dict inputs
        # We'll test this indirectly by having a dict with non-dict values
        dictionary = {"config": "not_a_dict"}
        original_dict = dictionary.copy()
        
        # This should not raise an error and should return without changes
        # since "config" is not a dict when we try to go deeper
        _remove_nested_key_recursive(dictionary, ["config", "nested", "key"])
        
        # Dictionary should remain unchanged
        assert dictionary == original_dict

    def test_remove_key_intermediate_non_dict(self):
        """Test when intermediate path contains non-dict values."""
        dictionary = {
            "config": {
                "model": "string_not_dict",
                "other": {"nested": "value"}
            }
        }
        original_dict = dictionary.copy()
        
        _remove_nested_key_recursive(dictionary, ["config", "model", "nested", "key"])
        
        # Dictionary should be unchanged since model is not a dict
        assert dictionary == original_dict
