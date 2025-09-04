#!/usr/bin/env python3
"""
Standalone test script for dataset_config.py to avoid import issues.
"""

import sys
import os
import importlib.util

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../../../'))

# Import dataset_config module directly
dataset_config_path = os.path.join(root_dir, 'advsecurenet', 'shared', 'types', 'configs', 'dataset_config.py')
spec = importlib.util.spec_from_file_location("dataset_config", dataset_config_path)
dataset_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_config)

# Import preprocess_config module directly
preprocess_config_path = os.path.join(root_dir, 'advsecurenet', 'shared', 'types', 'configs', 'preprocess_config.py')
spec2 = importlib.util.spec_from_file_location("preprocess_config", preprocess_config_path)
preprocess_config = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(preprocess_config)

# Now we can use the classes
BaseDatasetCliConfig = dataset_config.BaseDatasetCliConfig
UserSplitConfig = dataset_config.UserSplitConfig
CreateDatasetCliConfig = dataset_config.CreateDatasetCliConfig
AttacksDatasetCliConfig = dataset_config.AttacksDatasetCliConfig
ResolvedSplitConfig = dataset_config.ResolvedSplitConfig
ResolvedDatasetConfig = dataset_config.ResolvedDatasetConfig
_get_identifier = dataset_config._get_identifier
_get_user_splits = dataset_config._get_user_splits
_create_resolved_split = dataset_config._create_resolved_split
resolve_dataset_config = dataset_config.resolve_dataset_config
PreprocessConfig = preprocess_config.PreprocessConfig

def test_all_functions():
    """Test all functions and classes to achieve 100% coverage."""
    
    print("Testing BaseDatasetCliConfig...")
    base_config = BaseDatasetCliConfig(dataset_name="test")
    assert base_config.dataset_name == "test"
    assert base_config.num_classes == 10
    assert base_config.preprocessing is None
    
    base_config2 = BaseDatasetCliConfig(
        dataset_name="test2",
        num_classes=5,
        preprocessing=PreprocessConfig()
    )
    assert base_config2.num_classes == 5
    
    print("Testing UserSplitConfig...")
    user_split = UserSplitConfig()
    assert user_split.identifier is None
    assert user_split.constructor_args == {}
    
    user_split2 = UserSplitConfig(
        identifier="custom",
        split_name="validation",
        dataset_kwargs={"key": "value"},
        constructor_args={"arg": "val"},
        path="/path"
    )
    assert user_split2.identifier == "custom"
    assert user_split2.path == "/path"
    
    print("Testing CreateDatasetCliConfig...")
    create_config = CreateDatasetCliConfig(dataset_name="create_test")
    assert create_config.dataset_name == "create_test"
    assert create_config.dataset_kwargs == {}
    assert create_config.load_splits == []
    
    create_config2 = CreateDatasetCliConfig(
        dataset_name="create_test2",
        identifier="custom_id",
        dataset_kwargs={"key": "value"},
        constructor_args={"arg": "val"},
        split_config={"train": UserSplitConfig()},
        load_splits=["train", "test"],
        random_sample_size=500
    )
    assert create_config2.identifier == "custom_id"
    assert create_config2.random_sample_size == 500
    
    print("Testing AttacksDatasetCliConfig...")
    attacks_config = AttacksDatasetCliConfig(dataset_name="attack_test")
    assert attacks_config.dataset_name == "attack_test"
    assert attacks_config.random_sample_size is None
    
    attacks_config2 = AttacksDatasetCliConfig(
        dataset_name="attack_test2",
        random_sample_size=100
    )
    assert attacks_config2.random_sample_size == 100
    
    print("Testing ResolvedSplitConfig...")
    resolved_split = ResolvedSplitConfig(identifier="resolved_id")
    assert resolved_split.identifier == "resolved_id"
    assert resolved_split.num_classes == 10
    assert resolved_split.kwargs == {}
    
    resolved_split2 = ResolvedSplitConfig(
        identifier="resolved_id2",
        num_classes=5,
        source_split_name="validation",
        preprocessing=PreprocessConfig(),
        kwargs={"key": "value"},
        constructor_args={"arg": "val"},
        path="/resolved/path"
    )
    assert resolved_split2.num_classes == 5
    assert resolved_split2.path == "/resolved/path"
    
    print("Testing ResolvedDatasetConfig...")
    splits = {"train": ResolvedSplitConfig(identifier="test_id")}
    resolved_config = ResolvedDatasetConfig(
        dataset_name="resolved_test",
        splits=splits
    )
    assert resolved_config.dataset_name == "resolved_test"
    assert resolved_config.random_sample_size is None
    
    resolved_config2 = ResolvedDatasetConfig(
        dataset_name="resolved_test2",
        splits=splits,
        random_sample_size=1000
    )
    assert resolved_config2.random_sample_size == 1000
    
    print("Testing _get_identifier...")
    config1 = CreateDatasetCliConfig(dataset_name="test", identifier="custom")
    result1 = _get_identifier(config1)
    assert result1 == "custom"
    
    config2 = CreateDatasetCliConfig(dataset_name="test")
    result2 = _get_identifier(config2)
    assert result2 == "test"
    
    print("Testing _get_user_splits...")
    # Test with split_config (highest priority)
    config3 = CreateDatasetCliConfig(
        dataset_name="test",
        split_config={"custom1": UserSplitConfig(), "custom2": UserSplitConfig()},
        load_splits=["ignored"]
    )
    result3 = _get_user_splits(config3)
    assert set(result3) == {"custom1", "custom2"}
    
    # Test with load_splits (second priority)
    config4 = CreateDatasetCliConfig(
        dataset_name="test",
        load_splits=["train", "test", "val"]
    )
    result4 = _get_user_splits(config4)
    assert result4 == ["train", "test", "val"]
    
    # Test default behavior
    config5 = CreateDatasetCliConfig(dataset_name="test")
    result5 = _get_user_splits(config5)
    assert result5 == ["train", "test"]
    
    # Test empty split_config falls back to load_splits
    config6 = CreateDatasetCliConfig(
        dataset_name="test",
        split_config={},
        load_splits=["custom"]
    )
    result6 = _get_user_splits(config6)
    assert result6 == ["custom"]
    
    # Test empty load_splits falls back to default
    config7 = CreateDatasetCliConfig(
        dataset_name="test",
        load_splits=[]
    )
    result7 = _get_user_splits(config7)
    assert result7 == ["train", "test"]
    
    print("Testing _create_resolved_split...")
    # Test without user config
    global_config = CreateDatasetCliConfig(
        dataset_name="global",
        num_classes=15,
        preprocessing=PreprocessConfig(),
        dataset_kwargs={"global": "param"},
        constructor_args={"global": "arg"}
    )
    
    resolved1 = _create_resolved_split(global_config, "train")
    assert resolved1.identifier == "global"
    assert resolved1.source_split_name == "train"
    assert resolved1.num_classes == 15
    assert resolved1.kwargs == {"global": "param"}
    assert resolved1.constructor_args == {"global": "arg"}
    assert resolved1.path is None
    
    # Test with user config overrides
    user_config = UserSplitConfig(
        identifier="user_id",
        split_name="validation",
        preprocessing=PreprocessConfig(),
        dataset_kwargs={"user": "param"},
        constructor_args={"user": "arg"},
        path="/user/path"
    )
    
    resolved2 = _create_resolved_split(global_config, "test", user_config)
    assert resolved2.identifier == "user_id"
    assert resolved2.source_split_name == "validation"
    assert resolved2.kwargs == {"user": "param"}
    assert resolved2.constructor_args == {"user": "arg"}
    assert resolved2.path == "/user/path"
    
    # Test with partial user config (fallback to global)
    user_partial = UserSplitConfig(identifier="partial_id")
    resolved3 = _create_resolved_split(global_config, "train", user_partial)
    assert resolved3.identifier == "partial_id"
    assert resolved3.source_split_name == "train"  # fallback
    assert resolved3.preprocessing == global_config.preprocessing  # fallback
    
    # Test with None values in global config
    global_none = CreateDatasetCliConfig(
        dataset_name="none_test",
        dataset_kwargs=None,
        constructor_args=None
    )
    resolved4 = _create_resolved_split(global_none, "train")
    assert resolved4.kwargs == {}
    assert resolved4.constructor_args == {}
    
    print("Testing resolve_dataset_config...")
    # Simple case
    simple_config = CreateDatasetCliConfig(
        dataset_name="simple",
        load_splits=["train", "test"]
    )
    resolved_simple = resolve_dataset_config(simple_config)
    assert resolved_simple.dataset_name == "simple"
    assert len(resolved_simple.splits) == 2
    assert "train" in resolved_simple.splits
    assert "test" in resolved_simple.splits
    
    # With split_config
    split_user_config = UserSplitConfig(identifier="custom_split_id")
    complex_config = CreateDatasetCliConfig(
        dataset_name="complex",
        split_config={"custom_split": split_user_config},
        random_sample_size=500
    )
    resolved_complex = resolve_dataset_config(complex_config)
    assert len(resolved_complex.splits) == 1
    assert "custom_split" in resolved_complex.splits
    assert resolved_complex.splits["custom_split"].identifier == "custom_split_id"
    assert resolved_complex.random_sample_size == 500
    
    # Test error case - no splits
    try:
        empty_config = CreateDatasetCliConfig(
            dataset_name="empty",
            load_splits=[]
        )
        resolve_dataset_config(empty_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No splits defined" in str(e)
    
    print("All tests passed! 🎉")

if __name__ == "__main__":
    test_all_functions()
