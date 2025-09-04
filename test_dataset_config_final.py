#!/usr/bin/env python3
"""
Final comprehensive test for dataset_config.py to achieve 100% coverage
"""

import sys
import os
import coverage

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Mock the PreprocessConfig import before importing dataset_config
class MockPreprocessConfig:
    """Mock PreprocessConfig for testing"""

    def __init__(self):
        pass


# Create mock modules to prevent import errors
class MockModule:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


sys.modules["advsecurenet"] = MockModule()
sys.modules["advsecurenet.shared"] = MockModule()
sys.modules["advsecurenet.shared.types"] = MockModule()
sys.modules["advsecurenet.shared.types.configs"] = MockModule()
sys.modules["advsecurenet.shared.types.configs.preprocess_config"] = MockModule(
    PreprocessConfig=MockPreprocessConfig
)

# Now we can import the module directly
import importlib.util
import types
import builtins

spec = importlib.util.spec_from_file_location(
    "dataset_config",
    "/Users/fabienmorgan/Desktop/Ausbildung/Master/Masters_Project/Code/advsecurenet_mp/advsecurenet/shared/types/configs/dataset_config.py",
)
dc = importlib.util.module_from_spec(spec)

# Replace the PreprocessConfig import with our mock
original_import = builtins.__import__


def mock_import(name, *args, **kwargs):
    if name == "advsecurenet.shared.types.configs.preprocess_config":
        mock_module = types.ModuleType(name)
        setattr(mock_module, "PreprocessConfig", MockPreprocessConfig)
        return mock_module
    return original_import(name, *args, **kwargs)


builtins.__import__ = mock_import

try:
    spec.loader.exec_module(dc)
finally:
    builtins.__import__ = original_import


def test_comprehensive_coverage():
    """Test all functionality in dataset_config.py"""

    print("Testing BaseDatasetCliConfig...")
    # Test default initialization
    base1 = dc.BaseDatasetCliConfig("test_dataset")
    assert base1.dataset_name == "test_dataset"
    assert base1.num_classes is None
    assert base1.preprocessing is None

    # Test full initialization
    base2 = dc.BaseDatasetCliConfig("test2", 10, MockPreprocessConfig())
    assert base2.dataset_name == "test2"
    assert base2.num_classes == 10
    assert base2.preprocessing is not None

    print("Testing UserSplitConfig...")
    # Test default initialization
    user1 = dc.UserSplitConfig()
    assert user1.identifier is None
    assert user1.source_split_name is None
    assert user1.preprocessing is None
    assert user1.kwargs == {}
    assert user1.constructor_args == {}
    assert user1.path is None

    # Test full initialization
    user2 = dc.UserSplitConfig(
        "user_id",
        "validation",
        MockPreprocessConfig(),
        {"key": "value"},
        {"arg": "value"},
        "/path/to/data",
    )
    assert user2.identifier == "user_id"
    assert user2.source_split_name == "validation"
    assert user2.preprocessing is not None
    assert user2.kwargs == {"key": "value"}
    assert user2.constructor_args == {"arg": "value"}
    assert user2.path == "/path/to/data"

    print("Testing CreateDatasetCliConfig...")
    # Test minimal initialization
    create1 = dc.CreateDatasetCliConfig("create_dataset")
    assert create1.dataset_name == "create_dataset"
    assert create1.num_classes is None
    assert create1.preprocessing is None
    assert create1.identifier is None
    assert create1.dataset_kwargs == {}
    assert create1.constructor_args == {}
    assert create1.split_config is None
    assert create1.load_splits is None
    assert create1.random_sample_size is None

    # Test full initialization
    split_config = {"train": dc.UserSplitConfig()}
    create2 = dc.CreateDatasetCliConfig(
        "create2",
        5,
        MockPreprocessConfig(),
        "custom_id",
        {"dataset": "kwargs"},
        {"constructor": "args"},
        split_config,
        ["train", "test"],
        1000,
    )
    assert create2.dataset_name == "create2"
    assert create2.num_classes == 5
    assert create2.preprocessing is not None
    assert create2.identifier == "custom_id"
    assert create2.dataset_kwargs == {"dataset": "kwargs"}
    assert create2.constructor_args == {"constructor": "args"}
    assert create2.split_config == split_config
    assert create2.load_splits == ["train", "test"]
    assert create2.random_sample_size == 1000

    print("Testing AttacksDatasetCliConfig...")
    # Test minimal initialization
    attack1 = dc.AttacksDatasetCliConfig("attack_dataset")
    assert attack1.dataset_name == "attack_dataset"
    assert attack1.random_sample_size is None

    # Test with sample size
    attack2 = dc.AttacksDatasetCliConfig("attack2", random_sample_size=500)
    assert attack2.dataset_name == "attack2"
    assert attack2.random_sample_size == 500

    print("Testing ResolvedSplitConfig...")
    # Test minimal initialization
    resolved1 = dc.ResolvedSplitConfig("resolved_id")
    assert resolved1.identifier == "resolved_id"
    assert resolved1.num_classes is None
    assert resolved1.source_split_name is None
    assert resolved1.preprocessing is None
    assert resolved1.kwargs == {}
    assert resolved1.constructor_args == {}
    assert resolved1.path is None

    # Test full initialization
    resolved2 = dc.ResolvedSplitConfig(
        "resolved2",
        8,
        "train",
        MockPreprocessConfig(),
        {"key": "value"},
        {"arg": "value"},
        "/resolved/path",
    )
    assert resolved2.identifier == "resolved2"
    assert resolved2.num_classes == 8
    assert resolved2.source_split_name == "train"
    assert resolved2.preprocessing is not None
    assert resolved2.kwargs == {"key": "value"}
    assert resolved2.constructor_args == {"arg": "value"}
    assert resolved2.path == "/resolved/path"

    print("Testing ResolvedDatasetConfig...")
    # Test minimal initialization
    splits = {"train": dc.ResolvedSplitConfig("train_id")}
    resolved_dataset1 = dc.ResolvedDatasetConfig("resolved_dataset", splits)
    assert resolved_dataset1.dataset_name == "resolved_dataset"
    assert resolved_dataset1.splits == splits
    assert resolved_dataset1.random_sample_size is None

    # Test with sample size
    resolved_dataset2 = dc.ResolvedDatasetConfig("resolved2", splits, 750)
    assert resolved_dataset2.dataset_name == "resolved2"
    assert resolved_dataset2.splits == splits
    assert resolved_dataset2.random_sample_size == 750

    print("Testing _get_identifier function...")
    # Test with custom identifier
    config_custom = dc.CreateDatasetCliConfig("test", identifier="custom_identifier")
    identifier1 = dc._get_identifier(config_custom)
    assert identifier1 == "custom_identifier"

    # Test without custom identifier (should return dataset_name)
    config_default = dc.CreateDatasetCliConfig("default_dataset")
    identifier2 = dc._get_identifier(config_default)
    assert identifier2 == "default_dataset"

    print("Testing _get_user_splits function...")
    # Test with split_config (should return split_config and ignore load_splits)
    split_config = {
        "custom_train": dc.UserSplitConfig(),
        "custom_test": dc.UserSplitConfig(),
    }
    config1 = dc.CreateDatasetCliConfig(
        "test", split_config=split_config, load_splits=["ignored"]
    )
    user_splits1 = dc._get_user_splits(config1)
    assert user_splits1 == split_config

    # Test with load_splits only
    config2 = dc.CreateDatasetCliConfig(
        "test", load_splits=["train", "validation", "test"]
    )
    user_splits2 = dc._get_user_splits(config2)
    expected_splits = {
        name: dc.UserSplitConfig() for name in ["train", "validation", "test"]
    }
    assert list(user_splits2.keys()) == ["train", "validation", "test"]

    # Test with no split_config and no load_splits (should default to train, test)
    config3 = dc.CreateDatasetCliConfig("test")
    user_splits3 = dc._get_user_splits(config3)
    assert list(user_splits3.keys()) == ["train", "test"]

    # Test with empty split_config but has load_splits
    config4 = dc.CreateDatasetCliConfig(
        "test", split_config={}, load_splits=["custom_split"]
    )
    user_splits4 = dc._get_user_splits(config4)
    assert list(user_splits4.keys()) == ["custom_split"]

    # Test with empty load_splits (should default to train, test)
    config5 = dc.CreateDatasetCliConfig("test", load_splits=[])
    user_splits5 = dc._get_user_splits(config5)
    assert list(user_splits5.keys()) == ["train", "test"]

    print("Testing _create_resolved_split function...")
    # Test without user config
    global_config = dc.CreateDatasetCliConfig(
        "global_test",
        15,
        MockPreprocessConfig(),
        dataset_kwargs={"global": "kwargs"},
        constructor_args={"global": "args"},
    )
    resolved_split1 = dc._create_resolved_split(global_config, "train")
    assert resolved_split1.identifier == "global_test"
    assert resolved_split1.num_classes == 15
    assert resolved_split1.source_split_name == "train"
    assert resolved_split1.preprocessing is not None
    assert resolved_split1.kwargs == {"global": "kwargs"}
    assert resolved_split1.constructor_args == {"global": "args"}
    assert resolved_split1.path is None

    # Test with user config (user config should override global)
    user_config = dc.UserSplitConfig(
        "user_identifier",
        "validation",
        MockPreprocessConfig(),
        {"user": "kwargs"},
        {"user": "args"},
        "/user/path",
    )
    resolved_split2 = dc._create_resolved_split(global_config, "test", user_config)
    assert resolved_split2.identifier == "user_identifier"
    assert resolved_split2.num_classes == 15  # Global value used
    assert resolved_split2.source_split_name == "validation"
    assert resolved_split2.preprocessing is not None
    assert resolved_split2.kwargs == {"user": "kwargs"}
    assert resolved_split2.constructor_args == {"user": "args"}
    assert resolved_split2.path == "/user/path"

    # Test with partial user config (some fields None)
    user_partial = dc.UserSplitConfig(identifier="partial_id")
    resolved_split3 = dc._create_resolved_split(
        global_config, "validation", user_partial
    )
    assert resolved_split3.identifier == "partial_id"
    assert resolved_split3.num_classes == 15  # Global value
    assert (
        resolved_split3.source_split_name == "validation"
    )  # Global overridden by split_name
    assert resolved_split3.preprocessing is not None  # Global value
    assert resolved_split3.kwargs == {"global": "kwargs"}  # Global value
    assert resolved_split3.constructor_args == {"global": "args"}  # Global value
    assert resolved_split3.path is None  # User value (None)

    # Test with global config having None values
    global_none = dc.CreateDatasetCliConfig(
        "none_test", dataset_kwargs=None, constructor_args=None
    )
    resolved_split4 = dc._create_resolved_split(global_none, "train")
    assert resolved_split4.kwargs == {}
    assert resolved_split4.constructor_args == {}

    print("Testing resolve_dataset_config function...")
    # Test simple config
    simple_config = dc.CreateDatasetCliConfig(
        "simple_dataset", load_splits=["train", "test"]
    )
    resolved_simple = dc.resolve_dataset_config(simple_config)
    assert resolved_simple.dataset_name == "simple_dataset"
    assert list(resolved_simple.splits.keys()) == ["train", "test"]
    assert resolved_simple.random_sample_size is None

    # Test complex config with split_config
    split_config = {"custom_train": dc.UserSplitConfig(identifier="custom_id")}
    complex_config = dc.CreateDatasetCliConfig(
        "complex_dataset", split_config=split_config, random_sample_size=2000
    )
    resolved_complex = dc.resolve_dataset_config(complex_config)
    assert resolved_complex.dataset_name == "complex_dataset"
    assert list(resolved_complex.splits.keys()) == ["custom_train"]
    assert resolved_complex.splits["custom_train"].identifier == "custom_id"
    assert resolved_complex.random_sample_size == 2000

    # Test error case - empty splits should raise ValueError
    try:
        empty_config = dc.CreateDatasetCliConfig(
            "empty_dataset", split_config=None, load_splits=[]
        )
        dc.resolve_dataset_config(empty_config)
        assert False, "Should have raised ValueError for empty splits"
    except ValueError as e:
        assert "No splits" in str(e), f"Unexpected error message: {e}"
        print(f"Correctly caught ValueError: {e}")

    print("All tests passed successfully!")


if __name__ == "__main__":
    # Start coverage
    cov = coverage.Coverage(source=["advsecurenet/shared/types/configs/dataset_config"])
    cov.start()

    try:
        # Run all tests
        test_comprehensive_coverage()
        print("\n" + "=" * 50)
        print("SUCCESS: All functionality tested!")

    finally:
        # Stop coverage and report
        cov.stop()
        cov.save()
        print("\n" + "=" * 50)
        print("COVERAGE REPORT:")
        print("=" * 50)
        cov.report(show_missing=True)
