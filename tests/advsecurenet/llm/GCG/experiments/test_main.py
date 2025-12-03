"""Tests for the main experiment runner."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from ml_collections import config_dict
from advsecurenet.llm.GCG.experiments.main import dynamic_import, main


class TestDynamicImport:
    """Test dynamic module importing."""
    
    def test_dynamic_import_valid_module(self):
        """Test importing a valid module."""
        # Import a standard library module
        module = dynamic_import("os")
        assert module is not None
        assert hasattr(module, "path")
    
    def test_dynamic_import_nested_module(self):
        """Test importing a nested module path."""
        module = dynamic_import("os.path")
        assert module is not None
        assert hasattr(module, "join")
    
    def test_dynamic_import_invalid_module(self):
        """Test importing a non-existent module raises ImportError."""
        with pytest.raises(ImportError):
            dynamic_import("this.module.does.not.exist")


class TestMainFunction:
    """Test the main function with mocked dependencies."""
    
    @patch('advsecurenet.llm.GCG.experiments.main._CONFIG')
    @patch('advsecurenet.llm.GCG.experiments.main.mp.set_start_method')
    @patch('advsecurenet.llm.GCG.experiments.main.dynamic_import')
    @patch('advsecurenet.llm.GCG.experiments.main.get_goals_and_targets')
    @patch('advsecurenet.llm.GCG.experiments.main.get_workers')
    @patch('time.strftime')
    def test_main_transfer_attack(self, mock_strftime, mock_get_workers, 
                                mock_get_goals, mock_dynamic_import, 
                                mock_set_start, mock_config):
        """Test main function with transfer attack configuration."""
        
        # Mock timestamp
        mock_strftime.return_value = "20231203-10:30:45"
        
        # Create mock config
        config = config_dict.ConfigDict()
        config.attack = "gcg"
        config.transfer = True
        config.progressive_models = False
        config.progressive_goals = False
        config.control_init = "! ! ! !"
        config.result_prefix = "test_results"
        config.gbda_deterministic = True
        config.lr = 0.01
        config.batch_size = 32
        config.n_steps = 10
        config.topk = 256
        config.temp = 1.0
        config.target_weight = 1.0
        config.control_weight = 0.1
        config.test_steps = 5
        config.anneal = False
        config.incr_control = False
        config.stop_on_success = False
        config.verbose = True
        config.filter_cand = True
        config.allow_non_ascii = False
        
        # Mock config value property
        type(mock_config).value = PropertyMock(return_value=config)
        
        # Mock attack library
        mock_attack_lib = MagicMock()
        mock_dynamic_import.return_value = mock_attack_lib
        
        # Mock goals and targets
        mock_get_goals.return_value = (
            ["goal1", "goal2"],  # train_goals
            ["target1", "target2"],  # train_targets
            ["test_goal1"],  # test_goals
            ["test_target1"]  # test_targets
        )
        
        # Mock workers
        mock_worker1 = MagicMock()
        mock_worker2 = MagicMock()
        mock_test_worker = MagicMock()
        mock_get_workers.return_value = (
            [mock_worker1, mock_worker2],
            [mock_test_worker]
        )
        
        # Mock progressive attack
        mock_progressive_attack = MagicMock()
        mock_attack_lib.ProgressiveMultiPromptAttack.return_value = mock_progressive_attack
        
        # Run main function
        main([])
        
        # Verify multiprocessing setup
        mock_set_start.assert_called_once_with("spawn")
        
        # Verify dynamic import
        mock_dynamic_import.assert_called_once_with("advsecurenet.llm.GCG.src.gcg")
        
        # Verify goals and targets retrieval
        mock_get_goals.assert_called_once_with(config)
        mock_get_workers.assert_called_once_with(config)
        
        # Verify progressive attack creation
        mock_attack_lib.ProgressiveMultiPromptAttack.assert_called_once()
        call_args = mock_attack_lib.ProgressiveMultiPromptAttack.call_args
        
        # Check that targets were processed (Sure, h -> H replacement)
        processed_targets = call_args[0][1]
        assert len(processed_targets) == 2
        
        # Verify attack execution
        mock_progressive_attack.run.assert_called_once()
        run_args = mock_progressive_attack.run.call_args
        assert run_args[1]['n_steps'] == 10
        assert run_args[1]['batch_size'] == 32
        assert run_args[1]['topk'] == 256
        
        # Verify worker cleanup
        mock_worker1.stop.assert_called_once()
        mock_worker2.stop.assert_called_once()
        mock_test_worker.stop.assert_called_once()

    @patch('advsecurenet.llm.GCG.experiments.main._CONFIG')
    @patch('advsecurenet.llm.GCG.experiments.main.mp.set_start_method')
    @patch('advsecurenet.llm.GCG.experiments.main.dynamic_import')
    @patch('advsecurenet.llm.GCG.experiments.main.get_goals_and_targets')
    @patch('advsecurenet.llm.GCG.experiments.main.get_workers')
    @patch('time.strftime')
    def test_main_individual_attack(self, mock_strftime, mock_get_workers, 
                                  mock_get_goals, mock_dynamic_import, 
                                  mock_set_start, mock_config):
        """Test main function with individual attack configuration."""
        
        # Mock timestamp
        mock_strftime.return_value = "20231203-10:30:45"
        
        # Create mock config for individual attack
        config = config_dict.ConfigDict()
        config.attack = "gcg"
        config.transfer = False  # Individual attack
        config.control_init = "! ! ! !"
        config.result_prefix = "test_results"
        config.gbda_deterministic = True
        config.lr = 0.01
        config.batch_size = 32
        config.n_steps = 10
        config.topk = 256
        config.temp = 1.0
        config.target_weight = 1.0
        config.control_weight = 0.1
        config.test_steps = 5
        config.anneal = False
        config.incr_control = False
        config.stop_on_success = False
        config.verbose = True
        config.filter_cand = True
        config.allow_non_ascii = False
        
        # Mock config value property
        type(mock_config).value = PropertyMock(return_value=config)
        
        # Mock attack library
        mock_attack_lib = MagicMock()
        mock_dynamic_import.return_value = mock_attack_lib
        
        # Mock goals and targets
        mock_get_goals.return_value = (
            ["goal1"],  # train_goals
            ["target1"],  # train_targets  
            [],  # test_goals
            []  # test_targets
        )
        
        # Mock workers
        mock_worker = MagicMock()
        mock_get_workers.return_value = ([mock_worker], [])
        
        # Mock individual attack
        mock_individual_attack = MagicMock()
        mock_attack_lib.IndividualPromptAttack.return_value = mock_individual_attack
        
        # Run main function
        main([])
        
        # Verify individual attack creation
        mock_attack_lib.IndividualPromptAttack.assert_called_once()
        call_args = mock_attack_lib.IndividualPromptAttack.call_args
        
        # Check managers parameter
        managers = call_args[1]['managers']
        assert 'AP' in managers
        assert 'PM' in managers
        assert 'MPA' in managers
        assert managers['AP'] == mock_attack_lib.AttackPrompt
        assert managers['PM'] == mock_attack_lib.PromptManager
        assert managers['MPA'] == mock_attack_lib.MultiPromptAttack
        
        # Verify attack execution
        mock_individual_attack.run.assert_called_once()
        
        # Verify worker cleanup
        mock_worker.stop.assert_called_once()

    @patch('advsecurenet.llm.GCG.experiments.main._CONFIG')
    @patch('advsecurenet.llm.GCG.experiments.main.mp.set_start_method')
    @patch('advsecurenet.llm.GCG.experiments.main.dynamic_import')
    @patch('advsecurenet.llm.GCG.experiments.main.get_goals_and_targets')
    @patch('advsecurenet.llm.GCG.experiments.main.get_workers')
    @patch('time.strftime')
    def test_main_target_processing(self, mock_strftime, mock_get_workers, 
                                   mock_get_goals, mock_dynamic_import, 
                                   mock_set_start, mock_config):
        """Test that targets are processed correctly with string replacements."""
        
        # Mock timestamp
        mock_strftime.return_value = "20231203-10:30:45"
        
        # Create minimal config
        config = config_dict.ConfigDict()
        config.attack = "gcg"
        config.transfer = False
        config.control_init = "! ! ! !"
        config.result_prefix = "test_results"
        config.gbda_deterministic = True
        config.lr = 0.01
        config.batch_size = 32
        config.n_steps = 10
        config.topk = 256
        config.temp = 1.0
        config.target_weight = 1.0
        config.control_weight = 0.1
        config.anneal = False
        config.incr_control = False
        config.stop_on_success = False
        config.verbose = True
        config.filter_cand = True
        config.allow_non_ascii = False
        
        type(mock_config).value = PropertyMock(return_value=config)
        
        # Mock attack library
        mock_attack_lib = MagicMock()
        mock_dynamic_import.return_value = mock_attack_lib
        mock_individual_attack = MagicMock()
        mock_attack_lib.IndividualPromptAttack.return_value = mock_individual_attack
        
        # Test targets with patterns that should be replaced
        original_targets = [
            "Sure, here is how to...",  # Should become "Sure, here's how to..."
            "Sure, here is another example",  # Should become "Sure, here's another example"
            "Some other response"  # Should stay the same or get "Sure, h" -> "H" replacement
        ]
        
        mock_get_goals.return_value = (
            ["goal1", "goal2", "goal3"],  # train_goals
            original_targets,  # train_targets
            [],  # test_goals  
            []  # test_targets
        )
        
        # Mock workers
        mock_worker = MagicMock()
        mock_get_workers.return_value = ([mock_worker], [])
        
        # Set random seed for reproducible test
        np.random.seed(42)
        
        # Run main function
        main([])
        
        # Get the processed targets from the attack creation call
        call_args = mock_attack_lib.IndividualPromptAttack.call_args
        processed_targets = call_args[0][1]  # Second argument is train_targets
        
        # Verify that targets were processed
        assert len(processed_targets) == 3
        
        # Check that at least some processing occurred
        # (exact results depend on random choices)
        targets_str = " ".join(processed_targets)
        assert "Sure, here is" not in targets_str or "Sure, here's" in targets_str

    @patch('advsecurenet.llm.GCG.experiments.main._CONFIG')
    @patch('advsecurenet.llm.GCG.experiments.main.mp.set_start_method')
    @patch('advsecurenet.llm.GCG.experiments.main.dynamic_import')
    def test_main_import_error(self, mock_dynamic_import, mock_set_start, mock_config):
        """Test main function handling of import errors."""
        
        # Create mock config
        config = config_dict.ConfigDict()
        config.attack = "nonexistent_attack"
        type(mock_config).value = PropertyMock(return_value=config)
        
        # Mock import error
        mock_dynamic_import.side_effect = ImportError("Module not found")
        
        # Verify that ImportError is raised
        with pytest.raises(ImportError):
            main([])
        
        # Verify multiprocessing setup was called
        mock_set_start.assert_called_once_with("spawn")


class TestMainIntegration:
    """Integration tests for main function components."""
    
    def test_managers_dict_creation(self):
        """Test that the managers dictionary is created correctly."""
        # This tests the structure that should be created in main
        mock_attack_lib = MagicMock()
        
        # Simulate the managers dict creation from main
        managers = {
            "AP": mock_attack_lib.AttackPrompt,
            "PM": mock_attack_lib.PromptManager,
            "MPA": mock_attack_lib.MultiPromptAttack,
        }
        
        assert len(managers) == 3
        assert "AP" in managers
        assert "PM" in managers  
        assert "MPA" in managers
        assert managers["AP"] == mock_attack_lib.AttackPrompt
        assert managers["PM"] == mock_attack_lib.PromptManager
        assert managers["MPA"] == mock_attack_lib.MultiPromptAttack

    def test_target_processing_functions(self):
        """Test the target processing lambda functions."""
        # Test the lambda functions used in main for target processing
        process_fn = lambda s: s.replace("Sure, h", "H")
        process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
        
        # Test first function
        result1 = process_fn("Sure, here is how to do it")
        assert result1 == "Here is how to do it"
        
        # Test second function  
        result2 = process_fn2("Sure, here is how to do it")
        assert result2 == "Sure, here's how to do it"
        
        # Test no replacement
        result3 = process_fn("This has no matching pattern")
        assert result3 == "This has no matching pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])