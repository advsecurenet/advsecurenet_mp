import pytest
import torch
import numpy as np
import json
import tempfile
from unittest.mock import MagicMock, patch
from advsecurenet.llm.GCG.src.attacks.multi_prompt import MultiPromptAttack


@pytest.fixture
def mock_worker():
    worker = MagicMock()
    worker.model = MagicMock()
    worker.tokenizer = MagicMock()
    worker.tokenizer.decode = MagicMock(return_value="test decoded")
    worker.tokenizer.vocab_size = 1000
    worker.conv_template = MagicMock()
    return worker


@pytest.fixture
def mock_prompt_manager():
    pm = MagicMock()
    pm.control_str = "test control"
    pm.control_toks = torch.tensor([1, 2, 3])
    return pm


@pytest.fixture
def mock_managers(mock_prompt_manager):
    return {
        'PM': MagicMock(return_value=mock_prompt_manager),
        'AP': MagicMock()
    }


@pytest.fixture
def basic_config(mock_worker, mock_managers):
    return {
        'goals': ['test goal'],
        'targets': ['test target'],
        'workers': [mock_worker],
        'managers': mock_managers
    }


class TestMultiPromptAttackInit:
    def test_basic_initialization(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        assert attack.goals == ['test goal']
        assert attack.targets == ['test target']
        assert len(attack.workers) == 1
        assert len(attack.models) == 1
        assert len(attack.prompts) == 1

    def test_custom_control_init(self, basic_config):
        basic_config['control_init'] = "custom control"
        attack = MultiPromptAttack(**basic_config)
        
        # Verify PM was called with custom control
        basic_config['managers']['PM'].assert_called_with(
            ['test goal'],
            ['test target'],
            basic_config['workers'][0].tokenizer,
            basic_config['workers'][0].conv_template,
            "custom control",
            ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
            basic_config['managers']
        )

    def test_custom_test_prefixes(self, basic_config):
        custom_prefixes = ["Sorry", "Cannot"]
        basic_config['test_prefixes'] = custom_prefixes
        attack = MultiPromptAttack(**basic_config)
        
        assert attack.test_prefixes == custom_prefixes

    def test_with_test_data(self, basic_config, mock_worker):
        basic_config.update({
            'test_goals': ['test goal 2'],
            'test_targets': ['test target 2'],
            'test_workers': [mock_worker]
        })
        attack = MultiPromptAttack(**basic_config)
        
        assert attack.test_goals == ['test goal 2']
        assert attack.test_targets == ['test target 2']
        assert len(attack.test_workers) == 1

    def test_multiple_workers(self, mock_worker, mock_managers, mock_prompt_manager):
        workers = [mock_worker, mock_worker, mock_worker]
        attack = MultiPromptAttack(
            goals=['goal'],
            targets=['target'],
            workers=workers,
            managers=mock_managers
        )
        
        assert len(attack.workers) == 3
        assert len(attack.models) == 3
        assert len(attack.prompts) == 3
        assert mock_managers['PM'].call_count == 3

    def test_with_logfile(self, basic_config):
        basic_config['logfile'] = "test.log"
        attack = MultiPromptAttack(**basic_config)
        
        assert attack.logfile == "test.log"


class TestMultiPromptAttackProperties:
    def test_control_str_getter(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        # Mock the first prompt's control_str
        attack.prompts[0].control_str = "test control string"
        
        assert attack.control_str == "test control string"

    def test_control_str_setter(self, basic_config, mock_worker):
        workers = [mock_worker, mock_worker]
        basic_config['workers'] = workers
        
        attack = MultiPromptAttack(**basic_config)
        attack.control_str = "new control"
        
        for prompt in attack.prompts:
            prompt.control_str = "new control"

    def test_control_toks_getter(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        # Mock control_toks for prompts
        attack.prompts[0].control_toks = torch.tensor([1, 2, 3])
        
        result = attack.control_toks
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([1, 2, 3]))

    def test_control_toks_setter_valid(self, basic_config, mock_worker):
        workers = [mock_worker, mock_worker]
        basic_config['workers'] = workers
        
        attack = MultiPromptAttack(**basic_config)
        new_control = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        attack.control_toks = new_control
        
        for i, prompt in enumerate(attack.prompts):
            prompt.control_toks = new_control[i]

    def test_control_toks_setter_invalid_length(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        with pytest.raises(ValueError, match="Must provide control tokens for each tokenizer"):
            attack.control_toks = [torch.tensor([1, 2]), torch.tensor([3, 4])]  # Too many


class TestGetFilteredCands:
    def test_get_filtered_cands_basic(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        control_cand = torch.tensor([[1, 2, 3], [4, 5, 6]])
        
        # Mock tokenizer decode
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["cand1", "cand2"]
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=False)
        
        assert len(cands) == 2
        assert count == 2
        assert cands == ["cand1", "cand2"]

    def test_get_filtered_cands_with_none_tokens(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        # Create control with some None values (simulating invalid tokens)
        control_cand = torch.tensor([[1, 2, 3], [4, 999999, 6]])  # Second has invalid token
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["valid_cand", ""]  # Second decode fails
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=True)
        
        assert len(cands) == 1  # Only valid candidate
        assert count == 1

    def test_get_filtered_cands_filter_duplicates(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        control_cand = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])  # First two identical
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["same", "same", "different"]
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=True)
        
        # Should filter out duplicate "same"
        assert len(cands) == 2
        assert "same" in cands
        assert "different" in cands

    def test_get_filtered_cands_with_curr_control(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        control_cand = torch.tensor([[1, 2, 3], [4, 5, 6]])
        curr_control = "current"
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["current", "new_cand"]  # First matches current
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=True, curr_control=curr_control)
        
        # Should filter out current control
        assert len(cands) == 1
        assert cands[0] == "new_cand"

    def test_get_filtered_cands_token_filtering(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        # Create tensor with values that would be filtered
        control_cand = torch.tensor([[1, 2, 50000], [4, 5, 6]])  # First has out-of-vocab token
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["", "valid"]  # First decode returns empty
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=True)
        
        assert len(cands) == 1
        assert cands[0] == "valid"

    def test_get_filtered_cands_exception_handling(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        control_cand = torch.tensor([[1, 2, 3]])
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = Exception("Decode error")
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=True)
        
        # Should handle exception gracefully
        assert len(cands) == 0
        assert count == 0

    def test_get_filtered_cands_no_filter(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        control_cand = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]])  # Has duplicates
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["same", "different", "same"]
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=False)
        
        # Should not filter duplicates when filter_cand=False
        assert len(cands) == 3
        assert count == 3

    def test_get_filtered_cands_tensor_conversion(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        # Test with numpy array instead of tensor
        control_cand = np.array([[1, 2, 3], [4, 5, 6]])
        
        worker = attack.workers[0]
        worker.tokenizer.decode.side_effect = ["cand1", "cand2"]
        worker.tokenizer.vocab_size = 1000
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=False)
        
        assert len(cands) == 2
        assert count == 2

    def test_get_filtered_cands_empty_input(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        control_cand = torch.tensor([]).reshape(0, 3)  # Empty tensor
        
        cands, count = attack.get_filtered_cands(0, control_cand, filter_cand=True)
        
        assert len(cands) == 0
        assert count == 0


class TestMultiPromptAttackEdgeCases:
    def test_empty_goals_targets(self, mock_worker, mock_managers):
        attack = MultiPromptAttack(
            goals=[],
            targets=[],
            workers=[mock_worker],
            managers=mock_managers
        )
        
        assert attack.goals == []
        assert attack.targets == []

    def test_mismatched_goals_targets_length(self, mock_worker, mock_managers):
        # This should still work as the attack handles it in prompt manager
        attack = MultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1'],  # Fewer targets
            workers=[mock_worker],
            managers=mock_managers
        )
        
        assert len(attack.goals) == 2
        assert len(attack.targets) == 1

    def test_no_workers(self, mock_managers):
        attack = MultiPromptAttack(
            goals=['goal'],
            targets=['target'],
            workers=[],
            managers=mock_managers
        )
        
        assert len(attack.workers) == 0
        assert len(attack.models) == 0
        assert len(attack.prompts) == 0

    def test_none_logfile(self, basic_config):
        basic_config['logfile'] = None
        attack = MultiPromptAttack(**basic_config)
        
        assert attack.logfile is None

    def test_managers_none(self, mock_worker):
        with pytest.raises((KeyError, TypeError)):
            MultiPromptAttack(
                goals=['goal'],
                targets=['target'],
                workers=[mock_worker],
                managers=None
            )


class TestMultiPromptAttackIntegration:
    def test_full_initialization_workflow(self, mock_worker, mock_managers, mock_prompt_manager):
        # Test complete initialization with all parameters
        attack = MultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1', 'target2'],
            workers=[mock_worker],
            control_init="custom init",
            test_prefixes=["Custom", "Prefixes"],
            logfile="test.log",
            managers=mock_managers,
            test_goals=['test_goal'],
            test_targets=['test_target'],
            test_workers=[mock_worker]
        )
        
        assert len(attack.goals) == 2
        assert len(attack.targets) == 2
        assert attack.test_prefixes == ["Custom", "Prefixes"]
        assert attack.logfile == "test.log"
        assert len(attack.test_goals) == 1
        assert len(attack.test_targets) == 1
        assert len(attack.test_workers) == 1

    def test_prompt_manager_integration(self, basic_config):
        attack = MultiPromptAttack(**basic_config)
        
        # Verify PM was called correctly
        pm_call = basic_config['managers']['PM'].call_args
        assert pm_call[0][0] == ['test goal']  # goals
        assert pm_call[0][1] == ['test target']  # targets
        assert pm_call[0][2] == basic_config['workers'][0].tokenizer
        assert pm_call[0][3] == basic_config['workers'][0].conv_template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])