import pytest
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from advsecurenet.llm.GCG.src.attacks.progressive import ProgressiveMultiPromptAttack


@pytest.fixture
def mock_worker():
    worker = MagicMock()
    worker.model = MagicMock()
    worker.model.name_or_path = "test-model"
    worker.tokenizer = MagicMock()
    worker.tokenizer.name_or_path = "test-tokenizer"
    worker.conv_template = MagicMock()
    worker.conv_template.name = "test-template"
    return worker


@pytest.fixture
def mock_mpa():
    mpa = MagicMock()
    mpa.run = MagicMock()
    mpa.control = "optimized control"
    return mpa


@pytest.fixture
def mock_managers(mock_mpa):
    return {
        'PM': MagicMock(),
        'AP': MagicMock(),
        'MPA': MagicMock(return_value=mock_mpa)
    }


@pytest.fixture
def basic_config(mock_worker, mock_managers):
    return {
        'goals': ['goal1', 'goal2'],
        'targets': ['target1', 'target2'],
        'workers': [mock_worker],
        'managers': mock_managers
    }


class TestProgressiveMultiPromptAttackInit:
    def test_basic_initialization(self, basic_config):
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.goals == ['goal1', 'goal2']
        assert attack.targets == ['target1', 'target2']
        assert len(attack.workers) == 1
        assert attack.progressive_goals is True
        assert attack.progressive_models is True
        assert attack.control == "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    def test_custom_progressive_flags(self, basic_config):
        basic_config['progressive_goals'] = False
        basic_config['progressive_models'] = False
        
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.progressive_goals is False
        assert attack.progressive_models is False

    def test_custom_control_init(self, basic_config):
        basic_config['control_init'] = "custom control"
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.control == "custom control"

    def test_custom_test_prefixes(self, basic_config):
        custom_prefixes = ["Sorry", "Cannot"]
        basic_config['test_prefixes'] = custom_prefixes
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.test_prefixes == custom_prefixes

    def test_with_test_data(self, basic_config, mock_worker):
        basic_config.update({
            'test_goals': ['test goal'],
            'test_targets': ['test target'],
            'test_workers': [mock_worker]
        })
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.test_goals == ['test goal']
        assert attack.test_targets == ['test target']
        assert len(attack.test_workers) == 1

    def test_with_logfile(self, basic_config, tmp_path):
        logfile = tmp_path / "test.json"
        basic_config['logfile'] = str(logfile)
        
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.logfile == str(logfile)
        assert logfile.exists()
        
        with open(logfile) as f:
            log_data = json.load(f)
        
        assert 'params' in log_data
        assert log_data['params']['goals'] == ['goal1', 'goal2']
        assert log_data['params']['progressive_goals'] is True

    def test_multiple_workers(self, mock_worker, mock_managers):
        workers = [mock_worker, mock_worker]
        attack = ProgressiveMultiPromptAttack(
            goals=['goal'],
            targets=['target'],
            workers=workers,
            managers=mock_managers
        )
        
        assert len(attack.workers) == 2


class TestProgressiveAttackRun:
    def test_run_basic_progressive_goals(self, basic_config):
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        # Mock MPA instances
        mpa_instances = [MagicMock(), MagicMock()]
        for i, mpa in enumerate(mpa_instances):
            mpa.control = f"control_{i}"
            mpa.run = MagicMock()
        
        basic_config['managers']['MPA'].side_effect = mpa_instances
        
        attack.run(n_steps=10, verbose=False)
        
        # Should create 2 MPA instances (one for each goal progressively)
        assert basic_config['managers']['MPA'].call_count == 2
        
        # Each MPA should run
        for mpa in mpa_instances:
            mpa.run.assert_called_once()

    def test_run_non_progressive_goals(self, basic_config):
        basic_config['progressive_goals'] = False
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        mpa_instance = MagicMock()
        mpa_instance.control = "final_control"
        mpa_instance.run = MagicMock()
        basic_config['managers']['MPA'].return_value = mpa_instance
        
        attack.run(n_steps=10, verbose=False)
        
        # Should create only 1 MPA instance (all goals at once)
        assert basic_config['managers']['MPA'].call_count == 1
        
        # Check that all goals were passed to MPA
        call_args = basic_config['managers']['MPA'].call_args[0]
        assert call_args[0] == ['goal1', 'goal2']  # All goals
        assert call_args[1] == ['target1', 'target2']  # All targets

    def test_run_progressive_models(self, mock_worker, mock_managers):
        workers = [mock_worker, MagicMock(), MagicMock()]  # 3 workers
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1', 'target2'],
            workers=workers,
            progressive_models=True,
            progressive_goals=False,  # Test models progression only
            managers=mock_managers
        )
        
        mpa_instances = [MagicMock(), MagicMock(), MagicMock()]
        for i, mpa in enumerate(mpa_instances):
            mpa.control = f"control_{i}"
            mpa.run = MagicMock()
        
        mock_managers['MPA'].side_effect = mpa_instances
        
        attack.run(n_steps=10, verbose=False)
        
        # Should create 3 MPA instances (one for each worker progressively)
        assert mock_managers['MPA'].call_count == 3

    def test_run_non_progressive_models(self, mock_worker, mock_managers):
        workers = [mock_worker, MagicMock()]
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1', 'target2'],
            workers=workers,
            progressive_models=False,
            managers=mock_managers
        )
        
        mpa_instance = MagicMock()
        mpa_instance.control = "final_control"
        mpa_instance.run = MagicMock()
        mock_managers['MPA'].return_value = mpa_instance
        
        attack.run(n_steps=10, verbose=False)
        
        # Should create only 1 MPA instance (all workers at once)
        assert mock_managers['MPA'].call_count == 1
        
        # Check that all workers were passed
        call_args = mock_managers['MPA'].call_args[0]
        assert len(call_args[2]) == 2  # All workers

    def test_run_both_progressive(self, mock_worker, mock_managers):
        workers = [mock_worker, MagicMock()]
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1', 'target2'],
            workers=workers,
            progressive_goals=True,
            progressive_models=True,
            managers=mock_managers
        )
        
        # Should create goals * workers = 2 * 2 = 4 MPA instances
        mpa_instances = [MagicMock() for _ in range(4)]
        for i, mpa in enumerate(mpa_instances):
            mpa.control = f"control_{i}"
            mpa.run = MagicMock()
        
        mock_managers['MPA'].side_effect = mpa_instances
        
        attack.run(n_steps=10, verbose=False)
        
        assert mock_managers['MPA'].call_count == 4

    def test_run_neither_progressive(self, mock_worker, mock_managers):
        workers = [mock_worker, MagicMock()]
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1', 'target2'],
            workers=workers,
            progressive_goals=False,
            progressive_models=False,
            managers=mock_managers
        )
        
        mpa_instance = MagicMock()
        mpa_instance.control = "final_control"
        mpa_instance.run = MagicMock()
        mock_managers['MPA'].return_value = mpa_instance
        
        attack.run(n_steps=10, verbose=False)
        
        # Should create only 1 MPA instance (all goals and workers at once)
        assert mock_managers['MPA'].call_count == 1

    def test_run_control_transfer(self, basic_config):
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        # Mock MPA instances with different controls
        mpa_instances = [MagicMock(), MagicMock()]
        mpa_instances[0].control = "control_1"
        mpa_instances[0].run = MagicMock()
        mpa_instances[1].control = "control_2"
        mpa_instances[1].run = MagicMock()
        
        basic_config['managers']['MPA'].side_effect = mpa_instances
        
        attack.run(n_steps=10, verbose=False)
        
        # Final control should be from last MPA
        assert attack.control == "control_2"

    def test_run_with_all_parameters(self, basic_config):
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        mpa_instance = MagicMock()
        mpa_instance.control = "final_control"
        mpa_instance.run = MagicMock()
        basic_config['managers']['MPA'].return_value = mpa_instance
        
        attack.run(
            n_steps=20,
            batch_size=64,
            topk=128,
            temp=0.5,
            allow_non_ascii=False,
            target_weight=1.0,
            control_weight=0.5,
            anneal=False,
            test_steps=25,
            incr_control=False,
            stop_on_success=False,
            verbose=True,
            filter_cand=False
        )
        
        # Check that parameters were passed to MPA.run
        call_args = mpa_instance.run.call_args[1]
        assert call_args['n_steps'] == 20
        assert call_args['batch_size'] == 64
        assert call_args['topk'] == 128

    def test_run_with_logfile_updates(self, basic_config, tmp_path):
        logfile = tmp_path / "test.json"
        basic_config['logfile'] = str(logfile)
        
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        mpa_instance = MagicMock()
        mpa_instance.control = "final_control"
        mpa_instance.run = MagicMock()
        basic_config['managers']['MPA'].return_value = mpa_instance
        
        attack.run(n_steps=15, batch_size=32, verbose=False)
        
        # Check logfile was updated
        with open(logfile) as f:
            log_data = json.load(f)
        
        assert log_data['params']['n_steps'] == 15
        assert log_data['params']['batch_size'] == 32

    def test_run_single_goal(self, mock_worker, mock_managers):
        attack = ProgressiveMultiPromptAttack(
            goals=['single_goal'],
            targets=['single_target'],
            workers=[mock_worker],
            progressive_goals=True,
            managers=mock_managers
        )
        
        mpa_instance = MagicMock()
        mpa_instance.control = "final_control"
        mpa_instance.run = MagicMock()
        mock_managers['MPA'].return_value = mpa_instance
        
        attack.run(n_steps=10, verbose=False)
        
        # Should still work with single goal
        assert mock_managers['MPA'].call_count == 1

    def test_run_empty_goals(self, mock_worker, mock_managers):
        attack = ProgressiveMultiPromptAttack(
            goals=[],
            targets=[],
            workers=[mock_worker],
            managers=mock_managers
        )
        
        control = attack.run(n_steps=10, verbose=False)
        
        # Should handle empty goals gracefully
        assert control == attack.control
        mock_managers['MPA'].assert_not_called()


class TestProgressiveAttackEdgeCases:
    def test_none_logfile(self, basic_config):
        basic_config['logfile'] = None
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        assert attack.logfile is None

    def test_mismatched_goals_targets(self, mock_worker, mock_managers):
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1'],  # Fewer targets
            workers=[mock_worker],
            managers=mock_managers
        )
        
        assert len(attack.goals) == 2
        assert len(attack.targets) == 1

    def test_no_workers(self, mock_managers):
        attack = ProgressiveMultiPromptAttack(
            goals=['goal'],
            targets=['target'],
            workers=[],
            managers=mock_managers
        )
        
        control = attack.run(n_steps=10, verbose=False)
        
        # Should handle no workers
        assert control == attack.control
        mock_managers['MPA'].assert_not_called()

    def test_missing_managers(self, mock_worker):
        with pytest.raises((KeyError, TypeError)):
            ProgressiveMultiPromptAttack(
                goals=['goal'],
                targets=['target'],
                workers=[mock_worker],
                managers=None
            )


class TestProgressiveAttackIntegration:
    def test_complete_progressive_workflow(self, mock_worker, mock_managers, tmp_path):
        # Test complete workflow with both progressions
        logfile = tmp_path / "full_test.json"
        workers = [mock_worker, MagicMock()]
        
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1', 'goal2'],
            targets=['target1', 'target2'],
            workers=workers,
            progressive_goals=True,
            progressive_models=True,
            logfile=str(logfile),
            test_goals=['test_goal'],
            test_targets=['test_target'],
            test_workers=[mock_worker],
            managers=mock_managers
        )
        
        # Mock 4 MPA instances (2 goals * 2 workers)
        mpa_instances = []
        for i in range(4):
            mpa = MagicMock()
            mpa.control = f"control_{i}"
            mpa.run = MagicMock()
            mpa_instances.append(mpa)
        
        mock_managers['MPA'].side_effect = mpa_instances
        
        final_control = attack.run(
            n_steps=10,
            batch_size=16,
            test_steps=5,
            verbose=True
        )
        
        # Verify complete execution
        assert mock_managers['MPA'].call_count == 4
        assert final_control == "control_3"  # Last control
        assert attack.control == "control_3"
        
        # Verify logfile
        assert logfile.exists()
        with open(logfile) as f:
            log_data = json.load(f)
        assert log_data['params']['n_steps'] == 10

    def test_mpa_call_structure_progressive_goals(self, basic_config):
        basic_config['progressive_models'] = False  # Only test goals progression
        attack = ProgressiveMultiPromptAttack(**basic_config)
        
        mpa_instances = [MagicMock(), MagicMock()]
        for i, mpa in enumerate(mpa_instances):
            mpa.control = f"control_{i}"
            mpa.run = MagicMock()
        
        basic_config['managers']['MPA'].side_effect = mpa_instances
        
        attack.run(n_steps=10, verbose=False)
        
        # Check MPA call arguments
        calls = basic_config['managers']['MPA'].call_args_list
        assert len(calls) == 2
        
        # First call: goal1 only
        assert calls[0][0][0] == ['goal1']
        assert calls[0][0][1] == ['target1']
        
        # Second call: goal1 + goal2
        assert calls[1][0][0] == ['goal1', 'goal2']
        assert calls[1][0][1] == ['target1', 'target2']

    def test_mpa_call_structure_progressive_models(self, mock_worker, mock_managers):
        workers = [mock_worker, MagicMock()]
        attack = ProgressiveMultiPromptAttack(
            goals=['goal1'],
            targets=['target1'],
            workers=workers,
            progressive_goals=False,  # Only test models progression
            progressive_models=True,
            managers=mock_managers
        )
        
        mpa_instances = [MagicMock(), MagicMock()]
        for i, mpa in enumerate(mpa_instances):
            mpa.control = f"control_{i}"
            mpa.run = MagicMock()
        
        mock_managers['MPA'].side_effect = mpa_instances
        
        attack.run(n_steps=10, verbose=False)
        
        # Check MPA call arguments
        calls = mock_managers['MPA'].call_args_list
        assert len(calls) == 2
        
        # First call: worker1 only
        assert len(calls[0][0][2]) == 1  # workers
        
        # Second call: worker1 + worker2
        assert len(calls[1][0][2]) == 2  # workers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])