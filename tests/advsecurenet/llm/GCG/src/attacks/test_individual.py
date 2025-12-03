import pytest
import json
import tempfile
import os
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from advsecurenet.llm.GCG.src.attacks.individual import IndividualPromptAttack


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
def mock_managers():
    mpa_mock = MagicMock()
    attack_mock = MagicMock()
    attack_mock.run = MagicMock()
    mpa_mock.return_value = attack_mock
    return {"AP": MagicMock(), "PM": MagicMock(), "MPA": mpa_mock}


@pytest.fixture
def basic_config(mock_worker, mock_managers):
    return {
        "goals": ["test goal"],
        "targets": ["test target"],
        "workers": [mock_worker],
        "managers": mock_managers,
    }


class TestIndividualPromptAttackInit:
    def test_basic_initialization(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)

        assert attack.goals == ["test goal"]
        assert attack.targets == ["test target"]
        assert len(attack.workers) == 1
        assert attack.control == "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        assert attack.control_init == "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        assert attack.test_prefixes == [
            "I'm sorry",
            "I apologize",
            "As an",
            "I'm just",
            "I'm an",
            "I cannot",
            "Hello!",
        ]

    def test_custom_control_init(self, basic_config):
        basic_config["control_init"] = "custom control"
        attack = IndividualPromptAttack(**basic_config)

        assert attack.control == "custom control"
        assert attack.control_init == "custom control"

    def test_custom_test_prefixes(self, basic_config):
        custom_prefixes = ["Sorry", "Cannot"]
        basic_config["test_prefixes"] = custom_prefixes
        attack = IndividualPromptAttack(**basic_config)

        assert attack.test_prefixes == custom_prefixes

    def test_with_test_data(self, basic_config, mock_worker):
        basic_config.update(
            {
                "test_goals": ["test goal 2"],
                "test_targets": ["test target 2"],
                "test_workers": [mock_worker],
            }
        )
        attack = IndividualPromptAttack(**basic_config)

        assert attack.test_goals == ["test goal 2"]
        assert attack.test_targets == ["test target 2"]
        assert len(attack.test_workers) == 1

    def test_with_logfile(self, basic_config, tmp_path):
        logfile = tmp_path / "test.json"
        basic_config["logfile"] = str(logfile)

        attack = IndividualPromptAttack(**basic_config)

        assert attack.logfile == str(logfile)
        assert logfile.exists()

        with open(logfile) as f:
            log_data = json.load(f)

        assert "params" in log_data
        assert "controls" in log_data
        assert "losses" in log_data
        assert "runtimes" in log_data
        assert "tests" in log_data
        assert log_data["params"]["goals"] == ["test goal"]

    def test_logfile_with_test_workers(self, basic_config, mock_worker, tmp_path):
        logfile = tmp_path / "test.json"
        basic_config["logfile"] = str(logfile)
        basic_config["test_workers"] = [mock_worker]

        attack = IndividualPromptAttack(**basic_config)

        with open(logfile) as f:
            log_data = json.load(f)

        assert len(log_data["params"]["test_models"]) == 1
        assert log_data["params"]["test_models"][0]["model_path"] == "test-model"

    def test_mpa_kwargs_filtering(self, basic_config):
        basic_config.update(
            {"mpa_batch_size": 32, "mpa_lr": 0.01, "other_param": "ignored"}
        )

        attack = IndividualPromptAttack(**basic_config)

        assert "batch_size" in attack.mpa_kewargs
        assert "lr" in attack.mpa_kewargs
        assert "other_param" not in attack.mpa_kewargs
        assert attack.mpa_kewargs["batch_size"] == 32


class TestFilterMpaKwargs:
    def test_filter_mpa_kwargs_basic(self):
        kwargs = {
            "mpa_batch_size": 32,
            "mpa_lr": 0.01,
            "other_param": "ignored",
            "mpa_n_steps": 10,
        }

        result = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)

        assert result == {"batch_size": 32, "lr": 0.01, "n_steps": 10}

    def test_filter_mpa_kwargs_empty(self):
        kwargs = {"other_param": "ignored"}
        result = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)
        assert result == {}

    def test_filter_mpa_kwargs_no_mpa_prefix(self):
        kwargs = {"batch_size": 32, "lr": 0.01}
        result = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)
        assert result == {}


class TestIndividualPromptAttackRun:
    def test_run_basic(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)

        control, n_steps = attack.run(n_steps=1, batch_size=1, verbose=False)

        assert control == attack.control
        assert n_steps == 1
        basic_config["managers"]["MPA"].assert_called_once()

    def test_run_with_logfile(self, basic_config, tmp_path):
        logfile = tmp_path / "test.json"
        basic_config["logfile"] = str(logfile)

        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=5, batch_size=32, test_steps=10, verbose=False)

        with open(logfile) as f:
            log_data = json.load(f)

        assert log_data["params"]["n_steps"] == 5
        assert log_data["params"]["batch_size"] == 32
        assert log_data["params"]["test_steps"] == 10

    def test_run_multiple_goals(self, basic_config):
        basic_config["goals"] = ["goal1", "goal2", "goal3"]
        basic_config["targets"] = ["target1", "target2", "target3"]

        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=1, verbose=False)

        assert basic_config["managers"]["MPA"].call_count == 3

    def test_run_with_all_parameters(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)

        attack.run(
            n_steps=10,
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
            filter_cand=False,
        )

        mpa_instance = basic_config["managers"]["MPA"].return_value
        mpa_instance.run.assert_called_once()
        call_args = mpa_instance.run.call_args[1]

        assert call_args["n_steps"] == 10
        assert call_args["batch_size"] == 64
        assert call_args["topk"] == 128
        assert call_args["temp"] == 0.5
        assert call_args["allow_non_ascii"] is False
        assert call_args["target_weight"] == 1.0
        assert call_args["control_weight"] == 0.5
        assert call_args["anneal"] is False
        assert call_args["test_steps"] == 25
        # incr_control is not passed to MPA.run() method - it's only used for logfile
        assert call_args["stop_on_success"] is False
        assert call_args["filter_cand"] is False

    def test_run_logfile_update_all_params(self, basic_config, tmp_path):
        logfile = tmp_path / "test.json"
        basic_config["logfile"] = str(logfile)

        attack = IndividualPromptAttack(**basic_config)
        attack.run(
            n_steps=15,
            test_steps=20,
            batch_size=128,
            topk=64,
            temp=1.5,
            allow_non_ascii=True,
            target_weight=2.0,
            control_weight=1.5,
            anneal=True,
            incr_control=True,
            stop_on_success=True,
            verbose=False,
        )

        with open(logfile) as f:
            log_data = json.load(f)

        params = log_data["params"]
        assert params["n_steps"] == 15
        assert params["test_steps"] == 20
        assert params["batch_size"] == 128
        assert params["topk"] == 64
        assert params["temp"] == 1.5
        assert params["allow_non_ascii"] is True
        assert params["target_weight"] == 2.0
        assert params["control_weight"] == 1.5
        assert params["anneal"] is True
        assert params["incr_control"] is True
        assert params["stop_on_success"] is True

    def test_run_mpa_creation_args(self, basic_config):
        basic_config["control_init"] = "custom control"
        basic_config["test_goals"] = ["test goal"]
        basic_config["test_targets"] = ["test target"]
        basic_config["test_workers"] = [MagicMock()]

        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=1, verbose=False)

        mpa_call_args = basic_config["managers"]["MPA"].call_args[0]
        mpa_call_kwargs = basic_config["managers"]["MPA"].call_args[1]

        assert mpa_call_args[0] == ["test goal"]  # goals slice
        assert mpa_call_args[1] == ["test target"]  # targets slice
        assert mpa_call_args[2] == basic_config["workers"]
        assert mpa_call_args[3] == "custom control"
        assert mpa_call_args[4] == attack.test_prefixes

    def test_run_single_goal_slicing(self, basic_config):
        basic_config["goals"] = ["goal1", "goal2"]
        basic_config["targets"] = ["target1", "target2"]

        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=1, verbose=False)

        # Check that each call gets only one goal/target
        calls = basic_config["managers"]["MPA"].call_args_list
        assert len(calls) == 2

        # First call
        assert calls[0][0][0] == ["goal1"]
        assert calls[0][0][1] == ["target1"]

        # Second call
        assert calls[1][0][0] == ["goal2"]
        assert calls[1][0][1] == ["target2"]

    def test_run_empty_goals(self, basic_config):
        basic_config["goals"] = []
        basic_config["targets"] = []

        attack = IndividualPromptAttack(**basic_config)
        control, n_steps = attack.run(n_steps=5, verbose=False)

        assert control == attack.control
        assert n_steps == 5
        basic_config["managers"]["MPA"].assert_not_called()


class TestIndividualPromptAttackEdgeCases:
    def test_none_logfile(self, basic_config):
        basic_config["logfile"] = None
        attack = IndividualPromptAttack(**basic_config)

        assert attack.logfile is None

    def test_missing_managers(self, mock_worker):
        # The IndividualPromptAttack doesn't validate managers parameter in __init__
        # It will only fail when trying to access self.managers["MPA"] during run()
        attack = IndividualPromptAttack(
            goals=["test"], targets=["test"], workers=[mock_worker]
            # missing managers
        )
        
        # The error happens when trying to run, not during initialization
        with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
            attack.run(n_steps=1, verbose=False)

    def test_empty_workers_list(self, mock_managers):
        attack = IndividualPromptAttack(
            goals=["test"], targets=["test"], workers=[], managers=mock_managers
        )
        assert attack.workers == []

    def test_mpa_kwargs_with_empty_dict(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)
        assert attack.mpa_kewargs == {}

    def test_run_with_numpy_inf(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=1, verbose=False)

        mpa_instance = basic_config["managers"]["MPA"].return_value
        call_args = mpa_instance.run.call_args[1]
        assert call_args["prev_loss"] == np.inf

    def test_run_anneal_from_zero(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=1, verbose=False)

        mpa_instance = basic_config["managers"]["MPA"].return_value
        call_args = mpa_instance.run.call_args[1]
        assert call_args["anneal_from"] == 0

    def test_run_log_first_true(self, basic_config):
        attack = IndividualPromptAttack(**basic_config)
        attack.run(n_steps=1, verbose=False)

        mpa_instance = basic_config["managers"]["MPA"].return_value
        call_args = mpa_instance.run.call_args[1]
        assert call_args["log_first"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
