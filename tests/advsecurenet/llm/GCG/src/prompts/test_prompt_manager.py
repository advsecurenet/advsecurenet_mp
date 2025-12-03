import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from advsecurenet.llm.GCG.src.prompts.prompt_manager import PromptManager


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded output"
    return tokenizer


@pytest.fixture
def mock_conv_template():
    template = MagicMock()
    template.name = "test-template"
    template.roles = ("User", "Assistant")
    return template


@pytest.fixture
def mock_attack_prompt():
    prompt = MagicMock()
    prompt.generate.return_value = torch.tensor([1, 2, 3, 4])
    prompt.test.return_value = [0.5, 0.3]
    prompt.test_loss.return_value = 0.25
    prompt.grad.return_value = torch.tensor([0.1, 0.2, 0.3])
    prompt.logits.return_value = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    prompt.target_loss.return_value = torch.tensor([[0.2], [0.3]])
    prompt.control_loss.return_value = torch.tensor([[0.1], [0.15]])
    prompt.control_str = "initial control"
    prompt.control_toks = torch.tensor([1, 2, 3])
    return prompt


@pytest.fixture
def basic_managers(mock_attack_prompt):
    managers = {"AP": lambda *args, **kwargs: mock_attack_prompt}
    return managers


class TestPromptManagerInit:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_init_basic(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([100, 200])

        goals = ["goal1", "goal2"]
        targets = ["target1", "target2"]

        manager = PromptManager(
            goals=goals,
            targets=targets,
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        assert len(manager._prompts) == 2
        assert manager.tokenizer == mock_tokenizer
        assert torch.equal(manager._nonascii_toks, torch.tensor([100, 200]))

    def test_init_mismatched_lengths(
        self, mock_tokenizer, mock_conv_template, basic_managers
    ):
        goals = ["goal1", "goal2"]
        targets = ["target1"]  # Different length

        with pytest.raises(ValueError, match="Length of goals and targets must match"):
            PromptManager(
                goals=goals,
                targets=targets,
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=basic_managers,
            )

    def test_init_empty_lists(self, mock_tokenizer, mock_conv_template, basic_managers):
        with pytest.raises(
            ValueError, match="Must provide at least one goal, target pair"
        ):
            PromptManager(
                goals=[],
                targets=[],
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=basic_managers,
            )

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_init_custom_parameters(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        custom_control = "custom control init"
        custom_prefixes = ["Sorry", "Cannot"]

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            control_init=custom_control,
            test_prefixes=custom_prefixes,
            managers=basic_managers,
        )

        assert len(manager._prompts) == 1

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_prompt_creation_parameters(
        self, mock_nonascii, mock_tokenizer, mock_conv_template
    ):
        mock_nonascii.return_value = torch.tensor([])

        mock_ap_class = MagicMock()
        mock_ap_instance = MagicMock()
        mock_ap_class.return_value = mock_ap_instance
        managers = {"AP": mock_ap_class}

        control_init = "test control"
        test_prefixes = ["prefix1", "prefix2"]

        PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            control_init=control_init,
            test_prefixes=test_prefixes,
            managers=managers,
        )

        # Verify AttackPrompt was called with correct parameters
        mock_ap_class.assert_called_once_with(
            "goal1",
            "target1",
            mock_tokenizer,
            mock_conv_template,
            control_init,
            test_prefixes,
        )


class TestPromptManagerGenerate:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_generate_default_config(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()
        model.generation_config.max_new_tokens = 16

        result = manager.generate(model)

        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.generate.assert_called_with(model, model.generation_config)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_generate_custom_config(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()
        custom_config = MagicMock()

        result = manager.generate(model, custom_config)

        assert len(result) == 1
        manager._prompts[0].generate.assert_called_with(model, custom_config)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_generate_str(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()
        mock_tokenizer.decode.return_value = "decoded text"

        with patch.object(manager, "generate", return_value=[torch.tensor([1, 2, 3])]):
            result = manager.generate_str(model)

            assert result == ["decoded text"]
            # Check that decode was called (can't use exact tensor comparison in mock)
            assert mock_tokenizer.decode.called


class TestPromptManagerTest:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_test_method(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()

        result = manager.test(model)

        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.test.assert_called_with(model, None)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_test_with_config(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()
        gen_config = MagicMock()

        result = manager.test(model, gen_config)

        manager._prompts[0].test.assert_called_with(model, gen_config)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_test_loss(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()

        result = manager.test_loss(model)

        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.test_loss.assert_called_with(model)


class TestPromptManagerGradAndLogits:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_grad(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])

        # Create separate mock prompts with different grad returns
        prompt1 = MagicMock()
        prompt1.grad.return_value = torch.tensor([0.1, 0.2])
        prompt2 = MagicMock()
        prompt2.grad.return_value = torch.tensor([0.3, 0.4])

        managers = {
            "AP": lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2
        }

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers,
        )

        model = MagicMock()

        result = manager.grad(model)

        # Should sum all gradients
        expected = torch.tensor([0.1, 0.2]) + torch.tensor([0.3, 0.4])
        assert torch.equal(result, expected)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_logits_without_ids(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()
        test_controls = ["control1", "control2"]

        result = manager.logits(model, test_controls, return_ids=False)

        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.logits.assert_called_with(model, test_controls, False)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_logits_with_ids(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])

        # Create prompts that return tuples when return_ids=True
        prompt1 = MagicMock()
        prompt1.logits.return_value = (torch.tensor([[0.1, 0.9]]), torch.tensor([1, 2]))
        prompt2 = MagicMock()
        prompt2.logits.return_value = (torch.tensor([[0.7, 0.3]]), torch.tensor([3, 4]))

        managers = {
            "AP": lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2
        }

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers,
        )

        model = MagicMock()

        logits, ids = manager.logits(model, return_ids=True)

        assert len(logits) == 2
        assert len(ids) == 2


class TestPromptManagerLoss:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_target_loss(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])

        # Create prompts with predictable target_loss returns
        prompt1 = MagicMock()
        prompt1.target_loss.return_value = torch.tensor(
            [[0.2, 0.3], [0.4, 0.5]]
        )  # shape: (2, 2)
        prompt2 = MagicMock()
        prompt2.target_loss.return_value = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]]
        )  # shape: (2, 2)

        managers = {
            "AP": lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2
        }

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers,
        )

        logits = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.7, 0.3]])]
        ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]

        result = manager.target_loss(logits, ids)

        # Should call target_loss on each prompt and concatenate/mean
        assert result.shape[0] == 2  # batch dimension

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_control_loss(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])

        prompt1 = MagicMock()
        prompt1.control_loss.return_value = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        prompt2 = MagicMock()
        prompt2.control_loss.return_value = torch.tensor([[0.2, 0.3], [0.4, 0.5]])

        managers = {
            "AP": lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2
        }

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers,
        )

        logits = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.7, 0.3]])]
        ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]

        result = manager.control_loss(logits, ids)

        # Should call control_loss on each prompt
        for i, (prompt, logit, id_tensor) in enumerate(
            zip(manager._prompts, logits, ids)
        ):
            prompt.control_loss.assert_called_with(logit, id_tensor)


class TestPromptManagerEdgeCases:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_single_prompt(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["single_goal"],
            targets=["single_target"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()

        # Test all methods with single prompt
        generate_result = manager.generate(model)
        test_result = manager.test(model)
        test_loss_result = manager.test_loss(model)
        grad_result = manager.grad(model)

        assert len(generate_result) == 1
        assert len(test_result) == 1
        assert len(test_loss_result) == 1

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_empty_nonascii_toks(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])  # Empty tensor

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        assert manager._nonascii_toks.numel() == 0

    def test_none_managers(self, mock_tokenizer, mock_conv_template):
        with pytest.raises(TypeError):  # Should fail when accessing None['AP']
            PromptManager(
                goals=["goal1"],
                targets=["target1"],
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=None,
            )

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_missing_ap_manager(
        self, mock_nonascii, mock_tokenizer, mock_conv_template
    ):
        mock_nonascii.return_value = torch.tensor([])

        invalid_managers = {"OTHER": MagicMock()}  # Missing 'AP'

        with pytest.raises(KeyError):
            PromptManager(
                goals=["goal1"],
                targets=["target1"],
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=invalid_managers,
            )


class TestPromptManagerIntegration:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_full_workflow(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([100])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        model = MagicMock()
        model.generation_config.max_new_tokens = 16

        # Test complete workflow
        generate_result = manager.generate(model)
        generate_str_result = manager.generate_str(model)
        test_result = manager.test(model)
        test_loss_result = manager.test_loss(model)
        grad_result = manager.grad(model)
        logits_result = manager.logits(model)

        # Verify all methods work
        assert len(generate_result) == 2
        assert len(generate_str_result) == 2
        assert len(test_result) == 2
        assert len(test_loss_result) == 2
        assert len(logits_result) == 2


class TestPromptManagerMissingCoverage:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_sample_control_not_implemented(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        with pytest.raises(
            NotImplementedError, match="Sampling control tokens not yet implemented"
        ):
            manager.sample_control()

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_dunder_methods(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        # Test __len__
        assert len(manager) == 2

        # Test __getitem__
        assert manager[0] == manager._prompts[0]
        assert manager[1] == manager._prompts[1]

        # Test __iter__
        prompts_list = list(manager)
        assert len(prompts_list) == 2
        assert prompts_list == manager._prompts

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_control_properties(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        mock_nonascii.return_value = torch.tensor([])

        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        # Test control_str getter
        assert manager.control_str == "initial control"

        # Test control_toks getter
        assert torch.equal(manager.control_toks, torch.tensor([1, 2, 3]))

        # Test control_str setter
        manager.control_str = "new control"
        for prompt in manager._prompts:
            assert prompt.control_str == "new control"

        # Test control_toks setter
        new_toks = torch.tensor([4, 5, 6])
        manager.control_toks = new_toks
        for prompt in manager._prompts:
            assert torch.equal(prompt.control_toks, new_toks)

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_disallowed_toks_property(
        self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers
    ):
        test_nonascii = torch.tensor([100, 200, 300])
        mock_nonascii.return_value = test_nonascii

        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers,
        )

        assert torch.equal(manager.disallowed_toks, test_nonascii)


class TestPromptManagerMultiPromptAttack:
    """Test the MultiPromptAttack class that's also in the same file."""

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_multi_prompt_attack_init(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        # Create mock workers
        worker1 = MagicMock()
        worker1.model = MagicMock()
        worker1.model.name_or_path = "model1"
        worker1.tokenizer = MagicMock()
        worker1.conv_template = MagicMock()

        worker2 = MagicMock()
        worker2.model = MagicMock()
        worker2.model.name_or_path = "model2"
        worker2.tokenizer = MagicMock()
        worker2.conv_template = MagicMock()

        workers = [worker1, worker2]

        # Create mock managers
        mock_pm = MagicMock()
        managers = {"PM": lambda *args, **kwargs: mock_pm}

        attack = MultiPromptAttack(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            workers=workers,
            managers=managers,
        )

        assert attack.goals == ["goal1", "goal2"]
        assert attack.targets == ["target1", "target2"]
        assert attack.workers == workers
        assert len(attack.models) == 2
        assert len(attack.prompts) == 2

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_multi_prompt_attack_control_properties(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        # Create mock workers
        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()

        # Create mock prompt managers
        pm1 = MagicMock()
        pm1.control_str = "control1"
        pm1.control_toks = torch.tensor([1, 2])

        pm2 = MagicMock()
        pm2.control_str = "control2"
        pm2.control_toks = torch.tensor([3, 4])

        managers = {"PM": lambda *args, **kwargs: pm1 if len(args) == 7 else pm2}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        # Test control_str getter/setter
        assert attack.control_str == "control1"
        attack.control_str = "new_control"
        pm1.control_str = "new_control"  # Simulate the setter effect

        # Test control_toks getter/setter
        control_toks_list = attack.control_toks
        assert len(control_toks_list) == 1

        # Test setting control_toks
        new_control = [torch.tensor([5, 6])]
        attack.control_toks = new_control

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_multi_prompt_attack_control_toks_validation(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()

        pm = MagicMock()
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"],
            targets=["target1"],
            workers=[worker, worker],  # 2 workers
            managers=managers,
        )

        # Should raise error if control tokens length doesn't match prompts length
        with pytest.raises(
            ValueError, match="Must provide control tokens for each tokenizer"
        ):
            attack.control_toks = [torch.tensor([1, 2])]  # Only 1 control for 2 prompts

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_get_filtered_cands(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        # Create mock worker with tokenizer
        worker = MagicMock()
        worker.model = MagicMock()
        # Provide enough decode return values for multiple calls
        worker.tokenizer.decode.side_effect = [
            "cand1",
            "cand2",
            "curr_control",
            "cand3",  # First 4 calls for filter test
            "cand1",
            "cand2",
            "curr_control",
            "cand3",  # Next 4 calls for no-filter test
        ]
        worker.tokenizer.return_value.input_ids = [1, 2, 3]  # Mock tokenized length
        worker.conv_template = MagicMock()

        pm = MagicMock()
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        control_cand = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        # Test with filtering
        cands = attack.get_filtered_cands(
            0, control_cand, filter_cand=True, curr_control="curr_control"
        )
        assert len(cands) == 4

        # Test without filtering
        cands_no_filter = attack.get_filtered_cands(0, control_cand, filter_cand=False)
        assert len(cands_no_filter) == 4

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_multi_prompt_attack_step_not_implemented(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()

        pm = MagicMock()
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        with pytest.raises(
            NotImplementedError, match="Attack step function not yet implemented"
        ):
            attack.step()

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_multi_prompt_attack_test_methods(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        # Create mock worker
        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()
        worker.results.get.return_value = [True, False, 0.5]  # jailbroken, match, loss

        pm = MagicMock()
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        # Test the test method
        model_tests_jb, model_tests_mb, model_tests_loss = attack.test(
            [worker], [pm], include_loss=True
        )

        assert len(model_tests_jb) == 1
        assert len(model_tests_mb) == 1
        assert len(model_tests_loss) == 1

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_parse_results(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()

        pm = MagicMock()
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1", "goal2"],  # 2 goals
            targets=["target1", "target2"],
            workers=[worker],  # 1 worker
            test_goals=["test_goal1"],  # 1 test goal
            test_targets=["test_target1"],
            test_workers=[worker],  # 1 test worker
            managers=managers,
        )

        # Create test results matrix: workers x goals
        # 2 workers (1 regular + 1 test) x 3 goals (2 regular + 1 test)
        results = np.array([[1, 0, 1], [0, 1, 0]])

        id_id, id_od, od_id, od_od = attack.parse_results(results)

        # id_id: in-distribution workers on in-distribution goals = results[:1, :2].sum() = 1
        # id_od: in-distribution workers on out-of-distribution goals = results[:1, 2:].sum() = 1
        # od_id: out-of-distribution workers on in-distribution goals = results[1:, :2].sum() = 1
        # od_od: out-of-distribution workers on out-of-distribution goals = results[1:, 2:].sum() = 0
        assert id_id == 1
        assert id_od == 1
        assert od_id == 1
        assert od_od == 0


class TestMultiPromptAttackAdvancedMethods:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    @patch("time.time")
    def test_run_method(self, mock_time, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])
        mock_time.side_effect = [0.0, 1.0, 2.0, 3.0]  # Mock time progression

        # Create mock worker
        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()
        worker.results.get.return_value = [
            False,
            False,
            0.5,
        ]  # Not jailbroken initially

        # Create mock prompt manager with step method
        pm = MagicMock()
        pm.step.return_value = ("new_control", 0.5)
        pm.control_str = "initial_control"
        pm.test_all.return_value = ([[False]], [[False]], [0.5])

        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        # Mock the step method to return control and loss
        attack.step = MagicMock(return_value=("optimized_control", 0.3))

        # Test run with minimal parameters - should not stop early since jailbreak fails
        result = attack.run(n_steps=2, verbose=False, stop_on_success=False)

        # Should return the final control, loss, and steps
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_test_all_method(self, mock_nonascii):
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        # Create mock workers
        main_worker = MagicMock()
        main_worker.model = MagicMock()
        main_worker.tokenizer = MagicMock()
        main_worker.conv_template = MagicMock()

        test_worker = MagicMock()
        test_worker.model = MagicMock()
        test_worker.tokenizer = MagicMock()
        test_worker.conv_template = MagicMock()

        # Create mock prompt managers
        pm = MagicMock()

        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"],
            targets=["target1"],
            workers=[main_worker],
            test_goals=["test_goal1"],
            test_targets=["test_target1"],
            test_workers=[test_worker],
            managers=managers,
        )

        # Mock the test method
        attack.test = MagicMock(
            return_value=([[True, False]], [[True, False]], [0.1, 0.2])
        )

        # Test test_all
        result = attack.test_all()

        # Verify result structure
        assert len(result) == 3
        jb_results, mb_results, loss_results = result
        assert isinstance(jb_results, list)
        assert isinstance(mb_results, list)
        assert isinstance(loss_results, list)


class TestPromptManagerRunMethodEdgeCases:
    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    @patch("time.time")
    @patch("random.random")
    def test_run_with_annealing(self, mock_random, mock_time, mock_nonascii):
        """Test the run method with annealing enabled."""
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])
        mock_time.return_value = 1.0
        mock_random.return_value = 0.5  # For annealing probability

        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()
        worker.results.get.return_value = [False, False, 0.8]

        pm = MagicMock()
        pm.control_str = "initial_control"
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        # Mock step method to return different controls and losses
        attack.step = MagicMock(
            side_effect=[
                ("control1", 0.9),  # Higher loss - should not be kept with annealing
                ("control2", 0.7),  # Lower loss - should be kept
            ]
        )

        # Test with annealing enabled
        result = attack.run(
            n_steps=2, anneal=True, verbose=False, stop_on_success=False
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks")
    def test_run_with_stop_on_success(self, mock_nonascii):
        """Test early stopping when jailbreak succeeds."""
        from advsecurenet.llm.GCG.src.prompts.prompt_manager import MultiPromptAttack

        mock_nonascii.return_value = torch.tensor([])

        worker = MagicMock()
        worker.model = MagicMock()
        worker.tokenizer = MagicMock()
        worker.conv_template = MagicMock()

        pm = MagicMock()
        pm.control_str = "initial_control"
        managers = {"PM": lambda *args, **kwargs: pm}

        attack = MultiPromptAttack(
            goals=["goal1"], targets=["target1"], workers=[worker], managers=managers
        )

        # Mock test method to return successful jailbreak
        attack.test = MagicMock(return_value=([[True]], [[True]], [0.1]))
        attack.step = MagicMock(return_value=("success_control", 0.1))

        # Should stop early due to success
        result = attack.run(n_steps=10, stop_on_success=True, verbose=False)

        assert isinstance(result, tuple)
        assert len(result) == 3
        # Should have stopped before completing all steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
