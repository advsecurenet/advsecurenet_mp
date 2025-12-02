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
    return prompt


@pytest.fixture
def basic_managers(mock_attack_prompt):
    managers = {'AP': lambda *args, **kwargs: mock_attack_prompt}
    return managers


class TestPromptManagerInit:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_init_basic(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([100, 200])
        
        goals = ["goal1", "goal2"]
        targets = ["target1", "target2"]
        
        manager = PromptManager(
            goals=goals,
            targets=targets,
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        assert len(manager._prompts) == 2
        assert manager.tokenizer == mock_tokenizer
        torch.testing.assert_tensor_equal(manager._nonascii_toks, torch.tensor([100, 200]))

    def test_init_mismatched_lengths(self, mock_tokenizer, mock_conv_template, basic_managers):
        goals = ["goal1", "goal2"]
        targets = ["target1"]  # Different length
        
        with pytest.raises(ValueError, match="Length of goals and targets must match"):
            PromptManager(
                goals=goals,
                targets=targets,
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=basic_managers
            )

    def test_init_empty_lists(self, mock_tokenizer, mock_conv_template, basic_managers):
        with pytest.raises(ValueError, match="Must provide at least one goal, target pair"):
            PromptManager(
                goals=[],
                targets=[],
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=basic_managers
            )

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_init_custom_parameters(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
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
            managers=basic_managers
        )
        
        assert len(manager._prompts) == 1

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_prompt_creation_parameters(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])
        
        mock_ap_class = MagicMock()
        mock_ap_instance = MagicMock()
        mock_ap_class.return_value = mock_ap_instance
        managers = {'AP': mock_ap_class}
        
        control_init = "test control"
        test_prefixes = ["prefix1", "prefix2"]
        
        PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            control_init=control_init,
            test_prefixes=test_prefixes,
            managers=managers
        )
        
        # Verify AttackPrompt was called with correct parameters
        mock_ap_class.assert_called_once_with(
            "goal1",
            "target1",
            mock_tokenizer,
            mock_conv_template,
            control_init,
            test_prefixes
        )


class TestPromptManagerGenerate:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_generate_default_config(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        model.generation_config.max_new_tokens = 16
        
        result = manager.generate(model)
        
        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.generate.assert_called_with(model, model.generation_config)

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_generate_custom_config(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        custom_config = MagicMock()
        
        result = manager.generate(model, custom_config)
        
        assert len(result) == 1
        manager._prompts[0].generate.assert_called_with(model, custom_config)

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_generate_str(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        mock_tokenizer.decode.return_value = "decoded text"
        
        with patch.object(manager, 'generate', return_value=[torch.tensor([1, 2, 3])]):
            result = manager.generate_str(model)
            
            assert result == ["decoded text"]
            mock_tokenizer.decode.assert_called_with(torch.tensor([1, 2, 3]))


class TestPromptManagerTest:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_test_method(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        
        result = manager.test(model)
        
        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.test.assert_called_with(model, None)

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_test_with_config(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        gen_config = MagicMock()
        
        result = manager.test(model, gen_config)
        
        manager._prompts[0].test.assert_called_with(model, gen_config)

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_test_loss(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        
        result = manager.test_loss(model)
        
        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.test_loss.assert_called_with(model)


class TestPromptManagerGradAndLogits:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_grad(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])
        
        # Create separate mock prompts with different grad returns
        prompt1 = MagicMock()
        prompt1.grad.return_value = torch.tensor([0.1, 0.2])
        prompt2 = MagicMock()
        prompt2.grad.return_value = torch.tensor([0.3, 0.4])
        
        managers = {'AP': lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2}
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers
        )
        
        model = MagicMock()
        
        result = manager.grad(model)
        
        # Should sum all gradients
        expected = torch.tensor([0.1, 0.2]) + torch.tensor([0.3, 0.4])
        torch.testing.assert_tensor_equal(result, expected)

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_logits_without_ids(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        model = MagicMock()
        test_controls = ["control1", "control2"]
        
        result = manager.logits(model, test_controls, return_ids=False)
        
        assert len(result) == 2
        for prompt in manager._prompts:
            prompt.logits.assert_called_with(model, test_controls, False)

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_logits_with_ids(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])
        
        # Create prompts that return tuples when return_ids=True
        prompt1 = MagicMock()
        prompt1.logits.return_value = (torch.tensor([[0.1, 0.9]]), torch.tensor([1, 2]))
        prompt2 = MagicMock()
        prompt2.logits.return_value = (torch.tensor([[0.7, 0.3]]), torch.tensor([3, 4]))
        
        managers = {'AP': lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2}
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers
        )
        
        model = MagicMock()
        
        logits, ids = manager.logits(model, return_ids=True)
        
        assert len(logits) == 2
        assert len(ids) == 2


class TestPromptManagerLoss:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_target_loss(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])
        
        # Create prompts with predictable target_loss returns
        prompt1 = MagicMock()
        prompt1.target_loss.return_value = torch.tensor([[0.2, 0.3], [0.4, 0.5]])  # shape: (2, 2)
        prompt2 = MagicMock()
        prompt2.target_loss.return_value = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # shape: (2, 2)
        
        managers = {'AP': lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2}
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers
        )
        
        logits = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.7, 0.3]])]
        ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        
        result = manager.target_loss(logits, ids)
        
        # Should call target_loss on each prompt and concatenate/mean
        assert result.shape[0] == 2  # batch dimension

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_control_loss(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])
        
        prompt1 = MagicMock()
        prompt1.control_loss.return_value = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        prompt2 = MagicMock()
        prompt2.control_loss.return_value = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
        
        managers = {'AP': lambda *args, **kwargs: prompt1 if "goal1" in args else prompt2}
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=managers
        )
        
        logits = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.7, 0.3]])]
        ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        
        result = manager.control_loss(logits, ids)
        
        # Should call control_loss on each prompt
        for i, (prompt, logit, id_tensor) in enumerate(zip(manager._prompts, logits, ids)):
            prompt.control_loss.assert_called_with(logit, id_tensor)


class TestPromptManagerEdgeCases:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_single_prompt(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])
        
        manager = PromptManager(
            goals=["single_goal"],
            targets=["single_target"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
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

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_empty_nonascii_toks(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([])  # Empty tensor
        
        manager = PromptManager(
            goals=["goal1"],
            targets=["target1"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
        )
        
        assert manager._nonascii_toks.numel() == 0

    def test_none_managers(self, mock_tokenizer, mock_conv_template):
        with pytest.raises(TypeError):  # Should fail when accessing None['AP']
            PromptManager(
                goals=["goal1"],
                targets=["target1"],
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=None
            )

    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_missing_ap_manager(self, mock_nonascii, mock_tokenizer, mock_conv_template):
        mock_nonascii.return_value = torch.tensor([])
        
        invalid_managers = {'OTHER': MagicMock()}  # Missing 'AP'
        
        with pytest.raises(KeyError):
            PromptManager(
                goals=["goal1"],
                targets=["target1"],
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                managers=invalid_managers
            )


class TestPromptManagerIntegration:
    @patch('advsecurenet.llm.GCG.src.prompts.prompt_manager.get_nonascii_toks')
    def test_full_workflow(self, mock_nonascii, mock_tokenizer, mock_conv_template, basic_managers):
        mock_nonascii.return_value = torch.tensor([100])
        
        manager = PromptManager(
            goals=["goal1", "goal2"],
            targets=["target1", "target2"],
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            managers=basic_managers
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])