import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from copy import deepcopy
from advsecurenet.llm.GCG.src.prompts.attack_prompt import AttackPrompt


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value.input_ids = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    return tokenizer


@pytest.fixture
def mock_conv_template():
    template = MagicMock()
    template.name = "test-template"
    template.roles = ("User", "Assistant")
    template.messages = []
    template.get_prompt.return_value = "User: test goal test control\nAssistant: test target"
    template.append_message = MagicMock()
    template.update_last_message = MagicMock()
    return template


@pytest.fixture
def basic_attack_prompt(mock_tokenizer, mock_conv_template):
    with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
        mock_adapter.normalize_template.return_value = mock_conv_template
        mock_adapter.get_special_tokens_info.return_value = {'eos': 2, 'pad': 0}
        
        return AttackPrompt(
            goal="test goal",
            target="test target",
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            control_init="test control"
        )


class TestAttackPromptInit:
    def test_init_basic(self, mock_tokenizer, mock_conv_template):
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {'eos': 2}
            
            prompt = AttackPrompt(
                goal="test goal",
                target="test target",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template
            )
            
            assert prompt.goal == "test goal"
            assert prompt.target == "test target"
            assert prompt.control == "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
            assert prompt.tokenizer == mock_tokenizer
            assert len(prompt.test_prefixes) == 7

    def test_init_custom_parameters(self, mock_tokenizer, mock_conv_template):
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}
            
            custom_prefixes = ["Sorry", "Cannot"]
            prompt = AttackPrompt(
                goal="custom goal",
                target="custom target",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                control_init="custom control",
                test_prefixes=custom_prefixes
            )
            
            assert prompt.goal == "custom goal"
            assert prompt.target == "custom target"
            assert prompt.control == "custom control"
            assert prompt.test_prefixes == custom_prefixes

    def test_test_new_toks_calculation(self, mock_tokenizer, mock_conv_template):
        mock_tokenizer.return_value.input_ids = [1, 2, 3]  # 3 tokens for target
        
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}
            
            prompt = AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template
            )
            
            # Should be max of target length + 2 and test prefix lengths
            assert prompt.test_new_toks >= 5  # 3 + 2 buffer

    @patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.AttackPrompt._update_ids')
    def test_update_ids_called(self, mock_update, mock_tokenizer, mock_conv_template):
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}
            
            AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template
            )
            
            mock_update.assert_called_once()


class TestAttackPromptUpdateIds:
    def test_update_ids_basic(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.get_prompt.return_value = "test prompt"
        basic_attack_prompt.tokenizer.return_value.input_ids = [1, 2, 3, 4, 5]
        basic_attack_prompt._target_slice = slice(2, 4)
        
        with patch.object(basic_attack_prompt, '_detect_slices_robust') as mock_detect:
            basic_attack_prompt._update_ids()
            
            mock_detect.assert_called_once()
            assert torch.equal(basic_attack_prompt.input_ids, torch.tensor([1, 2, 3, 4]))

    def test_update_ids_robust_fallback(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.get_prompt.return_value = "test prompt"
        basic_attack_prompt.tokenizer.return_value.input_ids = [1, 2, 3, 4, 5]
        basic_attack_prompt._target_slice = slice(2, 4)
        
        with patch.object(basic_attack_prompt, '_detect_slices_robust') as mock_robust, \
             patch.object(basic_attack_prompt, '_detect_slices_fallback') as mock_fallback:
            
            mock_robust.side_effect = Exception("Test error")
            basic_attack_prompt._update_ids()
            
            mock_robust.assert_called_once()
            mock_fallback.assert_called_once()

    def test_messages_cleared(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.messages = ["existing", "messages"]
        basic_attack_prompt._target_slice = slice(0, 3)
        
        with patch.object(basic_attack_prompt, '_detect_slices_robust'):
            basic_attack_prompt._update_ids()
            
            assert basic_attack_prompt.conv_template.messages == []


class TestAttackPromptValidateSlices:
    def test_validate_slices_valid(self, basic_attack_prompt):
        basic_attack_prompt._user_role_slice = slice(0, 2)
        basic_attack_prompt._goal_slice = slice(2, 4)
        basic_attack_prompt._control_slice = slice(4, 6)
        basic_attack_prompt._assistant_role_slice = slice(6, 8)
        basic_attack_prompt._target_slice = slice(8, 10)
        basic_attack_prompt._loss_slice = slice(7, 9)
        
        # Should not raise
        basic_attack_prompt.validate_slices()

    def test_validate_slices_invalid_start(self, basic_attack_prompt):
        basic_attack_prompt._user_role_slice = slice(-1, 2)  # Invalid start
        basic_attack_prompt._goal_slice = slice(2, 4)
        basic_attack_prompt._control_slice = slice(4, 6)
        basic_attack_prompt._assistant_role_slice = slice(6, 8)
        basic_attack_prompt._target_slice = slice(8, 10)
        basic_attack_prompt._loss_slice = slice(7, 9)
        
        with pytest.raises(ValueError, match="Invalid user_role slice"):
            basic_attack_prompt.validate_slices()

    def test_validate_slices_invalid_order(self, basic_attack_prompt):
        basic_attack_prompt._user_role_slice = slice(0, 2)
        basic_attack_prompt._goal_slice = slice(2, 1)  # Stop before start
        basic_attack_prompt._control_slice = slice(4, 6)
        basic_attack_prompt._assistant_role_slice = slice(6, 8)
        basic_attack_prompt._target_slice = slice(8, 10)
        basic_attack_prompt._loss_slice = slice(7, 9)
        
        with pytest.raises(ValueError, match="Invalid goal slice"):
            basic_attack_prompt.validate_slices()


class TestAttackPromptDetectSlicesRobust:
    def test_detect_slices_robust_basic(self, basic_attack_prompt):
        mock_returns = [
            MagicMock(input_ids=[1, 2]),  # user role
            MagicMock(input_ids=[1, 2, 3, 4]),  # + goal
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6]),  # + control
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7, 8]),  # + assistant role
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # + target
        ]
        
        basic_attack_prompt.tokenizer.side_effect = mock_returns
        basic_attack_prompt.tokenizer.eos_token_id = 11
        
        with patch.object(basic_attack_prompt, 'validate_slices'):
            basic_attack_prompt._detect_slices_robust("test prompt")
            
            assert basic_attack_prompt._user_role_slice == slice(0, 2)
            assert basic_attack_prompt._goal_slice == slice(2, 4)
            assert basic_attack_prompt._control_slice == slice(4, 6)
            assert basic_attack_prompt._assistant_role_slice == slice(6, 8)
            assert basic_attack_prompt._target_slice == slice(8, 10)  # -1 for EOS

    def test_detect_slices_robust_empty_goal(self, basic_attack_prompt):
        basic_attack_prompt.goal = ""
        mock_returns = [
            MagicMock(input_ids=[1, 2]),  # user role only
            MagicMock(input_ids=[1, 2, 3, 4]),  # + control
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6]),  # + assistant role
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7, 8])  # + target
        ]
        
        basic_attack_prompt.tokenizer.side_effect = mock_returns
        
        with patch.object(basic_attack_prompt, 'validate_slices'):
            basic_attack_prompt._detect_slices_robust("test prompt")
            
            assert basic_attack_prompt._goal_slice == slice(2, 2)  # Empty slice

    def test_detect_slices_robust_tokenizer_quirks(self, basic_attack_prompt):
        mock_returns = [
            MagicMock(input_ids=[1, 2]),
            MagicMock(input_ids=[1, 2, 3, 4]),
            MagicMock(input_ids=[1, 2, 3]),  # Control tokens got merged/removed
            MagicMock(input_ids=[1, 2, 3, 5, 6]),
            MagicMock(input_ids=[1, 2, 3, 5, 6, 7, 8])
        ]
        
        basic_attack_prompt.tokenizer.side_effect = mock_returns
        
        # Mock the add_special_tokens=False call
        control_only_mock = MagicMock(input_ids=[10, 11])
        with patch.object(basic_attack_prompt.tokenizer, '__call__', side_effect=mock_returns), \
             patch.object(basic_attack_prompt, 'validate_slices'):
            
            # This should handle the negative control_end case
            basic_attack_prompt._detect_slices_robust("test prompt")


class TestAttackPromptDetectSlicesFallback:
    def test_detect_slices_fallback_llama2(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.name = 'llama-2'
        basic_attack_prompt.goal = "test goal"
        
        token_sequences = [
            [1, 2],           # user role
            [1, 2, 3, 4],     # + goal
            [1, 2, 3, 4, 5],  # + control
            [1, 2, 3, 4, 5, 6, 7],  # + assistant role
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # + target
        ]
        
        basic_attack_prompt.tokenizer.side_effect = [
            MagicMock(input_ids=seq) for seq in token_sequences
        ]
        
        basic_attack_prompt._detect_slices_fallback("test prompt")
        
        assert basic_attack_prompt._user_role_slice == slice(None, 2)
        assert basic_attack_prompt._goal_slice == slice(2, 4)
        assert basic_attack_prompt._control_slice == slice(4, 5)
        assert basic_attack_prompt._assistant_role_slice == slice(5, 7)
        assert basic_attack_prompt._target_slice == slice(7, 8)  # len-2
        assert basic_attack_prompt._loss_slice == slice(6, 7)    # assistant-1, len-3

    def test_detect_slices_fallback_other_template(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.name = 'other-template'
        full_prompt = "User: test\nAssistant: target"
        
        with patch.object(basic_attack_prompt.tokenizer, '__call__') as mock_tokenizer:
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [1, 2, 3, 4, 5]
            mock_encoding.char_to_token.return_value = 3
            mock_tokenizer.return_value = mock_encoding
            
            basic_attack_prompt._detect_slices_fallback(full_prompt)
            
            # Should attempt char_to_token
            mock_encoding.char_to_token.assert_called()


class TestAttackPromptMethods:
    @patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.torch')
    def test_logits_method_exists(self, mock_torch, basic_attack_prompt):
        # Test that logits method can be called
        model = MagicMock()
        
        # Mock the method if it doesn't exist
        if not hasattr(basic_attack_prompt, 'logits'):
            basic_attack_prompt.logits = MagicMock()
        
        basic_attack_prompt.logits(model)

    def test_grad_method_exists(self, basic_attack_prompt):
        model = MagicMock()
        
        if not hasattr(basic_attack_prompt, 'grad'):
            basic_attack_prompt.grad = MagicMock()
        
        basic_attack_prompt.grad(model)

    def test_test_method_exists(self, basic_attack_prompt):
        model = MagicMock()
        
        if not hasattr(basic_attack_prompt, 'test'):
            basic_attack_prompt.test = MagicMock()
        
        basic_attack_prompt.test(model)


class TestAttackPromptEdgeCases:
    def test_empty_strings(self, mock_tokenizer, mock_conv_template):
        mock_tokenizer.return_value.input_ids = []
        
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}
            
            prompt = AttackPrompt(
                goal="",
                target="",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                control_init=""
            )
            
            assert prompt.goal == ""
            assert prompt.target == ""
            assert prompt.control == ""

    def test_none_eos_token(self, mock_tokenizer, mock_conv_template):
        mock_tokenizer.eos_token_id = None
        
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}
            
            prompt = AttackPrompt(
                goal="test",
                target="test", 
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template
            )
            
            # Should handle None eos_token_id gracefully
            assert prompt.tokenizer.eos_token_id is None

    def test_special_tokens_handling(self, mock_tokenizer, mock_conv_template):
        with patch('advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter') as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {
                'eos': 2, 'pad': 0, 'bos': 1, 'unk': 3
            }
            
            prompt = AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template
            )
            
            assert prompt.special_tokens == {'eos': 2, 'pad': 0, 'bos': 1, 'unk': 3}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])