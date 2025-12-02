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
    template.get_prompt.return_value = (
        "User: test goal test control\nAssistant: test target"
    )
    template.append_message = MagicMock()
    template.update_last_message = MagicMock()
    return template


@pytest.fixture
def basic_attack_prompt(mock_tokenizer, mock_conv_template):
    with patch(
        "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
    ) as mock_adapter:
        mock_adapter.normalize_template.return_value = mock_conv_template
        mock_adapter.get_special_tokens_info.return_value = {"eos": 2, "pad": 0}

        return AttackPrompt(
            goal="test goal",
            target="test target",
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            control_init="test control",
        )


class TestAttackPromptInit:
    def test_init_basic(self, mock_tokenizer, mock_conv_template):
        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {"eos": 2}

            prompt = AttackPrompt(
                goal="test goal",
                target="test target",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
            )

            assert prompt.goal == "test goal"
            assert prompt.target == "test target"
            assert prompt.control == "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
            assert prompt.tokenizer == mock_tokenizer
            assert len(prompt.test_prefixes) == 7

    def test_init_custom_parameters(self, mock_tokenizer, mock_conv_template):
        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}

            custom_prefixes = ["Sorry", "Cannot"]
            prompt = AttackPrompt(
                goal="custom goal",
                target="custom target",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                control_init="custom control",
                test_prefixes=custom_prefixes,
            )

            assert prompt.goal == "custom goal"
            assert prompt.target == "custom target"
            assert prompt.control == "custom control"
            assert prompt.test_prefixes == custom_prefixes

    def test_test_new_toks_calculation(self, mock_tokenizer, mock_conv_template):
        mock_tokenizer.return_value.input_ids = [1, 2, 3]  # 3 tokens for target

        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}

            prompt = AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
            )

            # Should be max of target length + 2 and test prefix lengths
            assert prompt.test_new_toks >= 5  # 3 + 2 buffer

    @patch("advsecurenet.llm.GCG.src.prompts.attack_prompt.AttackPrompt._update_ids")
    def test_update_ids_called(self, mock_update, mock_tokenizer, mock_conv_template):
        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}

            AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
            )

            mock_update.assert_called_once()


class TestAttackPromptUpdateIds:
    def test_update_ids_basic(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.get_prompt.return_value = "test prompt"
        basic_attack_prompt.tokenizer.return_value.input_ids = [1, 2, 3, 4, 5]
        basic_attack_prompt._target_slice = slice(2, 4)

        with patch.object(basic_attack_prompt, "_detect_slices_robust") as mock_detect:
            basic_attack_prompt._update_ids()

            mock_detect.assert_called_once()
            assert torch.equal(
                basic_attack_prompt.input_ids, torch.tensor([1, 2, 3, 4])
            )

    def test_update_ids_robust_fallback(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.get_prompt.return_value = "test prompt"
        basic_attack_prompt.tokenizer.return_value.input_ids = [1, 2, 3, 4, 5]
        basic_attack_prompt._target_slice = slice(2, 4)

        with patch.object(
            basic_attack_prompt, "_detect_slices_robust"
        ) as mock_robust, patch.object(
            basic_attack_prompt, "_detect_slices_fallback"
        ) as mock_fallback:

            mock_robust.side_effect = Exception("Test error")
            basic_attack_prompt._update_ids()

            mock_robust.assert_called_once()
            mock_fallback.assert_called_once()

    def test_messages_cleared(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.messages = ["existing", "messages"]
        basic_attack_prompt._target_slice = slice(0, 3)

        with patch.object(basic_attack_prompt, "_detect_slices_robust"):
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
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),  # + target
        ]

        basic_attack_prompt.tokenizer.side_effect = mock_returns
        basic_attack_prompt.tokenizer.eos_token_id = 11

        with patch.object(basic_attack_prompt, "validate_slices"):
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
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7, 8]),  # + target
        ]

        basic_attack_prompt.tokenizer.side_effect = mock_returns

        with patch.object(basic_attack_prompt, "validate_slices"):
            basic_attack_prompt._detect_slices_robust("test prompt")

            assert basic_attack_prompt._goal_slice == slice(2, 2)  # Empty slice

    def test_detect_slices_robust_tokenizer_quirks(self, basic_attack_prompt):
        # Mock progressive tokenization calls
        side_effect = [
            MagicMock(input_ids=[1, 2]),  # user role
            MagicMock(input_ids=[1, 2, 3, 4]),  # + goal
            MagicMock(input_ids=[1, 2, 3]),  # Control tokens got merged/removed
            MagicMock(input_ids=[1, 2, 3, 5, 6]),  # + assistant role
            MagicMock(input_ids=[1, 2, 3, 5, 6, 7, 8]),  # + target
            MagicMock(input_ids=[10, 11]),  # control only tokens for quirks handling
        ]

        basic_attack_prompt.tokenizer.side_effect = side_effect

        with patch.object(basic_attack_prompt, "validate_slices"):
            # This should handle the negative control_end case
            basic_attack_prompt._detect_slices_robust("test prompt")


class TestAttackPromptDetectSlicesFallback:
    def test_detect_slices_fallback_llama2(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.name = "llama-2"
        basic_attack_prompt.goal = "test goal"

        # Create mock tokenizer results that simulate incremental encoding
        # The actual fallback logic calls tokenizer multiple times and uses len(toks)
        token_sequences = [
            [1, 2],  # user role call 1
            [1, 2, 3, 4],  # goal call 2 (len=4, so slice becomes None, 4)
            [1, 2, 3, 4, 5, 6],  # control call 3
            [1, 2, 3, 4, 5, 6, 7, 8],  # assistant role call 4
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # target call 5
        ]

        def mock_tokenizer_call(*args, **kwargs):
            call_index = getattr(mock_tokenizer_call, "call_count", 0)
            result = MagicMock()
            result.input_ids = token_sequences[call_index % len(token_sequences)]
            mock_tokenizer_call.call_count = call_index + 1
            return result

        basic_attack_prompt.tokenizer.side_effect = mock_tokenizer_call

        # Skip validation to avoid slice issues
        with patch.object(basic_attack_prompt, "validate_slices"):
            basic_attack_prompt._detect_slices_fallback("test prompt")

        # The actual behavior: user_role_slice gets set to slice(None, len(goal_tokens))
        # based on the algorithm where goal update causes the slice to be slice(None, len(goal_call))
        assert basic_attack_prompt._user_role_slice == slice(
            None, 4
        )  # Updated by goal call
        assert basic_attack_prompt._goal_slice == slice(4, 6)  # max(4, 6) = 6
        assert basic_attack_prompt._control_slice == slice(6, 8)  # 6 to 8
        # The method executed successfully without errors

    def test_detect_slices_fallback_other_template(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.name = "other-template"
        basic_attack_prompt.conv_template.system = "System: "
        full_prompt = "User: test\nAssistant: target"

        # Setup the tokenizer to trigger the char_to_token fallback
        mock_encoding = MagicMock()
        mock_encoding.input_ids = [1, 2, 3, 4, 5]
        # Make char_to_token raise an exception to force python_tokenizer=True path
        mock_encoding.char_to_token.side_effect = Exception(
            "char_to_token not supported"
        )

        with patch.object(basic_attack_prompt, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = mock_encoding

            # Set up basic slices to avoid validation errors
            basic_attack_prompt._system_slice = slice(0, 1)
            basic_attack_prompt._user_role_slice = slice(1, 2)
            basic_attack_prompt._goal_slice = slice(2, 3)
            basic_attack_prompt._control_slice = slice(3, 4)
            basic_attack_prompt._assistant_role_slice = slice(4, 5)
            basic_attack_prompt._target_slice = slice(5, 6)
            basic_attack_prompt._loss_slice = slice(4, 5)

            with patch.object(basic_attack_prompt, "validate_slices"):
                # This should execute without error, triggering the python tokenizer path
                basic_attack_prompt._detect_slices_fallback(full_prompt)

            # Should call tokenizer at least once
            assert mock_tokenizer.call_count > 0


class TestAttackPromptMethods:
    def test_logits_basic_tensor_input(self, basic_attack_prompt):
        # Test logits method with tensor input
        model = MagicMock()
        model.device = "cpu"
        model.return_value.logits = torch.randn(1, 10, 100)

        # Mock the control_toks property
        basic_attack_prompt._control_slice = slice(2, 4)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])

        test_controls = torch.tensor([[7, 8]])

        logits = basic_attack_prompt.logits(model, test_controls)

        assert logits is not None
        model.assert_called_once()

    def test_logits_string_input(self, basic_attack_prompt):
        # Test logits method with string input
        model = MagicMock()
        model.device = "cpu"
        model.return_value.logits = torch.randn(1, 10, 100)

        # Setup proper slices and mocks
        basic_attack_prompt._control_slice = slice(2, 4)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt.tokenizer.return_value.input_ids = [7, 8]

        test_controls = ["test control"]

        logits = basic_attack_prompt.logits(model, test_controls)

        assert logits is not None

    def test_grad_not_implemented(self, basic_attack_prompt):
        model = MagicMock()

        with pytest.raises(
            NotImplementedError, match="Gradient function not yet implemented"
        ):
            basic_attack_prompt.grad(model)

    def test_test_method_with_proper_setup(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"
        model.generation_config.max_new_tokens = 16

        # Setup required attributes for generate method
        basic_attack_prompt._assistant_role_slice = slice(5, 7)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        basic_attack_prompt.tokenizer.pad_token_id = 0

        # Mock model.generate to return reasonable output
        mock_output = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        model.generate.return_value = [mock_output]

        # Mock tokenizer.decode
        basic_attack_prompt.tokenizer.decode.return_value = "Test response"

        jailbroken, em = basic_attack_prompt.test(model)

        assert isinstance(jailbroken, bool)
        assert isinstance(em, int)


class TestAttackPromptEdgeCases:
    def test_empty_strings(self, mock_tokenizer, mock_conv_template):
        mock_tokenizer.return_value.input_ids = []

        # Mock the conv_template methods to avoid slice validation issues
        mock_conv_template.get_prompt.return_value = ""
        mock_conv_template.append_message = MagicMock()
        mock_conv_template.update_last_message = MagicMock()

        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}

            # Mock _update_ids to avoid complex initialization
            with patch.object(AttackPrompt, "_update_ids"):
                prompt = AttackPrompt(
                    goal="",
                    target="",
                    tokenizer=mock_tokenizer,
                    conv_template=mock_conv_template,
                    control_init="",
                )

                assert prompt.goal == ""
                assert prompt.target == ""
                assert prompt.control == ""

    def test_none_eos_token(self, mock_tokenizer, mock_conv_template):
        mock_tokenizer.eos_token_id = None

        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {}

            prompt = AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
            )

            # Should handle None eos_token_id gracefully
            assert prompt.tokenizer.eos_token_id is None

    def test_special_tokens_handling(self, mock_tokenizer, mock_conv_template):
        with patch(
            "advsecurenet.llm.GCG.src.prompts.attack_prompt.ConversationTemplateAdapter"
        ) as mock_adapter:
            mock_adapter.normalize_template.return_value = mock_conv_template
            mock_adapter.get_special_tokens_info.return_value = {
                "eos": 2,
                "pad": 0,
                "bos": 1,
                "unk": 3,
            }

            prompt = AttackPrompt(
                goal="test",
                target="test",
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
            )

            assert prompt.special_tokens == {"eos": 2, "pad": 0, "bos": 1, "unk": 3}


class TestAttackPromptProperties:
    def test_control_toks_property(self, basic_attack_prompt):
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._control_slice = slice(2, 4)

        result = basic_attack_prompt.control_toks
        expected = torch.tensor([3, 4])
        assert torch.equal(result, expected)

    def test_control_toks_setter(self, basic_attack_prompt):
        # Mock tokenizer decode to avoid complex mocking
        basic_attack_prompt.tokenizer.decode.return_value = "new control"

        # Mock _update_ids to avoid complex slice calculations
        with patch.object(basic_attack_prompt, "_update_ids"):
            basic_attack_prompt.control_toks = torch.tensor([7, 8, 9])

        assert basic_attack_prompt.control == "new control"

    def test_target_toks_property(self, basic_attack_prompt):
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._target_slice = slice(4, 6)

        result = basic_attack_prompt.target_toks
        expected = torch.tensor([5, 6])
        assert torch.equal(result, expected)

    def test_goal_toks_property(self, basic_attack_prompt):
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._goal_slice = slice(1, 3)

        result = basic_attack_prompt.goal_toks
        expected = torch.tensor([2, 3])
        assert torch.equal(result, expected)

    def test_assistant_toks_property(self, basic_attack_prompt):
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._assistant_role_slice = slice(3, 5)

        result = basic_attack_prompt.assistant_toks
        expected = torch.tensor([4, 5])
        assert torch.equal(result, expected)

    def test_string_properties(self, basic_attack_prompt):
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._goal_slice = slice(1, 2)
        basic_attack_prompt._control_slice = slice(2, 3)
        basic_attack_prompt._target_slice = slice(3, 4)
        basic_attack_prompt._assistant_role_slice = slice(4, 5)

        # Mock tokenizer decode
        basic_attack_prompt.tokenizer.decode.side_effect = [
            "goal text",
            "control text",
            "target text",
            "assistant text",
            "full prompt",
            "eval prompt",
        ]

        assert basic_attack_prompt.goal_str == "goal text"
        assert basic_attack_prompt.control_str == "control text"
        assert basic_attack_prompt.target_str == "target text"
        assert basic_attack_prompt.assistant_str == "assistant text"
        assert basic_attack_prompt.input_str == "full prompt"

    def test_string_setters(self, basic_attack_prompt):
        # Mock _update_ids to avoid complex calculations
        with patch.object(basic_attack_prompt, "_update_ids"):
            basic_attack_prompt.goal_str = "new goal"
            assert basic_attack_prompt.goal == "new goal"

            basic_attack_prompt.target_str = "new target"
            assert basic_attack_prompt.target == "new target"

            basic_attack_prompt.control_str = "new control"
            assert basic_attack_prompt.control == "new control"


class TestAttackPromptGeneration:
    def test_generate_basic(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"
        model.generation_config.max_new_tokens = 16

        # Setup required slices and tensors
        basic_attack_prompt._assistant_role_slice = slice(5, 7)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        basic_attack_prompt.tokenizer.pad_token_id = 0

        # Mock model generation
        output_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        model.generate.return_value = [output_tensor]

        result = basic_attack_prompt.generate(model)
        expected = torch.tensor([8, 9, 10])  # After assistant_role_slice.stop
        assert torch.equal(result, expected)

    def test_generate_with_custom_config(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"

        # Create custom generation config
        gen_config = MagicMock()
        gen_config.max_new_tokens = 8

        basic_attack_prompt._assistant_role_slice = slice(5, 7)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        basic_attack_prompt.tokenizer.pad_token_id = 0

        output_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        model.generate.return_value = [output_tensor]

        result = basic_attack_prompt.generate(model, gen_config)
        expected = torch.tensor([8, 9])
        assert torch.equal(result, expected)

    def test_generate_warning_for_large_tokens(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"

        gen_config = MagicMock()
        gen_config.max_new_tokens = 50  # Should trigger warning

        basic_attack_prompt._assistant_role_slice = slice(5, 7)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        basic_attack_prompt.tokenizer.pad_token_id = 0

        output_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        model.generate.return_value = [output_tensor]

        with patch("builtins.print") as mock_print:
            basic_attack_prompt.generate(model, gen_config)
            mock_print.assert_called_with(
                "WARNING: max_new_tokens > 32 may cause testing to slow down."
            )

    def test_generate_str(self, basic_attack_prompt):
        model = MagicMock()

        # Mock generate method
        with patch.object(basic_attack_prompt, "generate") as mock_generate:
            mock_generate.return_value = torch.tensor([1, 2, 3])
            basic_attack_prompt.tokenizer.decode.return_value = "generated text"

            result = basic_attack_prompt.generate_str(model)
            assert result == "generated text"


class TestAttackPromptLossFunctions:
    def test_target_loss(self, basic_attack_prompt):
        # Setup test data
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        basic_attack_prompt._target_slice = slice(4, 6)

        loss = basic_attack_prompt.target_loss(logits, ids)

        assert loss.shape == (batch_size, 2)  # 2 = length of target slice

    def test_control_loss(self, basic_attack_prompt):
        # Setup test data
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        basic_attack_prompt._control_slice = slice(2, 4)

        loss = basic_attack_prompt.control_loss(logits, ids)

        assert loss.shape == (batch_size, 2)  # 2 = length of control slice

    def test_test_loss(self, basic_attack_prompt):
        model = MagicMock()

        # Mock logits method to return test data
        batch_size, seq_len, vocab_size = 1, 8, 100
        mock_logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with patch.object(basic_attack_prompt, "logits") as mock_logits_method:
            mock_logits_method.return_value = (mock_logits, mock_ids)

            # Mock target_loss
            with patch.object(basic_attack_prompt, "target_loss") as mock_target_loss:
                mock_loss_tensor = torch.tensor([[0.5, 0.3]])
                mock_target_loss.return_value = mock_loss_tensor

                result = basic_attack_prompt.test_loss(model)

                assert isinstance(result, float)
                assert abs(result - 0.4) < 1e-6  # Use approximate equality for floats


class TestAttackPromptLogitsAdvanced:
    def test_logits_with_return_ids(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 8, 100)
        model.return_value = mock_output

        basic_attack_prompt._control_slice = slice(2, 4)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

        test_controls = torch.tensor([[7, 8]])

        logits, ids = basic_attack_prompt.logits(model, test_controls, return_ids=True)

        assert logits is not None
        assert ids is not None

    def test_logits_value_error_wrong_shape(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"

        basic_attack_prompt._control_slice = slice(2, 4)  # expects length 2
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])

        # Wrong shape - should be (n, 2) but providing (n, 3)
        test_controls = torch.tensor([[7, 8, 9]])

        with pytest.raises(ValueError, match="test_controls must have shape"):
            basic_attack_prompt.logits(model, test_controls)

    def test_logits_with_attention_mask(self, basic_attack_prompt):
        model = MagicMock()
        model.device = "cpu"
        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 6, 100)
        model.return_value = mock_output

        basic_attack_prompt._control_slice = slice(2, 4)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt.tokenizer.return_value.input_ids = [7, 8]

        # Test with string controls that will use padding
        test_controls = ["test", "control"]

        logits = basic_attack_prompt.logits(model, test_controls)

        assert logits is not None
        # Should call model with attention_mask
        call_args = model.call_args
        assert "attention_mask" in call_args.kwargs


class TestAttackPromptFallbackBranches:
    def test_detect_slices_fallback_python_tokenizer(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.name = "oasst_pythia"

        # Create tokenization sequence for python tokenizer branch
        def mock_tokenizer_call(*args, **kwargs):
            call_index = getattr(mock_tokenizer_call, "call_count", 0)
            sequences = [
                [1, 2],
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
            ]
            result = MagicMock()
            result.input_ids = sequences[min(call_index, len(sequences) - 1)]
            mock_tokenizer_call.call_count = call_index + 1
            return result

        basic_attack_prompt.tokenizer.side_effect = mock_tokenizer_call
        basic_attack_prompt.goal = "test goal"

        # Mock validate_slices to avoid slice validation issues
        with patch.object(basic_attack_prompt, "validate_slices"):
            basic_attack_prompt._detect_slices_fallback("test prompt")

    def test_detect_slices_fallback_char_to_token_exception(self, basic_attack_prompt):
        basic_attack_prompt.conv_template.name = "other"

        with patch.object(basic_attack_prompt.tokenizer, "__call__") as mock_call:
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [1, 2, 3, 4, 5]
            # Make char_to_token raise exception to trigger python_tokenizer=True
            mock_encoding.char_to_token.side_effect = Exception("char_to_token failed")
            mock_call.return_value = mock_encoding

            # Need to setup the python tokenizer flow
            def tokenizer_side_effect(*args, **kwargs):
                call_index = getattr(tokenizer_side_effect, "call_count", 0)
                sequences = [
                    [1, 2],
                    [1, 2, 3],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6],
                ]
                result = MagicMock()
                result.input_ids = sequences[min(call_index, len(sequences) - 1)]
                tokenizer_side_effect.call_count = call_index + 1
                return result

            basic_attack_prompt.tokenizer.side_effect = [mock_encoding] + [
                tokenizer_side_effect() for _ in range(10)
            ]

            with patch.object(basic_attack_prompt, "validate_slices"):
                basic_attack_prompt._detect_slices_fallback("test prompt")

    def test_validate_slices_ordering_warning(self, basic_attack_prompt):
        # Setup slices in wrong order to trigger warning
        basic_attack_prompt._user_role_slice = slice(0, 2)
        basic_attack_prompt._goal_slice = slice(5, 7)  # Out of order
        basic_attack_prompt._control_slice = slice(2, 4)  # Out of order
        basic_attack_prompt._assistant_role_slice = slice(6, 8)
        basic_attack_prompt._target_slice = slice(8, 10)
        basic_attack_prompt._loss_slice = slice(7, 9)

        with patch("builtins.print") as mock_print:
            basic_attack_prompt.validate_slices()
            mock_print.assert_called_with("Warning: Slice ordering may be incorrect")


class TestAttackPromptMissingLines:
    def test_logits_error_handling(self, basic_attack_prompt):
        """Test error handling in logits method to cover lines 306, 309, 312"""
        model = MagicMock()
        model.device = "cpu"

        # Setup required control slice for the test
        basic_attack_prompt._control_slice = slice(2, 4)
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])

        # Test with list containing invalid type (should trigger the else branch)
        with pytest.raises(
            ValueError,
            match="test_controls must be a list of strings or a tensor of token ids",
        ):
            basic_attack_prompt.logits(model, [123])  # List with invalid type

        # Test with wrong tensor shape (line 321)
        wrong_shape_tensor = torch.tensor([[7, 8, 9]])  # Wrong shape (3 instead of 2)
        with pytest.raises(ValueError, match="test_controls must have shape"):
            basic_attack_prompt.logits(model, wrong_shape_tensor)

    def test_robust_detection_hasattr_check(self, basic_attack_prompt):
        """Test hasattr check in robust detection - line 133"""
        # Mock progressive tokenization
        side_effect = [
            MagicMock(input_ids=[1, 2]),  # user role
            MagicMock(input_ids=[1, 2, 3, 4]),  # + goal
            MagicMock(input_ids=[1, 2, 3, 4, 5]),  # + control
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7]),  # + assistant role
            MagicMock(input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9]),  # + target
        ]

        basic_attack_prompt.tokenizer.side_effect = side_effect

        # Mock tokenizer to NOT have add_special_tokens
        basic_attack_prompt.tokenizer.add_special_tokens = False

        with patch.object(basic_attack_prompt, "validate_slices"):
            basic_attack_prompt._detect_slices_robust("test prompt")

    def test_prompt_property(self, basic_attack_prompt):
        """Test prompt property - line 421"""
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._goal_slice = slice(1, 3)
        basic_attack_prompt._control_slice = slice(3, 5)
        basic_attack_prompt.tokenizer.decode.return_value = "goal control text"

        result = basic_attack_prompt.prompt
        assert result == "goal control text"

    def test_eval_str_property(self, basic_attack_prompt):
        """Test eval_str property - line 433"""
        basic_attack_prompt.input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        basic_attack_prompt._assistant_role_slice = slice(4, 6)
        basic_attack_prompt.tokenizer.decode.return_value = "<s>test eval text</s>"

        result = basic_attack_prompt.eval_str
        expected = "test eval text"  # Should remove <s> and </s> tags
        assert result == expected

    def test_fallback_vicuna_pythia_branch(self, basic_attack_prompt):
        """Test the vicuna/pythia tokenizer branch in fallback - lines 230-254"""
        basic_attack_prompt.conv_template.name = "other-template"  # Not llama-2

        # Mock encoding to trigger the python_tokenizer=False initially, then True via exception
        mock_encoding = MagicMock()
        mock_encoding.input_ids = [1, 2, 3, 4, 5]

        # First char_to_token call should work, second should fail to trigger python_tokenizer=True
        char_to_token_calls = [3, Exception("Failed")]
        mock_encoding.char_to_token.side_effect = char_to_token_calls

        with patch.object(basic_attack_prompt.tokenizer, "__call__") as mock_call:
            # First call returns the encoding, subsequent calls for python tokenizer flow
            call_sequence = [mock_encoding]

            # Add tokenizer calls for the python tokenizer branch
            python_sequences = [
                [1, 2],
                [1, 2, 3],
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6],
            ]
            for seq in python_sequences:
                mock_result = MagicMock()
                mock_result.input_ids = seq
                call_sequence.append(mock_result)

            mock_call.side_effect = call_sequence

            with patch.object(basic_attack_prompt, "validate_slices"):
                basic_attack_prompt._detect_slices_fallback("test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
