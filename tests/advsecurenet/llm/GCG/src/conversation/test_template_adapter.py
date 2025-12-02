import pytest
import re
from unittest.mock import MagicMock, patch, Mock
from advsecurenet.llm.GCG.src.conversation.template_adapter import (
    ConversationTemplateAdapter,
)


@pytest.fixture
def mock_fastchat_conversation():
    mock_conv = MagicMock()
    mock_conv.name = "test-template"
    mock_conv.system_message = "System message"
    mock_conv.roles = ["User", "Assistant"]
    mock_conv.messages = []
    mock_conv.offset = 0
    mock_conv.sep = "\n"
    mock_conv.sep2 = None
    return mock_conv


class TestConversationTemplateAdapter:
    def test_init(self):
        adapter = ConversationTemplateAdapter()
        assert adapter is not None

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    def test_get_fastchat_template_with_fastchat(
        self, mock_get_template, mock_fastchat_conversation
    ):
        mock_get_template.return_value = mock_fastchat_conversation

        result = ConversationTemplateAdapter.get_fastchat_template("test-model")

        assert result == mock_fastchat_conversation
        mock_get_template.assert_called_once_with("test-model")

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        False,
    )
    def test_get_fastchat_template_without_fastchat(self):
        result = ConversationTemplateAdapter.get_fastchat_template("test-model")

        assert result is None

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    @patch("advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template")
    def test_get_fastchat_template_exception_handling(
        self, mock_get_conv_template, mock_get_template
    ):
        # Mock both functions to fail so all fallbacks fail
        mock_get_template.side_effect = Exception("Template not found")
        mock_get_conv_template.side_effect = Exception("Template not found")

        result = ConversationTemplateAdapter.get_fastchat_template("unknown-model")

        assert result is None

    def test_template_map_contains_common_models(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        # Test common model families are mapped
        assert "llama" in template_map
        assert "gpt2" in template_map
        assert "mistral" in template_map
        assert "qwen" in template_map
        assert "opt" in template_map

    def test_template_map_values_are_strings(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        for model_family, template_name in template_map.items():
            assert isinstance(model_family, str)
            assert isinstance(template_name, str)
            assert len(template_name) > 0

    def test_get_fastchat_template_method_exists(self):
        # Method should exist as static method on class
        assert hasattr(ConversationTemplateAdapter, "get_fastchat_template")
        assert callable(getattr(ConversationTemplateAdapter, "get_fastchat_template"))

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    def test_fastchat_template_detection_flow(self, mock_get_template):
        # Test the flow from model name to template
        mock_conv = MagicMock()
        mock_conv.name = "test_template"
        mock_get_template.return_value = mock_conv

        # Test with a known model pattern (using static method correctly)
        for model_name in [
            "llama-7b-chat",
            "microsoft/DialoGPT-medium",
            "Qwen/Qwen-7B-Chat",
        ]:
            result = ConversationTemplateAdapter.get_fastchat_template(model_name)
            assert result == mock_conv

    def test_template_map_coverage(self):
        # Ensure template map covers major model families
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        expected_families = [
            "llama",
            "mistral",
            "qwen",
            "gpt2",
            "opt",
            "dialogpt",
            "phi",
            "gemma",
            "vicuna",
            "alpaca",
        ]

        for family in expected_families:
            assert family in template_map, f"Missing template mapping for {family}"

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    @patch("advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template")
    def test_get_fastchat_template_pattern_matching(
        self, mock_get_conv_template, mock_get_template
    ):
        # Auto-detection fails, pattern matching succeeds
        mock_get_template.side_effect = Exception("Auto-detection failed")

        mock_conv = MagicMock()
        mock_conv.name = "llama-2"
        mock_get_conv_template.return_value = mock_conv

        result = ConversationTemplateAdapter.get_fastchat_template("llama-chat-model")

        assert result == mock_conv
        mock_get_conv_template.assert_called_with("llama-2")

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    @patch("advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template")
    def test_get_fastchat_template_final_fallback(
        self, mock_get_conv_template, mock_get_template
    ):
        # Auto-detection and pattern matching fail, fallback succeeds
        mock_get_template.side_effect = Exception("Auto-detection failed")

        # Pattern matching fails for all families
        def pattern_side_effect(template_name):
            if template_name == "zero_shot":  # First fallback template succeeds
                mock_conv = MagicMock()
                mock_conv.name = "zero_shot"
                return mock_conv
            raise Exception("Template not found")

        mock_get_conv_template.side_effect = pattern_side_effect

        result = ConversationTemplateAdapter.get_fastchat_template(
            "completely-unknown-model"
        )

        assert result.name == "zero_shot"

    def test_get_fastchat_template_invalid_model_name_handling(self):
        # Test None and empty string handling
        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
            True,
        ):
            result_none = ConversationTemplateAdapter.get_fastchat_template(None)
            result_empty = ConversationTemplateAdapter.get_fastchat_template("")

            assert result_none is None
            assert result_empty is None

    def test_detect_model_family(self):
        # Test model family detection
        test_cases = [
            ("microsoft/DialoGPT-medium", "dialogpt"),
            ("meta-llama/Llama-2-7b-chat", "llama"),
            ("mistralai/Mistral-7B-Instruct", "mistral"),
            ("Qwen/Qwen-7B-Chat", "qwen"),
            ("facebook/opt-350m", "opt"),
            ("some-unknown-model", "generic"),
        ]

        for model_name, expected_family in test_cases:
            result = ConversationTemplateAdapter.detect_model_family(model_name)
            assert result == expected_family

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_template_adapter_class_structure(self):
        # Test that class has expected structure
        adapter = ConversationTemplateAdapter()

        assert hasattr(adapter, "FASTCHAT_TEMPLATE_MAP")
        assert hasattr(adapter, "get_fastchat_template")
        assert isinstance(adapter.FASTCHAT_TEMPLATE_MAP, dict)

    def test_template_map_format_consistency(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        for model_key, template_value in template_map.items():
            # Model keys should be lowercase
            assert model_key.islower(), f"Model key {model_key} should be lowercase"

            # Template values should not be empty
            assert (
                len(template_value.strip()) > 0
            ), f"Template value for {model_key} is empty"

            # Template values should use valid characters
            assert re.match(
                r"^[a-zA-Z0-9_\-\.]+$", template_value
            ), f"Invalid template name: {template_value}"

    def test_common_model_mappings(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        # Test specific mappings
        assert template_map["llama"] == "llama-2"
        assert template_map["mistral"] == "mistral"
        assert template_map["qwen"] == "qwen"
        assert template_map["gpt2"] == "zero_shot"
        assert template_map["opt"] == "zero_shot"
        assert template_map["dialogpt"] == "one_shot"

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    def test_get_fastchat_template_various_inputs(
        self, mock_get_template, mock_fastchat_conversation
    ):
        mock_get_template.return_value = mock_fastchat_conversation

        # Test various model name formats
        test_cases = [
            "simple-model",
            "organization/model-name",
            "model-with-numbers-123",
            "model_with_underscores",
            "UPPERCASE-MODEL",
        ]

        for model_name in test_cases:
            result = ConversationTemplateAdapter.get_fastchat_template(model_name)
            assert result == mock_fastchat_conversation
            mock_get_template.assert_called_with(model_name)

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    @patch("advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template")
    def test_fastchat_template_error_scenarios(
        self, mock_get_conv_template, mock_get_template
    ):
        # Test different types of errors - mock both functions to fail all fallbacks
        error_scenarios = [
            ValueError("Invalid model"),
            RuntimeError("Template error"),
            KeyError("Missing key"),
            Exception("Generic error"),
        ]

        for error in error_scenarios:
            mock_get_template.side_effect = error
            mock_get_conv_template.side_effect = error

            result = ConversationTemplateAdapter.get_fastchat_template("error-model")
            assert result is None

    def test_adapter_static_method_accessibility(self):
        # Test that static method can be called without instantiation
        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
            False,
        ):
            result = ConversationTemplateAdapter.get_fastchat_template("test-model")
            assert result is None

    def test_template_map_completeness(self):
        # Ensure no empty or None values in template map
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        for model_key, template_value in template_map.items():
            assert model_key is not None
            assert template_value is not None
            assert len(model_key) > 0
            assert len(template_value) > 0

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    def test_fastchat_integration_mock_verification(
        self, mock_get_template, mock_fastchat_conversation
    ):
        # Verify mock conversation object structure
        mock_get_template.return_value = mock_fastchat_conversation

        result = ConversationTemplateAdapter.get_fastchat_template("test-model")

        # Verify returned object has expected attributes
        assert hasattr(result, "name")
        assert hasattr(result, "roles")
        assert hasattr(result, "messages")
        assert result.name == "test-template"

    def test_class_documentation_and_structure(self):
        # Test class structure and documentation
        adapter = ConversationTemplateAdapter()

        assert adapter.__class__.__doc__ is not None
        assert "Universal adapter" in adapter.__class__.__doc__
        assert "HuggingFace" in adapter.__class__.__doc__
        assert "FastChat" in adapter.__class__.__doc__

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_format_for_gcg_fastchat_format(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.bos_token = "<s>"

        mock_conv = MagicMock()
        mock_conv.name = "test-template"
        mock_conv.roles = ["User", "Assistant"]
        mock_conv.messages = []
        mock_conv.get_prompt.return_value = (
            "User: test prompt\nAssistant: test target</s>"
        )

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_universal_conversation_format"
        ) as mock_get_format:
            mock_get_format.return_value = {
                "format_type": "fastchat",
                "fastchat_template": mock_conv,
                "template_name": "test-template",
            }

            result = ConversationTemplateAdapter.format_for_gcg(
                "test prompt", "test target", "test-model", mock_tokenizer
            )

            assert "User: test prompt" in result
            assert "Assistant: test target" in result
            assert "</s>" not in result  # Should be removed for GCG

    def test_format_for_gcg_chat_template_format(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = (
            "User: test prompt\nAssistant: test target</s>"
        )

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_universal_conversation_format"
        ) as mock_get_format:
            mock_get_format.return_value = {
                "format_type": "chat_template",
                "eos_token": "</s>",
                "template_name": "huggingface_builtin",
            }

            result = ConversationTemplateAdapter.format_for_gcg(
                "test prompt", "test target", "test-model", mock_tokenizer
            )

            assert "User: test prompt" in result
            assert "Assistant: test target" in result
            assert "</s>" not in result  # Should be removed for GCG

    def test_format_for_gcg_manual_format(self):
        mock_tokenizer = MagicMock()

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_universal_conversation_format"
        ) as mock_get_format:
            mock_get_format.return_value = {
                "format_type": "manual",
                "template": "Human: {user_message}\nAssistant: {bot_response}",
                "roles": ["Human", "Assistant"],
                "template_name": "manual_generic",
            }

            result = ConversationTemplateAdapter.format_for_gcg(
                "test prompt", "test target", "test-model", mock_tokenizer
            )

            assert result == "Human: test prompt\nAssistant: test target"

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_normalize_template(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "test-model"

        mock_conv_template = MagicMock()

        mock_fastchat_conv = MagicMock()
        mock_fastchat_conv.name = "test-template"
        mock_fastchat_conv.roles = ["User", "Assistant"]
        mock_fastchat_conv.sep = "\n"
        mock_fastchat_conv.sep2 = None
        mock_fastchat_conv.system = "System message"

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_fastchat_template"
        ) as mock_get_template:
            mock_get_template.return_value = mock_fastchat_conv

            result = ConversationTemplateAdapter.normalize_template(
                mock_conv_template, mock_tokenizer
            )

            assert result.roles == ("User", "Assistant")
            assert result.sep == "\n"
            assert result.eos_token == ""  # Should be disabled for GCG


class TestConversationTemplateAdapterIntegration:
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
    )
    def test_full_template_workflow(
        self, mock_get_template, mock_fastchat_conversation
    ):
        mock_get_template.return_value = mock_fastchat_conversation

        adapter = ConversationTemplateAdapter()

        # Test complete workflow
        model_name = "test/model"
        template = adapter.get_fastchat_template(model_name)

        assert template is not None
        assert template.name == "test-template"
        mock_get_template.assert_called_once_with(model_name)

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        False,
    )
    def test_fallback_behavior_without_fastchat(self):
        adapter = ConversationTemplateAdapter()

        # Should handle gracefully without FastChat
        result = adapter.get_fastchat_template("any-model")
        assert result is None

    def test_template_map_usage_patterns(self):
        # Test that template map can be used for model family detection
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP

        # Simulate model family detection
        test_models = [
            ("microsoft/DialoGPT-small", "dialogpt"),
            ("meta-llama/Llama-2-7b-chat", "llama"),
            ("mistralai/Mistral-7B-Instruct", "mistral"),
            ("Qwen/Qwen-7B-Chat", "qwen"),
            ("facebook/opt-350m", "opt"),
        ]

        for model_name, expected_family in test_models:
            # Test that we can map model names to families
            model_lower = model_name.lower()
            found_family = None

            for family in template_map.keys():
                if family in model_lower:
                    found_family = family
                    break

            assert (
                found_family == expected_family
            ), f"Failed to map {model_name} to {expected_family}"


class TestConversationTemplateAdapterMissingMethods:
    def test_detect_model_family(self):
        # Test model family detection
        test_cases = [
            ("microsoft/DialoGPT-medium", "dialogpt"),
            ("meta-llama/Llama-2-7b-chat", "llama"),
            ("mistralai/Mistral-7B-Instruct", "mistral"),
            ("Qwen/Qwen-7B-Chat", "qwen"),
            ("facebook/opt-350m", "opt"),
            ("some-unknown-model", "generic"),
        ]

        for model_name, expected_family in test_cases:
            result = ConversationTemplateAdapter.detect_model_family(model_name)
            assert result == expected_family

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_get_universal_conversation_format_with_fastchat(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "</s>"

        mock_conv = MagicMock()
        mock_conv.name = "test-template"
        mock_conv.roles = ["User", "Assistant"]
        mock_conv.sep = "\n"
        mock_conv.sep2 = None
        mock_conv.system = "System message"

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_fastchat_template"
        ) as mock_get_template:
            mock_get_template.return_value = mock_conv

            result = ConversationTemplateAdapter.get_universal_conversation_format(
                "test-model", mock_tokenizer
            )

            assert result["format_type"] == "fastchat"
            assert result["fastchat_template"] == mock_conv
            assert result["roles"] == ["User", "Assistant"]
            assert result["template_name"] == "test-template"

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_get_universal_conversation_format_chat_template_fallback(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.chat_template = "some template"

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_fastchat_template"
        ) as mock_get_template:
            mock_get_template.return_value = None  # FastChat fails

            result = ConversationTemplateAdapter.get_universal_conversation_format(
                "test-model", mock_tokenizer
            )

            assert result["format_type"] == "chat_template"
            assert result["use_builtin"] == True
            assert result["template_name"] == "huggingface_builtin"

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        False,
    )
    def test_get_universal_conversation_format_manual_fallback(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.chat_template = None

        result = ConversationTemplateAdapter.get_universal_conversation_format(
            "llama-model", mock_tokenizer
        )

        assert result["format_type"] == "manual"
        assert result["template_name"] == "manual_instruct"
        assert result["roles"] == ["[INST]", "[/INST]"]

    def test_manual_fallback_dialogpt(self):
        mock_tokenizer = MagicMock()

        result = ConversationTemplateAdapter._get_manual_fallback(
            "dialogpt-model", mock_tokenizer
        )

        assert result["format_type"] == "manual"
        assert result["template_name"] == "manual_dialogpt"
        assert result["roles"] == ["User", "Bot"]

    def test_manual_fallback_generic(self):
        mock_tokenizer = MagicMock()

        result = ConversationTemplateAdapter._get_manual_fallback(
            "unknown-model", mock_tokenizer
        )

        assert result["format_type"] == "manual"
        assert result["template_name"] == "manual_generic"
        assert result["roles"] == ["Human", "Assistant"]

    def test_get_special_tokens_info(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.unk_token_id = 3
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.unk_token = "<unk>"

        result = ConversationTemplateAdapter.get_special_tokens_info(mock_tokenizer)

        assert result["bos_token_id"] == 1
        assert result["eos_token_id"] == 2
        assert result["bos_token"] == "<s>"
        assert result["eos_token"] == "</s>"

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch("advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template")
    def test_list_supported_templates_success(self, mock_get_conv_template):
        # Mock successful template retrieval for some templates
        mock_get_conv_template.side_effect = lambda x: (
            MagicMock() if x in ["vicuna_v1.1", "llama-2"] else Exception("Not found")
        )

        result = ConversationTemplateAdapter.list_supported_templates()

        assert isinstance(result, list)
        assert "vicuna_v1.1" in result
        assert "llama-2" in result

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        False,
    )
    def test_list_supported_templates_no_fastchat(self):
        result = ConversationTemplateAdapter.list_supported_templates()

        assert result == ["FastChat not installed"]


class TestConversationTemplateAdapterErrorHandling:
    """Test error handling scenarios to improve coverage."""

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_format_for_gcg_fastchat_exception(self):
        mock_tokenizer = MagicMock()

        mock_conv = MagicMock()
        mock_conv.get_prompt.side_effect = Exception("FastChat error")

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_universal_conversation_format"
        ) as mock_get_format:
            mock_get_format.return_value = {
                "format_type": "fastchat",
                "fastchat_template": mock_conv,
                "template_name": "test-template",
            }

            # Should catch exception and fall through to manual formatting
            result = ConversationTemplateAdapter.format_for_gcg(
                "test prompt", "test target", "test-model", mock_tokenizer
            )
            assert isinstance(result, str)

    def test_format_for_gcg_chat_template_exception(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception(
            "Chat template error"
        )

        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_universal_conversation_format"
        ) as mock_get_format:
            mock_get_format.return_value = {
                "format_type": "chat_template",
                "template_name": "huggingface_builtin",
            }

            # Should catch exception and fall through to manual formatting
            result = ConversationTemplateAdapter.format_for_gcg(
                "test prompt", "test target", "test-model", mock_tokenizer
            )
            assert isinstance(result, str)

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    def test_normalize_template_fallback(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "test-model"
        mock_conv_template = MagicMock()

        # FastChat normalization fails
        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_fastchat_template"
        ) as mock_get_template:
            mock_get_template.side_effect = Exception("FastChat normalization failed")

            with patch(
                "advsecurenet.llm.GCG.src.conversation.template_adapter.ConversationTemplateAdapter.get_universal_conversation_format"
            ) as mock_get_format:
                mock_get_format.return_value = {
                    "roles": ["Human", "Assistant"],
                    "sep": "\n",
                    "sep2": "\n\n",
                }

                result = ConversationTemplateAdapter.normalize_template(
                    mock_conv_template, mock_tokenizer
                )

                assert result.roles == ("Human", "Assistant")
                assert result.eos_token == ""

    @patch(
        "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
        True,
    )
    @patch("advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template")
    def test_list_supported_templates_exception(self, mock_get_conv_template):
        # All template checks raise exceptions - return empty list as that's the actual behavior
        mock_get_conv_template.side_effect = Exception("All templates fail")

        result = ConversationTemplateAdapter.list_supported_templates()

        # When all templates fail, it returns an empty list, not the error message
        assert result == []

    def test_import_handling_simulation(self):
        # Test that import statements are properly handled (lines 9-11)
        # We can't directly test the import since it happens at module load,
        # but we can test that FASTCHAT_AVAILABLE is set correctly
        from advsecurenet.llm.GCG.src.conversation.template_adapter import (
            FASTCHAT_AVAILABLE,
        )

        # FASTCHAT_AVAILABLE should be a boolean
        assert isinstance(FASTCHAT_AVAILABLE, bool)


class TestConversationTemplateAdapterEdgeCases:
    def test_empty_model_name(self):
        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
            True,
        ):
            with patch(
                "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
            ) as mock_get_template:
                with patch(
                    "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template"
                ) as mock_get_conv_template:
                    # Mock both functions to fail so all fallbacks fail
                    mock_get_template.side_effect = Exception("Empty model name")
                    mock_get_conv_template.side_effect = Exception("Empty model name")

                    result = ConversationTemplateAdapter.get_fastchat_template("")
                    assert result is None

    def test_none_model_name(self):
        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
            True,
        ):
            with patch(
                "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
            ) as mock_get_template:
                with patch(
                    "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conv_template"
                ) as mock_get_conv_template:
                    # Mock both functions to fail so all fallbacks fail
                    mock_get_template.side_effect = Exception("None model name")
                    mock_get_conv_template.side_effect = Exception("None model name")

                    result = ConversationTemplateAdapter.get_fastchat_template(None)
                    assert result is None

    def test_special_characters_in_model_name(self):
        with patch(
            "advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE",
            True,
        ):
            with patch(
                "advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template"
            ) as mock_get_template:
                mock_conv = MagicMock()
                mock_get_template.return_value = mock_conv

                special_names = [
                    "model@special",
                    "model#hash",
                    "model with spaces",
                    "model/with/many/slashes",
                    "model-with-many-dashes",
                    "model_with_many_underscores",
                ]

                for name in special_names:
                    result = ConversationTemplateAdapter.get_fastchat_template(name)
                    assert result == mock_conv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
