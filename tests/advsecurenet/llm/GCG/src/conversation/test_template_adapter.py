import pytest
import re
from unittest.mock import MagicMock, patch, Mock
from advsecurenet.llm.GCG.src.conversation.template_adapter import ConversationTemplateAdapter


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

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_get_fastchat_template_with_fastchat(self, mock_get_template, mock_fastchat_conversation):
        mock_get_template.return_value = mock_fastchat_conversation
        
        result = ConversationTemplateAdapter.get_fastchat_template("test-model")
        
        assert result == mock_fastchat_conversation
        mock_get_template.assert_called_once_with("test-model")

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', False)
    def test_get_fastchat_template_without_fastchat(self):
        result = ConversationTemplateAdapter.get_fastchat_template("test-model")
        
        assert result is None

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_get_fastchat_template_exception_handling(self, mock_get_template):
        mock_get_template.side_effect = Exception("Template not found")
        
        result = ConversationTemplateAdapter.get_fastchat_template("unknown-model")
        
        assert result is None

    def test_template_map_contains_common_models(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP
        
        # Test common model families are mapped
        assert 'llama' in template_map
        assert 'gpt2' in template_map
        assert 'mistral' in template_map
        assert 'qwen' in template_map
        assert 'opt' in template_map

    def test_template_map_values_are_strings(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP
        
        for model_family, template_name in template_map.items():
            assert isinstance(model_family, str)
            assert isinstance(template_name, str)
            assert len(template_name) > 0

    def test_get_conversation_template_method_exists(self):
        adapter = ConversationTemplateAdapter()
        
        # Method should exist even if not implemented
        assert hasattr(adapter, 'get_conversation_template')

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_fastchat_template_detection_flow(self, mock_get_template, mock_fastchat_conversation):
        # Test the flow from model name to template
        mock_get_template.return_value = mock_fastchat_conversation
        
        adapter = ConversationTemplateAdapter()
        
        # Test with a known model pattern
        for model_name in ["llama-7b-chat", "microsoft/DialoGPT-medium", "Qwen/Qwen-7B-Chat"]:
            result = adapter.get_fastchat_template(model_name)
            assert result == mock_fastchat_conversation

    def test_template_map_coverage(self):
        # Ensure template map covers major model families
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP
        
        expected_families = [
            'llama', 'mistral', 'qwen', 'gpt2', 'opt', 
            'dialogpt', 'phi', 'gemma', 'vicuna', 'alpaca'
        ]
        
        for family in expected_families:
            assert family in template_map, f"Missing template mapping for {family}"

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    def test_template_adapter_class_structure(self):
        # Test that class has expected structure
        adapter = ConversationTemplateAdapter()
        
        assert hasattr(adapter, 'FASTCHAT_TEMPLATE_MAP')
        assert hasattr(adapter, 'get_fastchat_template')
        assert isinstance(adapter.FASTCHAT_TEMPLATE_MAP, dict)

    def test_template_map_format_consistency(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP
        
        for model_key, template_value in template_map.items():
            # Model keys should be lowercase
            assert model_key.islower(), f"Model key {model_key} should be lowercase"
            
            # Template values should not be empty
            assert len(template_value.strip()) > 0, f"Template value for {model_key} is empty"
            
            # Template values should use valid characters
            assert re.match(r'^[a-zA-Z0-9_\-\.]+$', template_value), f"Invalid template name: {template_value}"

    def test_common_model_mappings(self):
        template_map = ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP
        
        # Test specific mappings
        assert template_map['llama'] == 'llama-2'
        assert template_map['mistral'] == 'mistral'
        assert template_map['qwen'] == 'qwen'
        assert template_map['gpt2'] == 'zero_shot'
        assert template_map['opt'] == 'zero_shot'
        assert template_map['dialogpt'] == 'one_shot'

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_get_fastchat_template_various_inputs(self, mock_get_template, mock_fastchat_conversation):
        mock_get_template.return_value = mock_fastchat_conversation
        
        # Test various model name formats
        test_cases = [
            "simple-model",
            "organization/model-name",
            "model-with-numbers-123",
            "model_with_underscores",
            "UPPERCASE-MODEL"
        ]
        
        for model_name in test_cases:
            result = ConversationTemplateAdapter.get_fastchat_template(model_name)
            assert result == mock_fastchat_conversation
            mock_get_template.assert_called_with(model_name)

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_fastchat_template_error_scenarios(self, mock_get_template):
        # Test different types of errors
        error_scenarios = [
            ValueError("Invalid model"),
            RuntimeError("Template error"),
            KeyError("Missing key"),
            Exception("Generic error")
        ]
        
        for error in error_scenarios:
            mock_get_template.side_effect = error
            
            result = ConversationTemplateAdapter.get_fastchat_template("error-model")
            assert result is None

    def test_adapter_static_method_accessibility(self):
        # Test that static method can be called without instantiation
        with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', False):
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

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_fastchat_integration_mock_verification(self, mock_get_template, mock_fastchat_conversation):
        # Verify mock conversation object structure
        mock_get_template.return_value = mock_fastchat_conversation
        
        result = ConversationTemplateAdapter.get_fastchat_template("test-model")
        
        # Verify returned object has expected attributes
        assert hasattr(result, 'name')
        assert hasattr(result, 'roles')
        assert hasattr(result, 'messages')
        assert result.name == "test-template"

    def test_class_documentation_and_structure(self):
        # Test class structure and documentation
        adapter = ConversationTemplateAdapter()
        
        assert adapter.__class__.__doc__ is not None
        assert "Universal adapter" in adapter.__class__.__doc__
        assert "HuggingFace" in adapter.__class__.__doc__
        assert "FastChat" in adapter.__class__.__doc__


class TestConversationTemplateAdapterIntegration:
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True)
    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template')
    def test_full_template_workflow(self, mock_get_template, mock_fastchat_conversation):
        mock_get_template.return_value = mock_fastchat_conversation
        
        adapter = ConversationTemplateAdapter()
        
        # Test complete workflow
        model_name = "test/model"
        template = adapter.get_fastchat_template(model_name)
        
        assert template is not None
        assert template.name == "test-template"
        mock_get_template.assert_called_once_with(model_name)

    @patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', False)
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
            ("facebook/opt-350m", "opt")
        ]
        
        for model_name, expected_family in test_models:
            # Test that we can map model names to families
            model_lower = model_name.lower()
            found_family = None
            
            for family in template_map.keys():
                if family in model_lower:
                    found_family = family
                    break
            
            assert found_family == expected_family, f"Failed to map {model_name} to {expected_family}"


class TestConversationTemplateAdapterEdgeCases:
    def test_empty_model_name(self):
        with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True):
            with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template') as mock_get_template:
                mock_get_template.side_effect = Exception("Empty model name")
                
                result = ConversationTemplateAdapter.get_fastchat_template("")
                assert result is None

    def test_none_model_name(self):
        with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True):
            with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template') as mock_get_template:
                mock_get_template.side_effect = Exception("None model name")
                
                result = ConversationTemplateAdapter.get_fastchat_template(None)
                assert result is None

    def test_special_characters_in_model_name(self):
        with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.FASTCHAT_AVAILABLE', True):
            with patch('advsecurenet.llm.GCG.src.conversation.template_adapter.get_conversation_template') as mock_get_template:
                mock_conv = MagicMock()
                mock_get_template.return_value = mock_conv
                
                special_names = [
                    "model@special",
                    "model#hash", 
                    "model with spaces",
                    "model/with/many/slashes",
                    "model-with-many-dashes",
                    "model_with_many_underscores"
                ]
                
                for name in special_names:
                    result = ConversationTemplateAdapter.get_fastchat_template(name)
                    assert result == mock_conv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])