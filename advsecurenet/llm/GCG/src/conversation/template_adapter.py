"""Universal adapter for conversation templates across HuggingFace models.

This module provides a unified interface for handling conversation templates
from various sources including FastChat, HuggingFace built-in chat templates,
and manual fallbacks. It's specifically optimized for GCG (Greedy Coordinate
Gradient) attacks by handling special token removal and template normalization.

The adapter supports:
- FastChat conversation templates for popular models
- HuggingFace built-in chat templates
- Manual fallback templates for unsupported models
- Special token handling for adversarial attacks
"""

import re
from typing import Dict, Optional, Any, List

try:
    from fastchat.conversation import get_conv_template, Conversation
    from fastchat.model.model_adapter import get_conversation_template

    FASTCHAT_AVAILABLE = True
    print("FastChat available for conversation templates")
except ImportError:
    FASTCHAT_AVAILABLE = False
    print("FastChat not available. Install with: pip install fschat")


class ConversationTemplateAdapter:
    """Universal adapter for conversation templates across HuggingFace models.

    This class provides a unified interface for handling conversation templates
    from various sources, with intelligent fallbacks and special handling for
    adversarial attacks like GCG.

    Features:
    - Automatic FastChat template detection
    - HuggingFace built-in chat template support
    - Manual fallback templates for unsupported models
    - Special token removal for adversarial compatibility
    - Model family detection and template mapping

    Attributes:
        FASTCHAT_TEMPLATE_MAP: Mapping of model families to FastChat templates
    """

    # FastChat template fallback mapping for common model families
    FASTCHAT_TEMPLATE_MAP = {
        "dialogpt": "one_shot",
        "gpt2": "zero_shot",
        "llama": "llama-2",
        "mistral": "mistral",
        "mixtral": "mistral",
        "phi": "phi",
        "qwen": "qwen",
        "gemma": "gemma",
        "vicuna": "vicuna_v1.1",
        "alpaca": "alpaca",
        "wizard": "vicuna_v1.1",
        "orca": "orca",
        "guanaco": "guanaco",
        "claude": "claude",
        "chatglm": "chatglm",
        "baichuan": "baichuan-chat",
        "internlm": "internlm-chat",
        "t5": "one_shot",
        "bart": "zero_shot",
        "opt": "zero_shot",
        "bloom": "zero_shot",
        "codegen": "zero_shot",
        "incoder": "zero_shot",
        "gpt-neo": "zero_shot",
        "gpt-j": "zero_shot",
    }

    @staticmethod
    def get_fastchat_template(model_name: str):
        """Get FastChat conversation template for any model.

        Attempts to retrieve a conversation template using FastChat's auto-detection,
        falls back to pattern-based matching, and finally tries common templates.

        Args:
            model_name: Name or path of the HuggingFace model

        Returns:
            FastChat Conversation object if found, None otherwise
        """
        if not FASTCHAT_AVAILABLE:
            return None

        if not model_name:
            print(f"Invalid model name: {model_name}")
            return None

        try:
            # First, try direct model name lookup (FastChat's auto-detection)
            conv = get_conversation_template(model_name)
            print(f"FastChat auto-detected template '{conv.name}' for {model_name}")
            return conv
        except Exception as e:
            print(f"FastChat auto-detection failed for {model_name}: {e}")

            # Fallback to pattern-based template selection
            model_lower = model_name.lower()
            for (
                family,
                template_name,
            ) in ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP.items():
                if family in model_lower:
                    try:
                        conv = get_conv_template(template_name)
                        print(
                            f"Using FastChat template '{template_name}' for {model_name} (family: {family})"
                        )
                        return conv
                    except Exception:
                        continue

            # Final fallback - try common templates
            fallback_templates = ["zero_shot", "one_shot", "raw", "vicuna_v1.1"]
            for template in fallback_templates:
                try:
                    conv = get_conv_template(template)
                    print(
                        f"Using FastChat fallback template '{template}' for {model_name}"
                    )
                    return conv
                except:
                    continue

            print(f"Could not find any FastChat template for {model_name}")
            return None

    @staticmethod
    def detect_model_family(model_name: str) -> str:
        """Detect model family from HuggingFace model name.

        Uses pattern matching against known model families to determine
        the appropriate conversation template category.

        Args:
            model_name: Name or path of the HuggingFace model

        Returns:
            Detected model family string, 'generic' if no match found
        """
        model_name_lower = model_name.lower()

        for family in ConversationTemplateAdapter.FASTCHAT_TEMPLATE_MAP.keys():
            if family in model_name_lower:
                return family

        return "generic"

    @staticmethod
    def get_universal_conversation_format(model_name: str, tokenizer) -> Dict[str, Any]:
        """Get universal conversation format using multiple fallback strategies.

        Attempts to obtain conversation format in order of preference:
        1. FastChat templates (most comprehensive)
        2. HuggingFace built-in chat templates
        3. Manual fallback templates

        Args:
            model_name: Name or path of the HuggingFace model
            tokenizer: HuggingFace tokenizer instance

        Returns:
            Dictionary containing conversation format configuration including
            format_type, roles, separators, and template information
        """
        # Try FastChat first
        if FASTCHAT_AVAILABLE:
            fastchat_conv = ConversationTemplateAdapter.get_fastchat_template(
                model_name
            )
            if fastchat_conv is not None:
                return {
                    "format_type": "fastchat",
                    "fastchat_template": fastchat_conv,
                    "roles": list(fastchat_conv.roles),
                    "sep": getattr(fastchat_conv, "sep", "\n"),
                    "sep2": getattr(
                        fastchat_conv, "sep2", getattr(fastchat_conv, "sep", "\n")
                    ),
                    "system": getattr(fastchat_conv, "system", ""),
                    "eos_token": getattr(tokenizer, "eos_token", ""),
                    "template_name": fastchat_conv.name,
                }

        # Fallback to HuggingFace built-in chat template
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            return {
                "format_type": "chat_template",
                "use_builtin": True,
                "roles": ["user", "assistant"],
                "sep": "",
                "eos_token": getattr(tokenizer, "eos_token", ""),
                "template_name": "huggingface_builtin",
            }

        # Final manual fallback
        return ConversationTemplateAdapter._get_manual_fallback(model_name, tokenizer)

    @staticmethod
    def _get_manual_fallback(model_name: str, tokenizer) -> Dict[str, Any]:
        """Manual fallback conversation templates when FastChat is unavailable.

        Provides basic conversation templates for common model families
        when other template sources are not accessible.

        Args:
            model_name: Name or path of the HuggingFace model
            tokenizer: HuggingFace tokenizer instance

        Returns:
            Dictionary with manual conversation format configuration
        """
        family = ConversationTemplateAdapter.detect_model_family(model_name)

        # Simplified manual formats as last resort
        if family == "dialogpt":
            return {
                "format_type": "manual",
                "roles": ["User", "Bot"],
                "sep": " ",
                "template": "{user_role}: {user_message} {bot_role}: {bot_response}",
                "eos_token": "",  # Remove problematic EOS for GCG
                "template_name": "manual_dialogpt",
            }
        elif family in ["llama", "mistral"]:
            return {
                "format_type": "manual",
                "roles": ["[INST]", "[/INST]"],
                "sep": " ",
                "template": "[INST] {user_message} [/INST] {bot_response}",
                "eos_token": "",
                "template_name": "manual_instruct",
            }
        else:
            return {
                "format_type": "manual",
                "roles": ["Human", "Assistant"],
                "sep": "\n",
                "template": "Human: {user_message}\nAssistant: {bot_response}",
                "eos_token": "",
                "template_name": "manual_generic",
            }

    @staticmethod
    def format_for_gcg(
        prompt: str,
        target: str,
        model_name: str,
        tokenizer,
        avoid_special_tokens: bool = True,
    ) -> str:
        """Format prompt for GCG attack with proper conversation structure.

        Creates a formatted conversation prompt suitable for GCG attacks,
        handling special tokens and template-specific formatting.

        Args:
            prompt: User input/attack prompt
            target: Desired model response/target
            model_name: Name or path of the HuggingFace model
            tokenizer: HuggingFace tokenizer instance
            avoid_special_tokens: Whether to remove problematic special tokens

        Returns:
            Formatted conversation string ready for GCG attack
        """
        conv_format = ConversationTemplateAdapter.get_universal_conversation_format(
            model_name, tokenizer
        )

        print(
            f"Using template '{conv_format.get('template_name', 'unknown')}' for {model_name}"
        )

        # Use FastChat template
        if conv_format["format_type"] == "fastchat":
            try:
                fastchat_conv = conv_format["fastchat_template"]

                # Reset conversation
                fastchat_conv.messages = []

                # Add user message
                fastchat_conv.append_message(fastchat_conv.roles[0], prompt)

                # Add assistant message (our target)
                fastchat_conv.append_message(fastchat_conv.roles[1], target)

                # Get formatted prompt
                formatted = fastchat_conv.get_prompt()

                # Remove problematic tokens for GCG
                if avoid_special_tokens:
                    problematic_tokens = [
                        getattr(tokenizer, "eos_token", ""),
                        getattr(tokenizer, "bos_token", ""),
                        "<|endoftext|>",
                        "<|im_start|>",
                        "<|im_end|>",
                        "<s>",
                        "</s>",
                        "<|end|>",
                        "<|begin_of_text|>",
                        "<|end_of_text|>",
                    ]
                    for token in problematic_tokens:
                        if token and token in formatted:
                            formatted = formatted.replace(token, "")
                            print(
                                f"Removed problematic token '{token}' for GCG compatibility"
                            )

                return formatted.strip()

            except Exception as e:
                print(f"FastChat formatting failed: {e}, using fallback")

        # Use HuggingFace chat template
        elif conv_format["format_type"] == "chat_template":
            try:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if avoid_special_tokens and conv_format["eos_token"]:
                    formatted = formatted.replace(conv_format["eos_token"], "")
                return formatted.strip()
            except Exception as e:
                print(f"HuggingFace chat template failed: {e}, using manual fallback")

        # Manual formatting fallback
        template = conv_format.get(
            "template", "Human: {user_message}\nAssistant: {bot_response}"
        )

        if "{user_role}" in template:
            formatted = template.format(
                user_role=conv_format["roles"][0],
                user_message=prompt,
                bot_role=conv_format["roles"][1],
                bot_response=target,
            )
        else:
            formatted = template.format(user_message=prompt, bot_response=target)

        return formatted.strip()

    @staticmethod
    def normalize_template(conv_template, tokenizer):
        """Normalize conversation template using FastChat knowledge.

        Updates the conversation template with proper settings from FastChat
        or fallback configurations, optimized for GCG attacks.

        Args:
            conv_template: Conversation template object to normalize
            tokenizer: HuggingFace tokenizer instance

        Returns:
            Normalized conversation template object
        """
        model_name = getattr(tokenizer, "name_or_path", "unknown")

        if FASTCHAT_AVAILABLE:
            try:
                fastchat_conv = ConversationTemplateAdapter.get_fastchat_template(
                    model_name
                )
                if fastchat_conv is not None:
                    # Use FastChat's template settings
                    conv_template.roles = tuple(fastchat_conv.roles)
                    conv_template.sep = getattr(fastchat_conv, "sep", "\n")
                    conv_template.sep2 = getattr(
                        fastchat_conv, "sep2", fastchat_conv.sep
                    )
                    conv_template.system = getattr(fastchat_conv, "system", "")

                    # For GCG, disable problematic tokens
                    conv_template.eos_token = ""

                    print(
                        f"Normalized template using FastChat '{fastchat_conv.name}' for {model_name}"
                    )
                    return conv_template
            except Exception as e:
                print(f"FastChat normalization failed: {e}")

        # Fallback normalization
        conv_format = ConversationTemplateAdapter.get_universal_conversation_format(
            model_name, tokenizer
        )
        conv_template.roles = tuple(conv_format["roles"])
        conv_template.sep = conv_format["sep"]
        conv_template.sep2 = conv_format.get("sep2", conv_format["sep"])
        conv_template.eos_token = ""  # Always disable for GCG

        return conv_template

    @staticmethod
    def get_special_tokens_info(tokenizer):
        """Extract special token information from any HuggingFace tokenizer.

        Safely extracts both token IDs and token strings for special tokens,
        handling cases where tokens may not be defined.

        Args:
            tokenizer: HuggingFace tokenizer instance

        Returns:
            Dictionary containing special token IDs and strings, with None
            values for undefined tokens
        """
        return {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "bos_token": getattr(tokenizer, "bos_token", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "unk_token": getattr(tokenizer, "unk_token", None),
        }

    @staticmethod
    def list_supported_templates() -> List[str]:
        """List all available FastChat conversation templates.

        Discovers and returns a list of FastChat conversation template names
        that are available in the current environment.

        Returns:
            List of template names if FastChat is available,
            error message list if not installed
        """
        if not FASTCHAT_AVAILABLE:
            return ["FastChat not installed"]

        try:
            template_names = []
            common_templates = [
                "vicuna_v1.1",
                "llama-2",
                "mistral",
                "alpaca",
                "zero_shot",
                "one_shot",
                "raw",
                "phi",
                "qwen",
                "gemma",
                "guanaco",
                "orca",
                "claude",
                "chatglm",
                "baichuan-chat",
                "internlm-chat",
            ]

            for template in common_templates:
                try:
                    get_conv_template(template)
                    template_names.append(template)
                except:
                    pass

            return template_names
        except:
            return ["Could not retrieve FastChat templates"]
