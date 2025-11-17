class ConversationTemplateAdapter:
    """Adapter to normalize different conversation templates."""
    
    @staticmethod
    def normalize_template(conv_template, tokenizer):
        """Normalize conversation template for consistent handling."""
        
        # Ensure we have proper role names
        if not hasattr(conv_template, 'roles') or len(conv_template.roles) < 2:
            conv_template.roles = ['User', 'Assistant']
        
        # Handle missing separators
        if not hasattr(conv_template, 'sep'):
            conv_template.sep = '\n'
        if not hasattr(conv_template, 'sep2'):
            conv_template.sep2 = conv_template.sep
            
        # Ensure proper system message handling
        if not hasattr(conv_template, 'system'):
            conv_template.system = ""
            
        return conv_template
    
    @staticmethod
    def get_special_tokens_info(tokenizer):
        """Extract special token information for slice detection."""
        return {
            'bos_token_id': getattr(tokenizer, 'bos_token_id', None),
            'eos_token_id': getattr(tokenizer, 'eos_token_id', None),
            'pad_token_id': getattr(tokenizer, 'pad_token_id', None),
            'unk_token_id': getattr(tokenizer, 'unk_token_id', None),
        }
