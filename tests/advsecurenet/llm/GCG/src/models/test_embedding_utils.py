import pytest
import torch
import json
import numpy as np
from unittest.mock import MagicMock, patch
from advsecurenet.llm.GCG.src.models.embedding_utils import (
    NpEncoder, get_embedding_layer, get_embedding_matrix, 
    get_embeddings, get_nonascii_toks
)


class TestNpEncoder:
    def test_numpy_integer_encoding(self):
        encoder = NpEncoder()
        np_int = np.int32(42)
        
        result = encoder.default(np_int)
        
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_floating_encoding(self):
        encoder = NpEncoder()
        np_float = np.float64(3.14)
        
        result = encoder.default(np_float)
        
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_array_encoding(self):
        encoder = NpEncoder()
        np_array = np.array([1, 2, 3])
        
        result = encoder.default(np_array)
        
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_2d_array_encoding(self):
        encoder = NpEncoder()
        np_array = np.array([[1, 2], [3, 4]])
        
        result = encoder.default(np_array)
        
        assert result == [[1, 2], [3, 4]]

    def test_non_numpy_object(self):
        encoder = NpEncoder()
        regular_obj = "not numpy"
        
        with pytest.raises(TypeError):
            encoder.default(regular_obj)

    def test_json_dumps_integration(self):
        data = {
            'int': np.int64(123),
            'float': np.float32(1.5),
            'array': np.array([1, 2, 3])
        }
        
        result = json.dumps(data, cls=NpEncoder)
        parsed = json.loads(result)
        
        assert parsed['int'] == 123
        assert parsed['float'] == 1.5
        assert parsed['array'] == [1, 2, 3]


class TestGetEmbeddingLayer:
    def test_gptj_model(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'GPTJForCausalLM'
        
        # Mock isinstance to return True for GPTJForCausalLM
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTJForCausalLM') as mock_gptj:
            mock_model.__class__ = mock_gptj
            
            mock_embedding = MagicMock()
            mock_model.transformer.wte = mock_embedding
            
            result = get_embedding_layer(mock_model)
            
            assert result == mock_embedding

    def test_gpt2_model(self):
        mock_model = MagicMock()
        
        # Mock isinstance to return True for GPT2LMHeadModel
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPT2LMHeadModel') as mock_gpt2:
            mock_model.__class__ = mock_gpt2
            
            mock_embedding = MagicMock()
            mock_model.transformer.wte = mock_embedding
            
            result = get_embedding_layer(mock_model)
            
            assert result == mock_embedding

    def test_llama_model(self):
        mock_model = MagicMock()
        
        # Mock isinstance to return True for LlamaForCausalLM
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.LlamaForCausalLM') as mock_llama:
            mock_model.__class__ = mock_llama
            
            mock_embedding = MagicMock()
            mock_model.model.embed_tokens = mock_embedding
            
            result = get_embedding_layer(mock_model)
            
            assert result == mock_embedding

    def test_gpt_neox_model(self):
        mock_model = MagicMock()
        
        # Mock isinstance to return True for GPTNeoXForCausalLM
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTNeoXForCausalLM') as mock_neox:
            mock_model.__class__ = mock_neox
            
            mock_embedding = MagicMock()
            mock_model.base_model.embed_in = mock_embedding
            
            result = get_embedding_layer(mock_model)
            
            assert result == mock_embedding

    def test_transformer_wte_fallback(self):
        mock_model = MagicMock()
        
        # Mock all isinstance checks to return False
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTJForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPT2LMHeadModel'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.LlamaForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTNeoXForCausalLM'):
            
            mock_embedding = MagicMock()
            mock_model.transformer.wte = mock_embedding
            
            # Make hasattr checks work
            def mock_hasattr(obj, name):
                if name == 'transformer':
                    return True
                elif name == 'wte' and hasattr(obj, 'transformer'):
                    return True
                return False
            
            with patch('builtins.hasattr', side_effect=mock_hasattr):
                result = get_embedding_layer(mock_model)
                
                assert result == mock_embedding

    def test_model_embed_tokens_fallback(self):
        mock_model = MagicMock()
        
        # Mock all isinstance checks to return False
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTJForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPT2LMHeadModel'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.LlamaForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTNeoXForCausalLM'):
            
            mock_embedding = MagicMock()
            mock_model.model.embed_tokens = mock_embedding
            
            # Mock hasattr to match the fallback pattern
            def mock_hasattr(obj, name):
                if name == 'transformer':
                    return False
                elif name == 'model':
                    return True
                elif name == 'embed_tokens' and hasattr(obj, 'model'):
                    return True
                return False
            
            with patch('builtins.hasattr', side_effect=mock_hasattr):
                result = get_embedding_layer(mock_model)
                
                assert result == mock_embedding

    def test_embedding_layer_search_fallback(self):
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Mock all isinstance and hasattr checks to return False initially
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTJForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPT2LMHeadModel'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.LlamaForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTNeoXForCausalLM'), \
             patch('builtins.hasattr', return_value=False):
            
            # Mock named_modules to return an embedding layer
            mock_model.named_modules.return_value = [
                ('other_layer', MagicMock()),
                ('word_embeddings', mock_embedding)
            ]
            
            with patch('builtins.isinstance') as mock_isinstance:
                mock_isinstance.side_effect = lambda obj, cls: (
                    cls == torch.nn.Embedding and obj == mock_embedding
                )
                
                result = get_embedding_layer(mock_model)
                
                assert result == mock_embedding

    def test_no_embedding_layer_found(self):
        mock_model = MagicMock()
        
        # Mock all checks to fail
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTJForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPT2LMHeadModel'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.LlamaForCausalLM'), \
             patch('advsecurenet.llm.GCG.src.models.embedding_utils.GPTNeoXForCausalLM'), \
             patch('builtins.hasattr', return_value=False):
            
            mock_model.named_modules.return_value = [
                ('other_layer', MagicMock())
            ]
            
            with patch('builtins.isinstance', return_value=False):
                with pytest.raises(ValueError, match="Could not find embedding layer"):
                    get_embedding_layer(mock_model)


class TestGetEmbeddingMatrix:
    def test_get_embedding_matrix(self):
        mock_model = MagicMock()
        mock_embedding_layer = MagicMock()
        mock_weight = torch.randn(100, 64)
        mock_embedding_layer.weight = mock_weight
        
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
            result = get_embedding_matrix(mock_model)
            
            assert result == mock_weight
            mock_get_layer.assert_called_once_with(mock_model)


class TestGetEmbeddings:
    def test_get_embeddings_with_half(self):
        mock_model = MagicMock()
        mock_embedding_layer = MagicMock()
        mock_output = MagicMock()
        mock_output.half = MagicMock(return_value="half_output")
        mock_embedding_layer.return_value = mock_output
        mock_embedding_layer.weight = MagicMock()
        mock_embedding_layer.weight.half = MagicMock()  # hasattr check
        
        input_ids = torch.tensor([1, 2, 3])
        
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
            result = get_embeddings(mock_model, input_ids)
            
            assert result == "half_output"
            mock_embedding_layer.assert_called_once_with(input_ids)
            mock_output.half.assert_called_once()

    def test_get_embeddings_without_half(self):
        mock_model = MagicMock()
        mock_embedding_layer = MagicMock()
        mock_output = MagicMock()
        mock_embedding_layer.return_value = mock_output
        mock_embedding_layer.weight = MagicMock()
        
        # Remove half attribute to simulate no half precision support
        if hasattr(mock_embedding_layer.weight, 'half'):
            delattr(mock_embedding_layer.weight, 'half')
        
        input_ids = torch.tensor([1, 2, 3])
        
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
            with patch('builtins.hasattr', return_value=False):
                result = get_embeddings(mock_model, input_ids)
                
                assert result == mock_output
                mock_embedding_layer.assert_called_once_with(input_ids)


class TestGetNonasciiToks:
    def test_get_nonascii_toks_basic(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 10
        
        # Mock decode to return ascii and non-ascii strings
        mock_tokenizer.decode.side_effect = lambda x: {
            3: "a",      # ascii
            4: "b",      # ascii
            5: "é",      # non-ascii
            6: "ñ",      # non-ascii
            7: "c",      # ascii
            8: "🚀",     # non-ascii
            9: "d"       # ascii
        }[x[0]] if len(x) == 1 else "multi"
        
        # Mock special token IDs
        mock_tokenizer.bos_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.pad_token_id = 2
        mock_tokenizer.unk_token_id = None
        
        result = get_nonascii_toks(mock_tokenizer, device='cpu')
        
        # Should include non-ascii tokens (5, 6, 8) and special tokens (0, 1, 2)
        expected = torch.tensor([5, 6, 8, 0, 1, 2], device='cpu')
        
        # Sort both tensors for comparison since order might vary
        assert torch.equal(torch.sort(result)[0], torch.sort(expected)[0])

    def test_get_nonascii_toks_no_special_tokens(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 6
        
        mock_tokenizer.decode.side_effect = lambda x: {
            3: "a",      # ascii
            4: "é",      # non-ascii
            5: "b"       # ascii
        }[x[0]] if len(x) == 1 else "multi"
        
        # All special tokens are None
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        
        result = get_nonascii_toks(mock_tokenizer, device='cpu')
        
        expected = torch.tensor([4], device='cpu')
        assert torch.equal(result, expected)

    def test_get_nonascii_toks_cuda_device(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 5
        
        mock_tokenizer.decode.side_effect = lambda x: {
            3: "é",      # non-ascii
            4: "a"       # ascii
        }[x[0]] if len(x) == 1 else "multi"
        
        mock_tokenizer.bos_token_id = 0
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        
        result = get_nonascii_toks(mock_tokenizer, device='cuda')
        
        expected = torch.tensor([3, 0], device='cuda')
        assert torch.equal(torch.sort(result)[0], torch.sort(expected)[0])
        assert result.device.type == 'cuda'

    def test_get_nonascii_toks_printability_check(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 7
        
        # Include non-printable ascii character
        mock_tokenizer.decode.side_effect = lambda x: {
            3: "a",      # ascii and printable
            4: "\x00",   # ascii but not printable
            5: "é",      # non-ascii
            6: "\n"      # ascii but not printable
        }[x[0]] if len(x) == 1 else "multi"
        
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        
        result = get_nonascii_toks(mock_tokenizer, device='cpu')
        
        # Should include non-ascii (5) and non-printable ascii (4, 6)
        expected = torch.tensor([4, 5, 6], device='cpu')
        assert torch.equal(torch.sort(result)[0], torch.sort(expected)[0])

    def test_get_nonascii_toks_empty_result(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 5
        
        # All tokens are ascii and printable
        mock_tokenizer.decode.side_effect = lambda x: {
            3: "a",
            4: "b"
        }[x[0]] if len(x) == 1 else "multi"
        
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        
        result = get_nonascii_toks(mock_tokenizer, device='cpu')
        
        expected = torch.tensor([], device='cpu', dtype=torch.long)
        assert torch.equal(result, expected)

    def test_is_ascii_helper_function(self):
        # Test the internal is_ascii function behavior
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 8
        
        test_cases = {
            3: "hello",    # ascii + printable = True (should NOT be in result)
            4: "café",     # non-ascii = False (should be in result)
            5: "\t",       # ascii but not printable = False (should be in result)
            6: "\x1f",     # ascii but not printable = False (should be in result) 
            7: "world"     # ascii + printable = True (should NOT be in result)
        }
        
        mock_tokenizer.decode.side_effect = lambda x: test_cases[x[0]] if len(x) == 1 else "multi"
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        
        result = get_nonascii_toks(mock_tokenizer, device='cpu')
        
        # Should include tokens 4, 5, 6 (non-ascii or non-printable)
        expected = torch.tensor([4, 5, 6], device='cpu')
        assert torch.equal(torch.sort(result)[0], torch.sort(expected)[0])


class TestIntegration:
    def test_full_embedding_workflow(self):
        # Create a more realistic mock model
        mock_model = MagicMock()
        mock_embedding_layer = MagicMock()
        mock_weight = torch.randn(100, 64)
        mock_embedding_layer.weight = mock_weight
        
        # Test full workflow
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
            # Test matrix retrieval
            matrix = get_embedding_matrix(mock_model)
            assert matrix == mock_weight
            
            # Test embedding computation
            input_ids = torch.tensor([1, 2, 3])
            mock_output = torch.randn(3, 64)
            mock_embedding_layer.return_value = mock_output
            
            with patch('builtins.hasattr', return_value=False):
                embeddings = get_embeddings(mock_model, input_ids)
                assert torch.equal(embeddings, mock_output)

    def test_npencoder_with_complex_data(self):
        # Test NpEncoder with nested structures
        complex_data = {
            'metrics': {
                'accuracy': np.float32(0.95),
                'loss': np.float64(0.123),
                'counts': np.array([10, 20, 30]),
                'confusion_matrix': np.array([[5, 1], [2, 8]])
            },
            'metadata': {
                'epochs': np.int64(100),
                'batch_size': np.int32(32)
            }
        }
        
        json_str = json.dumps(complex_data, cls=NpEncoder)
        parsed = json.loads(json_str)
        
        assert parsed['metrics']['accuracy'] == 0.95
        assert parsed['metrics']['loss'] == 0.123
        assert parsed['metrics']['counts'] == [10, 20, 30]
        assert parsed['metrics']['confusion_matrix'] == [[5, 1], [2, 8]]
        assert parsed['metadata']['epochs'] == 100
        assert parsed['metadata']['batch_size'] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])