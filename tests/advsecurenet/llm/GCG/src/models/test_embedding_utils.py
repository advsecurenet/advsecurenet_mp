import pytest
import torch
import json
import numpy as np
from unittest.mock import MagicMock, patch
from advsecurenet.llm.GCG.src.models.embedding_utils import (
    NpEncoder, get_embedding_layer, get_embedding_matrix, 
    get_embeddings, get_nonascii_toks
)
from transformers import LlamaForCausalLM, GPTNeoXForCausalLM


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

    def test_numpy_bool_encoding(self):
        encoder = NpEncoder()
        np_bool = np.bool_(True)
        
        result = encoder.default(np_bool)
        
        assert result is True
        assert isinstance(result, bool)

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
    def test_gpt_neox_model_isinstance(self):
        # Test direct GPTNeoXForCausalLM isinstance check (line 23)
        from advsecurenet.llm.GCG.src.models.embedding_utils import GPTNeoXForCausalLM
        
        # Create a mock that will pass the isinstance check for GPTNeoXForCausalLM
        mock_model = MagicMock(spec=GPTNeoXForCausalLM)
        mock_embedding = MagicMock()
        mock_model.base_model.embed_in = mock_embedding
        
        # Set the __class__ attribute to make isinstance work
        mock_model.__class__ = GPTNeoXForCausalLM
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_embeddings_word_embeddings_fallback(self):
        # Test embeddings.word_embeddings fallback (line 35)
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Remove all other attributes to force embeddings fallback
        mock_model.transformer = None
        mock_model.model = None
        mock_model.gpt_neox = None
        mock_model.embeddings.word_embeddings = mock_embedding
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_gptj_model_fallback_transformer_wte(self):
        # Test the transformer.wte fallback path (GPT-J style)
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_model.transformer.wte = mock_embedding
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_llama_model_fallback_model_embed_tokens(self):
        # Test the model.embed_tokens fallback path (LLaMA style)
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Remove transformer to avoid that fallback
        mock_model.transformer = None
        mock_model.model.embed_tokens = mock_embedding
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_gpt_neox_fallback_gpt_neox_embed_in(self):
        # Test the gpt_neox.embed_in fallback path (GPT-NeoX style)
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Remove other attributes to force this fallback
        mock_model.transformer = None
        mock_model.model = None
        mock_model.gpt_neox.embed_in = mock_embedding
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_transformer_word_embeddings_fallback(self):
        # Test transformer.word_embeddings fallback (BERT-style)
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Create proper nested mocking to control hasattr behavior
        mock_transformer = MagicMock()
        # Explicitly remove wte attribute so hasattr returns False
        del mock_transformer.wte
        mock_transformer.word_embeddings = mock_embedding
        mock_model.transformer = mock_transformer
        
        # Remove model and gpt_neox to force transformer path
        mock_model.model = None
        mock_model.gpt_neox = None
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_transformer_wte_fallback(self):
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_model.transformer.wte = mock_embedding
        
        # Simply test the fallback path by not setting __class__ to any specific type
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_model_embed_tokens_fallback(self):
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Remove transformer to force model.embed_tokens path
        mock_model.transformer = None
        mock_model.model.embed_tokens = mock_embedding
        
        # Simply test the fallback path by not setting __class__ to any specific type
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_embedding_layer_search_fallback(self):
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        
        # Remove all other attributes to force named_modules search
        mock_model.transformer = None
        mock_model.model = None
        mock_model.gpt_neox = None
        
        # Delete embeddings attribute entirely to avoid hasattr issue
        if hasattr(mock_model, 'embeddings'):
            delattr(mock_model, 'embeddings')
        
        # Mock named_modules to return an embedding layer
        mock_model.named_modules.return_value = [
            ('other_layer', MagicMock()),
            ('word_embeddings', mock_embedding)
        ]
        
        # Make the embedding layer be recognized as torch.nn.Embedding
        mock_embedding.__class__ = torch.nn.Embedding
        
        result = get_embedding_layer(mock_model)
        
        assert result == mock_embedding

    def test_no_embedding_layer_found(self):
        mock_model = MagicMock()
        
        # Remove all potential embedding attributes completely
        for attr in ['transformer', 'model', 'gpt_neox', 'embeddings']:
            if hasattr(mock_model, attr):
                delattr(mock_model, attr)
        
        # Mock named_modules to return no embedding layers
        mock_model.named_modules.return_value = [
            ('other_layer', MagicMock())
        ]
        
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
            
            assert torch.equal(result, mock_weight)
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
        
        # Create a mock weight without half attribute
        mock_weight = MagicMock()
        if hasattr(mock_weight, 'half'):
            delattr(mock_weight, 'half')
        mock_embedding_layer.weight = mock_weight
        
        input_ids = torch.tensor([1, 2, 3])
        
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
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

    def test_get_nonascii_toks_with_unk_token(self):
        # Test the unk_token_id branch (line 70)
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 6
        
        mock_tokenizer.decode.side_effect = lambda x: {
            3: "a",      # ascii
            4: "é",      # non-ascii
            5: "b"       # ascii
        }[x[0]] if len(x) == 1 else "multi"
        
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = 99  # Test this specific branch
        
        result = get_nonascii_toks(mock_tokenizer, device='cpu')
        
        # Should include non-ascii token (4) and unk token (99)
        expected = torch.tensor([4, 99], device='cpu')
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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
        
        # Test matrix retrieval
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
            matrix = get_embedding_matrix(mock_model)
            assert torch.equal(matrix, mock_weight)
        
        # Test embedding computation
        input_ids = torch.tensor([1, 2, 3])
        mock_output = torch.randn(3, 64)
        mock_embedding_layer.return_value = mock_output
        
        # Create a mock weight without half attribute
        mock_weight_no_half = MagicMock()
        if hasattr(mock_weight_no_half, 'half'):
            delattr(mock_weight_no_half, 'half')
        mock_embedding_layer.weight = mock_weight_no_half
        
        with patch('advsecurenet.llm.GCG.src.models.embedding_utils.get_embedding_layer') as mock_get_layer:
            mock_get_layer.return_value = mock_embedding_layer
            
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
        
        # Use approximate equality for float comparison
        assert abs(parsed['metrics']['accuracy'] - 0.95) < 1e-6
        assert parsed['metrics']['loss'] == 0.123
        assert parsed['metrics']['counts'] == [10, 20, 30]
        assert parsed['metrics']['confusion_matrix'] == [[5, 1], [2, 8]]
        assert parsed['metadata']['epochs'] == 100
        assert parsed['metadata']['batch_size'] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])