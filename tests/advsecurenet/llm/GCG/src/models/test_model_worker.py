import pytest
import torch
import torch.multiprocessing as mp
from unittest.mock import MagicMock, patch, Mock
from advsecurenet.llm.GCG.src.models.model_worker import ModelWorker


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.name_or_path = "test-tokenizer"
    return tokenizer


@pytest.fixture
def mock_conv_template():
    template = MagicMock()
    template.name = "test-template"
    return template


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.name_or_path = "test-model"
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


class TestModelWorkerInit:
    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    def test_basic_initialization(self, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={"low_cpu_mem_usage": True},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        # Verify model loading
        mock_from_pretrained.assert_called_once_with(
            "test/model",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Verify model setup
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()
        
        # Verify worker attributes
        assert worker.model == mock_model
        assert worker.tokenizer == mock_tokenizer
        assert worker.conv_template == mock_conv_template
        assert worker.process is None

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    def test_initialization_with_cuda_device(self, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cuda:0"
        )
        
        mock_model.to.assert_called_once_with("cuda:0")

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    def test_initialization_with_custom_kwargs(self, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        custom_kwargs = {
            "use_cache": False,
            "low_cpu_mem_usage": True,
            "custom_param": "value"
        }
        
        worker = ModelWorker(
            model_path="custom/model",
            model_kwargs=custom_kwargs,
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        mock_from_pretrained.assert_called_once_with(
            "custom/model",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            custom_param="value"
        )

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    def test_multiprocessing_queues_created(self, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        # Verify queues are created and have expected methods
        assert hasattr(worker.tasks, 'put')
        assert hasattr(worker.tasks, 'get')
        assert hasattr(worker.tasks, 'task_done')
        assert hasattr(worker.tasks, 'join')
        
        assert hasattr(worker.results, 'put')
        assert hasattr(worker.results, 'get')
        assert hasattr(worker.results, 'task_done')
        assert hasattr(worker.results, 'join')


class TestModelWorkerRun:
    def test_run_static_method_grad_task(self):
        # Create mock objects
        mock_model = MagicMock()
        mock_ob = MagicMock()
        mock_ob.grad = MagicMock(return_value="grad_result")
        
        # Create mock queues
        tasks = MagicMock()
        results = MagicMock()
        
        # Setup task sequence: grad task, then None to stop
        tasks.get.side_effect = [
            (mock_ob, "grad", ("arg1",), {"kwarg1": "value1"}),
            None
        ]
        
        with patch('torch.enable_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        # Verify grad was called
        mock_ob.grad.assert_called_once_with("arg1", kwarg1="value1")
        results.put.assert_called_with("grad_result")
        tasks.task_done.assert_called()

    def test_run_static_method_logits_task(self):
        mock_model = MagicMock()
        mock_ob = MagicMock()
        mock_ob.logits = MagicMock(return_value="logits_result")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (mock_ob, "logits", (), {}),
            None
        ]
        
        with patch('torch.no_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        mock_ob.logits.assert_called_once_with()
        results.put.assert_called_with("logits_result")

    def test_run_static_method_contrast_logits_task(self):
        mock_model = MagicMock()
        mock_ob = MagicMock()
        mock_ob.contrast_logits = MagicMock(return_value="contrast_result")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (mock_ob, "contrast_logits", ("arg",), {}),
            None
        ]
        
        with patch('torch.no_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        mock_ob.contrast_logits.assert_called_once_with("arg")
        results.put.assert_called_with("contrast_result")

    def test_run_static_method_test_task(self):
        mock_model = MagicMock()
        mock_ob = MagicMock()
        mock_ob.test = MagicMock(return_value="test_result")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (mock_ob, "test", (), {"test_param": True}),
            None
        ]
        
        with patch('torch.no_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        mock_ob.test.assert_called_once_with(test_param=True)
        results.put.assert_called_with("test_result")

    def test_run_static_method_test_loss_task(self):
        mock_model = MagicMock()
        mock_ob = MagicMock()
        mock_ob.test_loss = MagicMock(return_value="test_loss_result")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (mock_ob, "test_loss", ("loss_arg",), {}),
            None
        ]
        
        with patch('torch.no_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        mock_ob.test_loss.assert_called_once_with("loss_arg")
        results.put.assert_called_with("test_loss_result")

    def test_run_static_method_custom_function_task(self):
        mock_model = MagicMock()
        mock_custom_fn = MagicMock(return_value="custom_result")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (None, mock_custom_fn, ("arg1", "arg2"), {"custom_kwarg": "value"}),
            None
        ]
        
        with patch('torch.no_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        mock_custom_fn.assert_called_once_with("arg1", "arg2", custom_kwarg="value")
        results.put.assert_called_with("custom_result")

    def test_run_static_method_multiple_tasks(self):
        mock_model = MagicMock()
        mock_ob1 = MagicMock()
        mock_ob1.logits = MagicMock(return_value="result1")
        mock_ob2 = MagicMock()
        mock_ob2.test = MagicMock(return_value="result2")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (mock_ob1, "logits", (), {}),
            (mock_ob2, "test", (), {}),
            None
        ]
        
        with patch('torch.no_grad'):
            ModelWorker.run(mock_model, tasks, results)
        
        # Verify both tasks were processed
        assert results.put.call_count == 2
        assert tasks.task_done.call_count == 2


class TestModelWorkerStart:
    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    @patch('torch.multiprocessing.Process')
    def test_start_creates_process(self, mock_process_class, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        result = worker.start()
        
        # Verify process creation
        mock_process_class.assert_called_once_with(
            target=ModelWorker.run,
            args=(mock_model, worker.tasks, worker.results)
        )
        mock_process.start.assert_called_once()
        
        # Verify worker state
        assert worker.process == mock_process
        assert result == worker  # Should return self

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    @patch('torch.multiprocessing.Process')
    @patch('builtins.print')
    def test_start_prints_message(self, mock_print, mock_process_class, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        worker.start()
        
        # Verify print was called
        mock_print.assert_called_once_with("Started worker 12345 for model test-model")


class TestModelWorkerStop:
    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.empty_cache')
    def test_stop_with_process(self, mock_empty_cache, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        # Mock a running process
        mock_process = MagicMock()
        worker.process = mock_process
        
        # Mock tasks queue
        worker.tasks = MagicMock()
        
        result = worker.stop()
        
        # Verify stop sequence
        worker.tasks.put.assert_called_once_with(None)  # Send stop signal
        mock_process.join.assert_called_once()  # Wait for process
        mock_empty_cache.assert_called_once()  # Clear GPU memory
        
        assert result == worker  # Should return self

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.empty_cache')
    def test_stop_without_process(self, mock_empty_cache, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        # No process set (worker.process is None)
        worker.tasks = MagicMock()
        
        result = worker.stop()
        
        # Should still send stop signal and clear cache
        worker.tasks.put.assert_called_once_with(None)
        mock_empty_cache.assert_called_once()
        
        assert result == worker


class TestModelWorkerCall:
    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.models.model_worker.deepcopy')
    def test_call_basic(self, mock_deepcopy, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        mock_ob = MagicMock()
        mock_deepcopy.return_value = mock_ob
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        worker.tasks = MagicMock()
        
        result = worker(mock_ob, "test_fn", "arg1", "arg2", kwarg1="value1")
        
        # Verify deepcopy was called
        mock_deepcopy.assert_called_once_with(mock_ob)
        
        # Verify task was queued
        worker.tasks.put.assert_called_once_with((mock_ob, "test_fn", ("arg1", "arg2"), {"kwarg1": "value1"}))
        
        assert result == worker  # Should return self

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.models.model_worker.deepcopy')
    def test_call_with_no_args_kwargs(self, mock_deepcopy, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        mock_ob = MagicMock()
        mock_deepcopy.return_value = mock_ob
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        worker.tasks = MagicMock()
        
        result = worker(mock_ob, "simple_fn")
        
        # Verify task was queued with empty args and kwargs
        worker.tasks.put.assert_called_once_with((mock_ob, "simple_fn", (), {}))
        
        assert result == worker


class TestModelWorkerEdgeCases:
    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    def test_initialization_model_loading_failure(self, mock_from_pretrained, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            ModelWorker(
                model_path="invalid/model",
                model_kwargs={},
                tokenizer=mock_tokenizer,
                conv_template=mock_conv_template,
                device="cpu"
            )

    @patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained')
    def test_empty_model_kwargs(self, mock_from_pretrained, mock_model, mock_tokenizer, mock_conv_template):
        mock_from_pretrained.return_value = mock_model
        
        worker = ModelWorker(
            model_path="test/model",
            model_kwargs={},  # Empty kwargs
            tokenizer=mock_tokenizer,
            conv_template=mock_conv_template,
            device="cpu"
        )
        
        # Should still work with empty kwargs
        mock_from_pretrained.assert_called_once_with(
            "test/model",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    def test_run_with_exception_in_task(self):
        mock_model = MagicMock()
        mock_ob = MagicMock()
        mock_ob.grad.side_effect = Exception("Task execution failed")
        
        tasks = MagicMock()
        results = MagicMock()
        
        tasks.get.side_effect = [
            (mock_ob, "grad", (), {}),
            None
        ]
        
        # Should not crash even if task raises exception
        with patch('torch.enable_grad'):
            try:
                ModelWorker.run(mock_model, tasks, results)
            except Exception:
                # Exception should not propagate from worker
                pass

    def test_multiple_start_calls(self, mock_tokenizer, mock_conv_template):
        with patch('advsecurenet.llm.GCG.src.models.model_worker.AutoModelForCausalLM.from_pretrained'):
            with patch('torch.multiprocessing.Process') as mock_process_class:
                mock_process = MagicMock()
                mock_process_class.return_value = mock_process
                
                worker = ModelWorker(
                    model_path="test/model",
                    model_kwargs={},
                    tokenizer=mock_tokenizer,
                    conv_template=mock_conv_template,
                    device="cpu"
                )
                
                # Start multiple times
                worker.start()
                worker.start()
                
                # Should create new process each time
                assert mock_process_class.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])