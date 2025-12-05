"""Model worker for multiprocessing-based model inference and gradient computation.

This module provides a worker class that enables running model operations
in separate processes, useful for distributed computing, parallel processing,
and memory management during adversarial attacks like GCG.

The ModelWorker class handles:
- Loading and managing transformer models in separate processes
- Asynchronous task execution with gradient and no-gradient contexts
- Memory management and process lifecycle
- Task queuing and result collection
"""

from transformers import AutoModelForCausalLM
import torch
import torch.multiprocessing as mp
from copy import deepcopy


class ModelWorker(object):
    """Worker class for running model operations in separate processes.

    This class enables multiprocessing-based model inference and gradient
    computation, which is useful for:
    - Parallel processing of multiple models
    - Memory isolation between different attack experiments
    - Distributed GCG attacks
    - Resource management in memory-constrained environments

    The worker communicates through multiprocessing queues and supports
    various operation types including gradient computation, logits calculation,
    and custom function execution.

    Attributes:
        model: The loaded transformer model instance
        tokenizer: Associated tokenizer for the model
        conv_template: Conversation template for formatting inputs
        tasks: Multiprocessing queue for incoming tasks
        results: Multiprocessing queue for task results
        process: The subprocess running the worker
    """

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        """Initialize the ModelWorker with model and configuration.

        Args:
            model_path: Path or name of the HuggingFace model to load
            model_kwargs: Additional keyword arguments for model loading
            tokenizer: Tokenizer instance associated with the model
            conv_template: Conversation template for input formatting
            device: Device (CPU/GPU) to load the model on
        """
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                **model_kwargs,
            )
            .to(device)
            .eval()
        )
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue()
        self.process = None

    @staticmethod
    def run(model, tasks, results):
        """Main worker loop that processes tasks from the queue.

        This static method runs in a separate process and continuously
        processes tasks from the task queue. It handles different types
        of operations with appropriate gradient contexts:

        - 'grad': Operations requiring gradient computation
        - 'logits': Forward pass for logits computation
        - 'contrast_logits': Contrastive logits computation
        - 'test': Model testing operations
        - 'test_loss': Loss computation for testing
        - Custom functions: Direct function execution

        Args:
            model: The model instance to operate on
            tasks: Multiprocessing queue containing tasks to execute
            results: Multiprocessing queue to put results into

        Note:
            This method runs indefinitely until a None task is received,
            which signals the worker to terminate.
        """
        while True:
            task = tasks.get()
            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

    def start(self):
        """Start the worker process.

        Creates and starts a new process running the worker loop.
        The process will continuously process tasks from the task queue
        until explicitly stopped.

        Returns:
            ModelWorker: Self for method chaining

        Note:
            Prints the process ID and model path for debugging purposes.
        """
        self.process = mp.Process(
            target=ModelWorker.run, args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        print(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self

    def stop(self):
        """Stop the worker process and clean up resources.

        Sends a termination signal (None) to the worker process,
        waits for it to finish, and clears GPU memory cache.

        Returns:
            ModelWorker: Self for method chaining

        Note:
            This method ensures proper cleanup of both the process
            and GPU memory to prevent resource leaks.
        """
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        """Submit a task to the worker process for execution.

        Makes the ModelWorker callable, allowing it to be used like a function
        to submit tasks for asynchronous execution in the worker process.

        Args:
            ob: Object instance to operate on (will be deep copied)
            fn: Function name or method to execute on the object
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            ModelWorker: Self for method chaining

        Note:
            The object is deep copied to ensure thread safety and avoid
            shared state issues between processes. Results must be retrieved
            separately from the results queue.
        """
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self
