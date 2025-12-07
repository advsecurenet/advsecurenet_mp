This folder contains examples of how to use the `AdvSecureNet` toolkit. The examples are organized in subfolders according to the type of task they address. We provide examples for both API and CLI usage. The examples are written in Python and can be run in a Jupyter notebook environment. The following is a list of the examples available:

### API Examples

- [**Adversarial Attacks**](./advsecurenet/adversarial_attacks/adversarial_attacks.ipynb): This file contains examples of how to use the `AdvSecureNet` toolkit to generate adversarial examples.
- [**Benign Training**](./advsecurenet/benign_training/benign_training.ipynb): This file shows how to train a benign model using the `AdvSecureNet` toolkit. It also demonstrates how to use a external model to train.
- [**Adversarial Training**](./advsecurenet/defenses/adversarial_training/adversarial_training.ipynb): This file shows how to train an adversarial model using the `AdvSecureNet` toolkit.
- [**Evaluation**](./advsecurenet/evaluation/evaluation.ipynb): This file shows how to use the evaluators provided by the `AdvSecureNet` toolkit to evaluate the performance of a model, attack, or defense.
- [**Differential Privacy**](./advsecurenet/differential_privacy/differential_privacy_training.ipynb): This file shows how to train a model with Differential Privacy using Opacus.
- [**Hugging Face**](./advsecurenet/huggingface/huggingface_api_examples.ipynb): This file demonstrates how to load models and datasets from Hugging Face.
- [**LLM GCG Attack**](./advsecurenet/llm/GCG/universal_gcg_demo.ipynb): This file demonstrates the Greedy Coordinate Gradient (GCG) attack on LLMs.
- [**LLM Fine-Tuning**](./advsecurenet/llm/llm_finetuning/Finetuning_Notebook.ipynb): This file shows how to fine-tune an LLM using the Hugging Face Trainer.

### CLI Examples

- [**Distributed Adversarial Attacks**](./cli/adversarial_attacks/distributed): This folder contains examples of how to use the `AdvSecureNet` toolkit to generate adversarial examples in a distributed manner.
- [**Non-Distributed Adversarial Attacks**](./cli/adversarial_attacks/distributed): This folder contains examples of how to use the `AdvSecureNet` toolkit to generate adversarial examples in a non-distributed manner.
- [**Distributed Benign Training**](./cli/benign_training/distributed): This folder contains examples of how to train a benign model using the `AdvSecureNet` toolkit in a distributed manner.
- [**Non-Distributed Benign Training**](./cli/benign_training/non_distributed): This folder contains examples of how to train a benign model using the `AdvSecureNet` toolkit in a non-distributed manner.
- [**Distributed Adversarial Training**](./cli/defenses/adversarial_training/distributed): This folder contains examples of how to train an adversarial model using the `AdvSecureNet` toolkit in a distributed manner.
- [**Non-Distributed Adversarial Training**](./cli/defenses/adversarial_training/non_distributed): This folder contains examples of how to train an adversarial model using the `AdvSecureNet` toolkit in a non-distributed manner.
- [**Adversarial Evaluation**](./cli/evaluation/adversarial_evaluation): This folder contains examples of how to use the evaluators provided by the `AdvSecureNet` toolkit to evaluate the performance of a model, attack, or defense. There are examples for both distributed and non-distributed evaluation.
- [**Benign Evaluation**](./cli/evaluation/benign_evaluation): This folder contains examples of how to use the evaluators provided by the `AdvSecureNet` toolkit to evaluate the performance of a model in a benign setting.
- [**Object Detection Attacks**](./cli/adversarial_od_attacks): This folder contains examples of how to run adversarial attacks on object detection models.
- [**LLM Attacks**](./cli/llm): This folder contains examples of how to run adversarial attacks on Large Language Models.
- [**Differential Privacy**](./cli/differential_privacy): This folder contains examples of how to train models with Differential Privacy using the CLI.
- [**Hugging Face**](./cli/huggingface): This folder contains examples of how to use Hugging Face models and datasets with the CLI.
