from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import math

# Load dataset
dataset = load_dataset("trl-lib/Capybara")["test"]  # 200 samples

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding works

# Use CPU
device = torch.device("cpu")

def compute_perplexity(model, tokenizer, dataset, max_samples=10):
    model.eval()
    model.to(device)
    losses = []

    for sample in dataset.select(range(max_samples)):
        # Format as chat if needed
        num_iter = 0
        if "messages" in sample:
            input_ids = tokenizer.apply_chat_template(
                sample["messages"], return_tensors="pt", truncation=True, max_length=512
            ).to(device)
        else:
            input_ids = tokenizer(
                sample["text"], return_tensors="pt", truncation=True, max_length=512
            ).input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())

        num_iter += 1
        print(f"Finished {num_iter}/{max_samples} samples")

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return perplexity

# Load base model on CPU
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B", trust_remote_code=True
).to(device)

# Load fine-tuned model on CPU
ft_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B", trust_remote_code=True
).to(device)

ft_model = PeftModel.from_pretrained(ft_model, "./Qwen/Qwen2-0.5B").to(device)

# Compute perplexities
base_ppl = compute_perplexity(base_model, tokenizer, dataset)
ft_ppl = compute_perplexity(ft_model, tokenizer, dataset)

print(f"Base Model Perplexity: {base_ppl:.2f}")
print(f"Fine-Tuned Model Perplexity: {ft_ppl:.2f}")
