import os
import subprocess

def run_attack():
    model = "llama2"
    setup = "behaviors"

    os.environ["WANDB_MODE"] = "disabled"
    # Optional: os.environ["TRANSFORMERS_CACHE"] = "YOUR_PATH/huggingface"

    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Folder '{results_dir}' created.")
    else:
        print(f"Folder '{results_dir}' already exists.")

    for data_offset in range(0, 100, 10):
        command = [
            "python3", "-u", "advsecurenet/llm/llm_eval/main.py",
            f"--config=advsecurenet/llm/llm_eval/configs/individual_{model}.py",
            "--config.attack=gcg",
            f"--config.train_data=advsecurenet/llm/datasets/advbench/harmful_{setup}.csv",
            f"--config.result_prefix=../results/individual_{setup}_{model}_gcg_offset{data_offset}",
            "--config.n_train_data=10",
            f"--config.data_offset={data_offset}",
            "--config.n_steps=1000",
            "--config.test_steps=50",
            "--config.batch_size=8"
        ]

        print(f"Running command with offset {data_offset}...")
        subprocess.run(command, check=True)

if __name__ == "__main__":
    run_attack()
