from advsecurenet.llm_finetuning.training_pipeline import ClassificationPipeline

Gpt2_finetuning = ClassificationPipeline(config_path="/Users/philip/Desktop/advsecurenet_mp/advsecurenet/llm_finetuning/configs/gpt2_classification_config.yaml")

Gpt2_finetuning.run()