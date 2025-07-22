from datasets import load_dataset
from advsecurenet.llm_new.trainer_pipeline.training_pipeline import TrainerPipeline

from transformers import RobertaForSequenceClassification
from advsecurenet.llm_new.trainer_pipeline.config import Config


if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("ag_news")
    
    custom_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
    config = Config(
        model_name="roberta-base",
        model_obj=custom_model,
        num_labels=4
    )

    pipeline = TrainerPipeline(
        config=config,
        dataset_input=dataset,
        text_column="text",
        label_column="label"
    )
    pipeline.run()
