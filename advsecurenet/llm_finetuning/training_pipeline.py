from trainer_pipeline.config import Config
from trainer_pipeline.model_loader import ModelLoader
from trainer_pipeline.data_loader import DataLoaderWrapper
from advsecurenet.llm_new.trainer_pipeline.trainer_wrapper import TrainerWrapper

class TrainerPipeline:
    def __init__(self, config, dataset_input, text_column, label_column, split_keys=("train", "test")):
        self.config = config
        self.dataset_input = dataset_input
        self.text_column = text_column
        self.label_column = label_column
        self.split_keys = split_keys

    def run(self):
        model_loader = ModelLoader(self.config)
        model, tokenizer = model_loader.load_model_and_tokenizer()

        data_loader = DataLoaderWrapper(tokenizer)
        train_dataset, val_dataset = data_loader.load_data(
            self.dataset_input,
            self.text_column,
            self.label_column,
            self.split_keys
        )

        trainer = TrainerWrapper(model, self.config.training_args, train_dataset, val_dataset)
        trainer.train()
