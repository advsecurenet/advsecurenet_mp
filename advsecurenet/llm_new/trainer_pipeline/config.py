from transformers import TrainingArguments

class Config:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        model_obj=None,
        num_labels: int = 2,
        output_dir: str = "./results"
    ):
        self.model_name = model_name
        self.model_obj = model_obj  # Optional model object, if provided
        self.num_labels = num_labels
        self.output_dir = output_dir

        self.epochs = 3
        self.batch_size = 16
        self.learning_rate = 2e-5

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
        )
