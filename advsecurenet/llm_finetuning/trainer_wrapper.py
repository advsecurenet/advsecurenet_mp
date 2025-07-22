from transformers import Trainer
from sklearn.metrics import accuracy_score

class TrainerWrapper:
    def __init__(self, model, args, train_dataset, eval_dataset):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        trainer.evaluate()
