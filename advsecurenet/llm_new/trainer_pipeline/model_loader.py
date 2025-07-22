from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if self.config.model_obj is not None:
            model = self.config.model_obj
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )

        return model, tokenizer
