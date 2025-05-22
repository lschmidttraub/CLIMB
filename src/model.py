import wandb
from config import WANDB_API_KEY, SEED
import numpy as np
import torch
from datasets import load_dataset
from zoneinfo import ZoneInfo
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"


class Task:
    def __init__(self, type, num_classes=1):
        if type not in ["classification", "regression"]:
            raise ValueError(
                "Invalid task type. Must be 'classification' or 'regression'."
            )
        if type == "classification" and num_classes == 1:
            raise ValueError(
                "Invalid number of classes. Must be at least 2 for classification tasks."
            )
        if type == "regression" and num_classes != 1:
            raise ValueError("Do not pass num_classes for regression tasks")
        self.type = type
        self.num_classes = num_classes


class CLIMBModel:
    def __init__(self, model_name, config, tokenizer, tokenize_function):
        self.model_name = model_name
        self.config = config
        self.tokenizer = tokenizer
        self.tokenize_function = tokenize_function
        self.mlm_model_path = f"mlm/{model_name}"
        self.init_mlm()

    def init_mlm(self):
        self.model = BertForMaskedLM.from_pretrained(
            self.model_name, config=self.config
        )

    def configure_mtr(self, task):
        self.model = BertForSequenceClassification.from_pretrained(
            f"{self.mlm_model_path}/best",
            num_classes=task.num_classes,
            ignore_mismatched_sizes=True,
        )

    def mlm(self, trainer_args, dataset, mlm_probability, tokenized=True):
        """
        Trains model on Masked Language Modeling (filling in gaps)
        """
        if not tokenized:
            dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=1000,
            )

        split = dataset.train_test_split(test_size=0.1, seed=42)
        trainer_args.output_dir = self.mlm_model_path
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = (predictions == labels).mean()
            if wandb.run:
                wandb.log({"accuracy": accuracy})
            return {"accuracy": accuracy}

        trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        now = datetime.datetime.now(tz=ZoneInfo("Europe/Berlin")).strftime(
            "%Y-%m-%d_%H-%M"
        )
        run = wandb.init(
            entity="chemical_language_model",
            project="CLIMB",
            name=f"{self.model_name}_mlm_{now}",
            config=self.config,
        )

        trainer.train()
        trainer.save_model(f"{self.mlm_model_path}/best")
        wandb.finish()
        return trainer

    def mtr(self, trainer_args, task, dataset, tokenized=True):
        """
        Trains entire model weights on regression/classification
        """
        if not tokenized:
            dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=128,
            )
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        self.configure_mtr(task)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            res = {}
            if task.type == "classification":
                predictions = np.argmax(logits, axis=-1)
                res = {"accuracy": (predictions == labels).mean()}
            elif task.type == "regression":
                res = {"mse": ((logits - labels) ** 2).mean().item()}
            if wandb.run:
                wandb.log(res)
            return res

        trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        now = datetime.datetime.now(tz=ZoneInfo("Europe/Berlin")).strftime(
            "%Y-%m-%d_%H-%M"
        )
        run = wandb.init(
            entity="chemical_language_model",
            project="CLIMB",
            name=f"{self.model_name}_mtr_{now}",
            config=self.config,
        )

        trainer.train()
        trainer.save_model(f"{trainer_args.output_dir}/best")
        wandb.finish()
        return trainer

    def push(self, model_name):
        self.model.push_to_hub(model_name)


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    regex_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "yzimmermann/REGEX-PubChem"
    )
    bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("leosct/smiles-bpe")

    ds = load_dataset("datasets/smiles-100k.hf")
    ds_select = ds.shuffle(seed=SEED).select(range(10_000))

    def tokenize_function(examples):
        return regex_tokenizer(examples["SMILES"], truncation=True, padding=False)

    tokenized_dataset = ds_select.map(
        tokenize_function,
        batched=True,  # Process data in batches
        batch_size=1000,  # Adjust batch size as needed
    )

    # Define the configuration
    config = BertConfig.from_pretrained("distilbert-base-uncased")

    config.vocab_size = regex_tokenizer.vocab_size
    config.pad_token_id = regex_tokenizer.pad_token_id
    config.hidden_size = 384
    print(config)

    distilBert = CLIMBModel(
        "distilbert-base-uncased", config, regex_tokenizer, tokenize_function
    )

    lr = 1e-4
    bs = 32
    epochs = 5

    training_args = TrainingArguments(
        output_dir="mlm_pretrain",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=lr,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=5,
        bf16=True,  # Use bf16 if on compatible hardware
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Metric to identify the best model
        greater_is_better=True,
        report_to="wandb",
    )

    distilBert.mlm(training_args, tokenized_dataset, 0.3)
