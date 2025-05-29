from model import CLIMBModel, Task
import wandb
from transformers import (
    BertConfig,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
import datetime
import pandas as pd

if __name__ == "__main__":
    run = wandb.init()
    cfg = wandb.config

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer)

    dataset_cfg = cfg["data_config"]
    dataset = pd.read_csv(dataset_cfg["path"])

    def tokenize_function(data):
        return tokenizer(data[dataset_cfg["features"]], truncation=True, padding=False)

    bert_model_config = BertConfig.from_pretrained(
        cfg.get("base_model_name", "distilbert-base-uncased")
    )

    bert_model_config.vocab_size = tokenizer.vocab_size
    bert_model_config.pad_token_id = tokenizer.pad_token_id

    print("Effective BertConfig:", bert_model_config)

    # --- Instantiate CLIMBModel ---
    # Use wandb.run.name for unique output paths per sweep run
    experiment_run_name = (
        wandb.run.name
        if wandb.run
        else f"local_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    climb_model_instance = CLIMBModel(
        model_name=cfg["name"],
        config=bert_model_config,
        tokenizer=tokenizer,
        tokenize_function=tokenize_function,
    )

    task = Task(dataset_cfg["task_type"], dataset_cfg["num_labels"])
    trainer_args = TrainingArguments(
        output_dir=f"mtr/{cfg['tokenizer']}",
        overwrite_output_dir=True,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=cfg["learning_rate"],
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=5,
        bf16=cfg["bf16"],  # Use bf16 if on compatible hardware
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Metric to identify the best model
        greater_is_better=False,
        report_to="wandb",
    )

    climb_model_instance.mtr(dataset=dataset, trainer_args=trainer_args, task=task)
