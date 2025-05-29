import wandb
from config import WANDB_API_KEY, HF_NOTEBOOK_KEY
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
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
from model import Task, CLIMBModel

wandb.login(key=WANDB_API_KEY)


def train_pipeline():
    run = wandb.init()
    cfg = wandb.config

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer)

    def tokenize_function(data):
        return tokenizer(data["smiles"], truncation=True, padding=False)

    bert_model_config = BertConfig.from_pretrained(
        cfg.get("base_model_hf_name", "distilbert-base-uncased")
    )
    bert_model_config.vocab_size = tokenizer.vocab_size
    bert_model_config.pad_token_id = tokenizer.pad_token_id

    bert_model_config.hidden_size = cfg.hidden_size
    bert_model_config.num_hidden_layers = cfg.num_hidden_layers

    print("Effective BertConfig:", bert_model_config)

    # --- Instantiate CLIMBModel ---
    # Use wandb.run.name for unique output paths per sweep run
    experiment_run_name = (
        wandb.run.name
        if wandb.run
        else f"local_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    climb_model_instance = CLIMBModel(
        model_name=cfg.name,
        config=bert_model_config,
        tokenizer=tokenizer,
        tokenize_function=tokenize_function,
    )
