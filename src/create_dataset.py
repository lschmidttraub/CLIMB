import os
import sys
import pandas as pd
from datasets import load_dataset, Dataset

ds_stream = load_dataset("yzimmermann/smiles-dump", split="train", streaming=True)
ds = ds_stream.take(100_000)
ds = Dataset.from_list(list(ds), features=ds_stream.features)

ds_select = ds.shuffle(seed=42).select(range(10_000))
