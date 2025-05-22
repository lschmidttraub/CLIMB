from datasets import load_dataset, Dataset


def stream_samples(ds_name, num_samples, buffer_size):
    assert num_samples <= buffer_size
    ds_stream = load_dataset(ds_name, split="train", streaming=True)

    buffer = ds_stream.take(buffer_size)

    ds = Dataset.from_list(list(buffer), features=ds_stream.features)

    ds_select = ds.shuffle(seed=42).select(range(num_samples))
    return ds_select


if __name__ == "__main__":
    smiles_ds = stream_samples("yzimmermann/smiles-dump", 100_000, 1_000_000)
    smiles_ds.save_to_disk("datasets/smiles_100k.hf")
