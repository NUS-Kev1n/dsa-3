from typing import Optional, Tuple, List
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from src.utils.io import get_directory_size_mb


def detect_columns(train_columns: List[str], text_column: Optional[str], label_column: Optional[str]) -> Tuple[str, str]:
    candidate_text_cols = [
        "text", "sentence", "review", "content", "document", "review_body",
        "sentence1"
    ]
    candidate_label_cols = ["label", "labels", "sentiment", "target", "polarity"]
    t_col = text_column or next((c for c in candidate_text_cols if c in train_columns), None)
    l_col = label_column or next((c for c in candidate_label_cols if c in train_columns), None)
    if t_col is None or l_col is None:
        raise ValueError(f"Could not auto-detect text/label columns. Train columns: {train_columns}")
    return t_col, l_col


def tokenize_dataset(raw_datasets, model_checkpoint: str, text_col: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        if text_col in examples:
            texts = examples[text_col]
        else:
            titles = examples.get("title", [""] * len(examples.get("content", [])))
            contents = examples.get("content", [])
            texts = [f"{t}. {c}".strip() for t, c in zip(titles, contents)]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

    return raw_datasets.map(tokenize_function, batched=True, remove_columns=None, num_proc=8)


def normalize_and_prune(tokenized_datasets, label_col: str) -> DatasetDict:
    def fix(split: str):
        ds = tokenized_datasets[split]
        if label_col != "label" and label_col in ds.column_names:
            ds = ds.rename_column(label_col, "label")
        required_cols = {"input_ids", "attention_mask", "token_type_ids", "label"}
        drop_cols = [c for c in ds.column_names if c not in required_cols]
        return ds.remove_columns(drop_cols)

    return DatasetDict({
        "train": fix("train"),
        "validation": fix("validation"),
        "test": fix("test"),
    })


def default_output_dir(dataset_name: str, dataset_config: Optional[str]) -> str:
    dataset_key = dataset_name if not dataset_config else f"{dataset_name}_{dataset_config}"
    return os.path.join("data", f"tokenized_{dataset_key}")


def prepare_single_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    model_checkpoint: str,
    val_size: float,
    max_length: int,
    seed: int,
    output_dir: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
):
    if dataset_config:
        raw = load_dataset(dataset_name, dataset_config)
    else:
        raw = load_dataset(dataset_name)

    if "train" not in raw:
        raise ValueError("Dataset must contain a 'train' split.")
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=val_size, seed=seed)
        raw["train"], raw["validation"] = split["train"], split["test"]
    if "test" not in raw:
        raw["test"] = raw["validation"]

    train_columns = raw["train"].column_names
    text_col, label_col = detect_columns(train_columns, text_column, label_column)

    test_columns = raw["test"].column_names
    if label_col not in test_columns:
        raw["test"] = raw["validation"]

    tokenized = tokenize_dataset(raw, model_checkpoint, text_col, max_length)
    tokenized_dd = normalize_and_prune(tokenized, label_col)

    out_dir = output_dir or default_output_dir(dataset_name, dataset_config)
    os.makedirs(out_dir, exist_ok=True)
    DatasetDict(tokenized_dd).save_to_disk(out_dir)
    size_mb = get_directory_size_mb(out_dir)
    return out_dir, size_mb


def prepare_all_datasets(
    model_checkpoint: str,
    val_size: float,
    max_length: int,
    seed: int,
):
    datasets_to_prepare = [
        ("imdb", None),
        ("yelp_polarity", None),
        ("rotten_tomatoes", None),
        ("amazon_polarity", None),
        ("glue", "sst2"),
    ]
    results = []
    for ds_name, ds_config in datasets_to_prepare:
        out_dir, size_mb = prepare_single_dataset(
            dataset_name=ds_name,
            dataset_config=ds_config,
            model_checkpoint=model_checkpoint,
            val_size=val_size,
            max_length=max_length,
            seed=seed,
        )
        results.append((ds_name, ds_config, out_dir, size_mb))
    return results


