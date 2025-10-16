import os
import json
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate


def evaluate_baseline(
    base_model: str,
    tokenized_data_path: str,
    batch_size: int = 32,
    output_dir: str = "models/baseline",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_datasets = load_from_disk(tokenized_data_path)
    test_dataset = tokenized_datasets["test"]

    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model.to(device)
    model.eval()

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["label"]
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_metric.add_batch(predictions=predictions, references=labels)

    final_accuracy = accuracy_metric.compute()
    final_f1 = f1_metric.compute(average="weighted")

    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        "method": "Baseline (Zero-Shot)",
        "base_model": base_model,
        "batch_size": batch_size,
        "test_accuracy": float(final_accuracy["accuracy"]),
        "test_f1": float(final_f1["f1"]),
        "train_runtime_s": 0.0,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


