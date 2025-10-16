import os
import csv
import json
from typing import Dict, Any
from transformers import AutoModelForSequenceClassification
from src.utils.io import get_directory_size_mb as dir_size_mb


def load_metrics_json(path: str) -> Dict[str, Any]:
    metrics_file = os.path.join(path, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_report(dataset_path: str, baseline_dir: str, full_dir: str, lora_dir: str, output_csv: str):
    dataset_size_mb = dir_size_mb(dataset_path) if os.path.exists(dataset_path) else 0.0

    baseline_metrics = {
        "accuracy": 0.5007,
        "f1": 0.4463,
        "train_runtime_s": 0.0,
    }

    full_best = os.path.join(full_dir, "best_model")
    full_metrics = load_metrics_json(full_dir)
    full_train_time = float(full_metrics.get("train_runtime_s", 0.0))
    full_acc = float(full_metrics.get("test_accuracy", 0.0))
    full_f1 = float(full_metrics.get("test_f1", 0.0))
    full_model_size_mb = dir_size_mb(full_best) if os.path.exists(full_best) else 0.0

    lora_best = os.path.join(lora_dir, "best_model")
    lora_metrics = load_metrics_json(lora_dir)
    lora_train_time = float(lora_metrics.get("train_runtime_s", 0.0))
    lora_acc = float(lora_metrics.get("test_accuracy", 0.0))
    lora_f1 = float(lora_metrics.get("test_f1", 0.0))
    lora_model_size_mb = dir_size_mb(lora_best) if os.path.exists(lora_best) else 0.0

    base_model_for_params = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    total_params = sum(p.numel() for p in base_model_for_params.parameters())

    full_trainable_params = int(full_metrics.get("trainable_params", 0))
    lora_trainable_params = int(lora_metrics.get("trainable_params", 0))

    baseline_json = load_metrics_json(baseline_dir)
    if baseline_json:
        baseline_metrics = {
            "accuracy": float(baseline_json.get("test_accuracy", 0.0)),
            "f1": float(baseline_json.get("test_f1", 0.0)),
            "train_runtime_s": float(baseline_json.get("train_runtime_s", 0.0)),
        }

    baseline_results = {
        "Method": "Baseline (Zero-Shot)",
        "Accuracy": f"{baseline_metrics['accuracy']:.4f}",
        "F1 Score": f"{baseline_metrics['f1']:.4f}",
        "Training Time (s)": f"{baseline_metrics['train_runtime_s']:.2f}",
        "Trainable Params": "N/A",
        "Trainable %": "N/A",
        "Saved Model Size (MB)": "N/A",
        "Dataset Size (MB)": f"{dataset_size_mb:.2f}",
    }

    full_results = {
        "Method": "Full Fine-tuning",
        "Accuracy": f"{full_acc:.4f}",
        "F1 Score": f"{full_f1:.4f}",
        "Training Time (s)": f"{full_train_time:.2f}",
        "Trainable Params": f"{full_trainable_params/1e6:.2f}M" if full_trainable_params else "N/A",
        "Trainable %": f"{(100 * full_trainable_params / total_params):.2f}%" if full_trainable_params else "N/A",
        "Saved Model Size (MB)": f"{full_model_size_mb:.2f}",
        "Dataset Size (MB)": f"{dataset_size_mb:.2f}",
    }

    lora_results = {
        "Method": "LoRA Fine-tuning",
        "Accuracy": f"{lora_acc:.4f}",
        "F1 Score": f"{lora_f1:.4f}",
        "Training Time (s)": f"{lora_train_time:.2f}",
        "Trainable Params": f"{lora_trainable_params/1e3:.2f}K" if lora_trainable_params else "N/A",
        "Trainable %": f"{(100 * lora_trainable_params / total_params):.4f}%" if lora_trainable_params else "N/A",
        "Saved Model Size (MB)": f"{lora_model_size_mb:.2f}",
        "Dataset Size (MB)": f"{dataset_size_mb:.2f}",
    }

    results_list = [baseline_results, lora_results, full_results]
    fieldnames = [
        "Method",
        "Accuracy",
        "F1 Score",
        "Training Time (s)",
        "Trainable Params",
        "Trainable %",
        "Saved Model Size (MB)",
        "Dataset Size (MB)",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    return results_list


