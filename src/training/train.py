import os
import json
import evaluate
import numpy as np
from typing import Optional
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


def train(
    tokenized_data_path: str,
    model_checkpoint: str,
    output_dir: str,
    method: str = "full",
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    train_bs: int = 16,
    eval_bs: int = 16,
    fp16: bool = False,
    seed: int = 42,
    early_stopping_patience: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    tokenized = load_from_disk(tokenized_data_path)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]
    test_dataset = tokenized["test"]

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "v_lin"],
        )
        model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=fp16,
        seed=seed,
        dataloader_num_workers=8,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > 0 else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    train_output = trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    final_model_path = f"{output_dir}/best_model"
    trainer.save_model(final_model_path)

    metrics = {
        "method": "LoRA Fine-tuning" if method == "lora" else "Full Fine-tuning",
        "model_checkpoint": model_checkpoint,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": train_bs,
        "per_device_eval_batch_size": eval_bs,
        "fp16": fp16,
        "early_stopping_patience": early_stopping_patience,
        "lora_r": lora_r if method == "lora" else None,
        "lora_alpha": lora_alpha if method == "lora" else None,
        "lora_dropout": lora_dropout if method == "lora" else None,
        "train_runtime_s": float(train_output.metrics.get("train_runtime", 0.0)),
        "test_accuracy": float(test_results.get("eval_accuracy", 0.0)),
        "test_f1": float(test_results.get("eval_f1", 0.0)),
    }

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    metrics.update({
        "trainable_params": int(trainable_params),
        "all_params": int(all_params),
        "trainable_percent": (100.0 * trainable_params / all_params) if all_params else 0.0,
    })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return final_model_path, metrics


