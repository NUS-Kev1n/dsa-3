import os
from src.data.prepare import prepare_all_datasets, prepare_single_dataset, default_output_dir
from src.evaluation.baseline import evaluate_baseline
from src.training.train import train as train_model
from src.reporting.generate import generate_report
from src.inference.batch import run_batch_inference


def run_pipeline(dataset_name: str, dataset_config: str | None, model_checkpoint: str, seed: int, fp16: bool, patience: int, train_bs_full: int = 64, train_bs_lora: int = 128, eval_bs: int = 64, lora_r: int = 8):
    if dataset_name == "all":
        prepared = prepare_all_datasets(model_checkpoint=model_checkpoint, val_size=0.1, max_length=512, seed=seed)
        # pick imdb for demonstration in the pipeline
        tokenized_dir = default_output_dir("imdb", None)
    else:
        out_dir, _ = prepare_single_dataset(dataset_name, dataset_config, model_checkpoint, val_size=0.1, max_length=512, seed=seed)
        tokenized_dir = out_dir

    key = dataset_name if not dataset_config else f"{dataset_name}_{dataset_config}"
    full_dir = os.path.join("models", f"{key}-full-finetune")
    lora_dir = os.path.join("models", f"{key}-lora-finetune")
    baseline_dir = os.path.join("models", f"{key}-baseline")

    evaluate_baseline(model_checkpoint, tokenized_dir, batch_size=32, output_dir=baseline_dir)
    train_model(tokenized_dir, model_checkpoint, full_dir, method='full', epochs=3, lr=2e-5, weight_decay=0.01, train_bs=train_bs_full, eval_bs=eval_bs, fp16=fp16, seed=seed, early_stopping_patience=patience)
    train_model(tokenized_dir, model_checkpoint, lora_dir, method='lora', epochs=3, lr=2e-4, weight_decay=0.01, train_bs=train_bs_lora, eval_bs=eval_bs, fp16=fp16, seed=seed, early_stopping_patience=patience, lora_r=lora_r, lora_alpha=16, lora_dropout=0.1)

    generate_report(tokenized_dir, baseline_dir, full_dir, lora_dir, "comparison_report_final.csv")
    run_batch_inference(os.path.join(full_dir, "best_model"), os.path.join(lora_dir, "best_model"), model_checkpoint, "test_sentences.txt", "inference_results_from_file.csv")


