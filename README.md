# Adapting DistilBERT for Sentiment Analysis: A Performance and Efficiency Comparison of Full vs. LoRA Fine-tuning
## Author: WUYIKAI

This project is an implementation for "Assignment 3: Fine-tuning Pretrained Transformers". It provides an end-to-end workflow for adapting a pre-trained Transformer to a downstream task, comparing full fine-tuning against a parameter-efficient alternative (LoRA).

## 1. Project Overview

The goal of this project is to explore how large language models can be effectively adapted for specific tasks. We demonstrate this by fine-tuning a model for sentiment analysis on movie reviews.

*   **Task**: Binary Text Classification (Sentiment Analysis on Positive/Negative reviews).
*   **Dataset**: **IMDB Movie Reviews**. For automated download and reproducibility, this project uses the version from the [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset consists of 50,000 movie reviews.
*   **Pre-trained Model**: **`distilbert-base-uncased`**. This is a distilled, lighter, and faster version of BERT, making it ideal for experiments with limited computational resources.
*   **Fine-tuning Strategies Compared**:
    1.  **Baseline (Zero-Shot)**: The original, pre-trained model without any fine-tuning.
    2.  **Full Fine-tuning**: All ~67 million parameters of the model are updated during training.
    3.  **Parameter-Efficient Fine-tuning (PEFT)**: Using **LoRA (Low-Rank Adaptation)**, which freezes the pre-trained model and only updates a tiny fraction of newly added parameters.

## 2. Key Results

The experiments clearly demonstrate the effectiveness of fine-tuning, with both strategies dramatically outperforming the baseline. Notably, LoRA achieves performance nearly identical to full fine-tuning while training only **1.9890%** of the parameters.

| Metric                | Baseline (Zero-Shot) | LoRA Fine-tuning       | Full Fine-tuning       |
| --------------------- | -------------------- | ---------------------- | ---------------------- |
| **Accuracy**          | 49.90%               | 91.92%                 | **92.82%**             |
| **F1 Score**          | 35.62%               | 91.92%                 | **92.80%**             |
| **Trainable Params**  | N/A                  | **1.33172M** (1.9890%) | ~66.96M (100%)         |
| **Training Time**     | N/A                  | **404.56s**            | 434.42s                |
| **Saved Model Size**  | N/A                  | **3.74 MB** (Adapters) | 256.35 MB (Full Model) |

## 3. Technology Stack

*   **Core Framework**: PyTorch
*   **NLP Toolkit**: Hugging Face `transformers`, `datasets`, `evaluate`
*   **PEFT Library**: Hugging Face `peft`
*   **Environment**: Python 3.9

## 4. File Structure & Description

Below is a description of the key files and directories.

| Path                             | Description |
| -------------------------------- | ----------- |
| `data/`                          | (Generated) Tokenized datasets saved by preparation utilities |
| `models/`                        | (Generated) Saved full models and LoRA adapters |
| `requirements.txt`               | Reproducible dependencies |
| `src/data/prepare.py`            | Dataset preparation utilities (`prepare_single_dataset`, `prepare_all_datasets`) |
| `src/training/train.py`          | Unified trainer (`method='full'|'lora'`) |
| `src/evaluation/baseline.py`     | Zero-shot baseline evaluation |
| `src/inference/batch.py`         | Batch inference utilities (compare Full vs LoRA) |
| `src/reporting/generate.py`      | Generate comparison CSV from saved metrics/models |
| `src/pipeline/run_all.py`        | Orchestrates the full pipeline programmatically |
| `src/utils/seed.py`              | Reproducible seeding and device helpers |
| `src/utils/io.py`                | Filesystem helpers (e.g., directory size) |
| `run_all.py`                     | CLI entry that calls `src.pipeline.run_all.run_pipeline` |
| `test_sentences.txt`             | Sample sentences for qualitative checks |

## 5. How to Run

Requires Python 3.9. Install dependencies:

```bash
pip install -r requirements.txt
```

### Quickstart (Recommended)

Run the full pipeline (prepare → baseline → full → LoRA → report → batch inference):

```bash
python run_all.py --dataset_name imdb --model_checkpoint distilbert-base-uncased --seed 42 --fp16 --early_stopping_patience 2
```

Other datasets:

```bash
# Yelp Polarity
python run_all.py --dataset_name yelp_polarity

# Rotten Tomatoes
python run_all.py --dataset_name rotten_tomatoes

# Amazon Polarity
python run_all.py --dataset_name amazon_polarity

# GLUE SST-2
python run_all.py --dataset_name glue --dataset_config sst2
```

### Advanced: Call modules directly

Prepare all datasets:

```bash
python -c "from src.data.prepare import prepare_all_datasets; prepare_all_datasets('distilbert-base-uncased', 0.1, 512, 42)"
```

Train and evaluate on IMDb only:

```bash
# Baseline
python -c "from src.evaluation.baseline import evaluate_baseline as E; E('distilbert-base-uncased','data/tokenized_imdb',32,'models/imdb-baseline')"

# Full fine-tuning
python -c "from src.training.train import train as T; T('data/tokenized_imdb','distilbert-base-uncased','models/imdb-full-finetune',method='full',epochs=3,lr=2e-5,weight_decay=0.01,train_bs=16,eval_bs=16,fp16=True,seed=42,early_stopping_patience=2)"

# LoRA fine-tuning
python -c "from src.training.train import train as T; T('data/tokenized_imdb','distilbert-base-uncased','models/imdb-lora-finetune',method='lora',epochs=3,lr=2e-4,weight_decay=0.01,train_bs=16,eval_bs=16,fp16=True,seed=42,early_stopping_patience=2,lora_r=8,lora_alpha=16,lora_dropout=0.1)"

# Report
python -c "from src.reporting.generate import generate_report as G; G('data/tokenized_imdb','models/imdb-baseline','models/imdb-full-finetune','models/imdb-lora-finetune','comparison_report_final.csv')"

# Batch inference (qualitative)
python -c "from src.inference.batch import run_batch_inference as R; R('models/imdb-full-finetune/best_model','models/imdb-lora-finetune/best_model','distilbert-base-uncased','test_sentences.txt','inference_results_from_file.csv')"
```

Artifacts:
- `models/**/metrics.json` holds training time and test metrics
- `comparison_report_final.csv` and `inference_results_from_file.csv` are generated at repo root
