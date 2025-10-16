Repository: https://github.com/NUS-Kev1n/dsa-3/tree/main

Author: WUYIKAI

## Assignment 3 Report: Sentiment Classification with DistilBERT (Full FT vs LoRA)

### 1) Data

- **Task**: Binary sentiment classification (positive vs negative).
- **Primary dataset**: IMDb (50k movie reviews) from Hugging Face Hub (mirror via `HF_ENDPOINT=https://hf-mirror.com`).
- **Multi-dataset support (prepared)**: `imdb`, `yelp_polarity`, `rotten_tomatoes`, `amazon_polarity`, `glue/sst2`（auto detection of text/label columns; missing splits auto-created）。
- **Column detection**: Auto-detects text from candidates like `text/sentence/review/content/document/review_body/sentence1` and label from `label/labels/sentiment/target/polarity`. Normalizes to `label`.
- **Splits**: Ensures `train/validation/test` exist. If missing, creates validation by `train_test_split(test_size=0.1, seed=42)`; for test without labels (e.g., SST-2), reuses validation for evaluation.
- **Tokenization**: `distilbert-base-uncased` tokenizer, `max_length=512`, `padding='max_length'`, `truncation=True`.
- **Persistence & size**: Tokenized datasets saved to disk (`data/tokenized_<dataset>`). IMDb tokenized size ≈ 122.86 MB.
- **Throughput**: Data mapping with `num_proc=8`.


### 2) Training Experiment Setup

- **Base model**: `distilbert-base-uncased` with a sequence classification head (`num_labels=2`).
- **Methods**:
  - **Baseline (Zero-Shot)**: No training; direct inference with the pre-trained checkpoint.
  - **Full Fine-tuning**: Update all model parameters.
  - **LoRA Fine-tuning**: Parameter-efficient fine-tuning (PEFT) with adapters (targets: `q_lin`, `v_lin`).
- **LoRA configuration**: `r=8`, `alpha=16`, `dropout=0.1`.
- **Optimization & schedules** (Trainer):
  - Epochs: `3`
  - Learning rate: Full `2e-5`; LoRA `2e-4`
  - Weight decay: `0.01`
  - Evaluation/save strategy: `epoch`; `load_best_model_at_end=True`; `metric_for_best_model='accuracy'`
  - Early stopping: patience `=2`
  - Mixed precision: `--fp16` (optional)
- **Batching (24GB GPU defaults)**:
  - Full FT: `per_device_train_batch_size=64`, `per_device_eval_batch_size=64`
  - LoRA FT: `per_device_train_batch_size=128`, `per_device_eval_batch_size=64`
- **Parallelism & reproducibility**:
  - Dataloaders: `dataloader_num_workers=8`
  - Baseline DataLoader: `num_workers=8`
  - Global seed: `42`
- **Metrics**: Accuracy, F1 (stored with runtime and parameter counts). Artifacts in `models/**/metrics.json`.
- **Pipeline orchestration**: `a/run_all.py` executes: prepare → baseline → full FT → LoRA FT → report → batch inference. Mirror endpoint set via `--hf_endpoint`.


### 3) Results

#### 3.1 Quantitative (IMDb)

| Method | Accuracy | F1 Score | Training Time (s) | Trainable Params | Trainable % | Saved Model Size (MB) | Dataset Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (Zero-Shot) | 0.4990 | 0.3562 | 0.00 | N/A | N/A | N/A | 122.86 |
| LoRA Fine-tuning | 0.9192 | 0.9192 | 404.56 | 1.33172M | 1.9890% | 3.74 | 122.86 |
| Full Fine-tuning | 0.9282 | 0.9280 | 434.42 | 66.96M | 100.00% | 256.35 | 122.86 |

Observations:
- Both fine-tuning strategies dramatically outperform the zero-shot baseline on IMDb.
- Full FT is best by a small margin (~0.9 percentage points), but LoRA is near-parity.
- LoRA trains ~2% of parameters and produces a tiny adapter (3.74 MB), offering major savings in storage and deployment while slightly reducing training time (~7%).

#### 3.2 Qualitative (Batch inference on curated sentences)

- **Strengths**: Clear positive/negative statements and contrastive structures (e.g., “although/while/but”) are handled well by both fine-tuned models.
- **Typical errors**: Sarcasm/irony and highly neutral factual statements (e.g., runtime, budget) remain challenging; occasional false positives are observed. LoRA can be slightly more “optimistic” on sarcastic negatives; full FT also shows sporadic false positives.


### 4) Conclusion

- **Effectiveness**: Fine-tuning DistilBERT on IMDb yields strong performance; LoRA achieves near-parity with full fine-tuning while being far more parameter- and storage-efficient.
- **Efficiency**: With only ~1.99% trainable parameters and a 3.74 MB adapter, LoRA substantially reduces training and deployment costs.
- **Reporting hygiene**: Standardize F1 computation across stages (e.g., weighted F1 throughout) for perfect comparability; consider targeted augmentation or calibration to mitigate sarcasm/neutrality errors.


