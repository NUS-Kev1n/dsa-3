import argparse
import subprocess
import sys
import os


def run_cmd(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: prepare -> baseline -> full -> lora -> reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com", help="Hugging Face mirror endpoint")
    parser.add_argument("--train_bs_full", type=int, default=64)
    parser.add_argument("--train_bs_lora", type=int, default=128)
    parser.add_argument("--eval_bs", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=8)
    args = parser.parse_args()

    # Use mirror endpoint for Hugging Face Hub to avoid network restrictions
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # Delegate to src pipeline
    run_cmd([sys.executable, "-c",
             "from src.pipeline.run_all import run_pipeline as P;"
             f"P('{args.dataset_name}',{repr(args.dataset_config) if args.dataset_config else 'None'},'{args.model_checkpoint}',{args.seed},{'True' if args.fp16 else 'False'},{args.early_stopping_patience},train_bs_full={args.train_bs_full},train_bs_lora={args.train_bs_lora},eval_bs={args.eval_bs},lora_r={args.lora_r})"])

    print("\n✅ All stages completed successfully.")


if __name__ == "__main__":
    main()


