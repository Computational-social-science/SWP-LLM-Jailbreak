"""
Original Condition — Papillon Baseline (no whitespace insertion)

Runs all 5 models on all 8 datasets on the "Original" condition of the paper:
the benchmark prompt passed through the Papillon baseline (Algorithm 1's
Q_base = Framework(Q_original)), with no whitespace inserted. This establishes
the reference ASR for each model-dataset pair.

Usage:
    python run_baseline.py --models all --datasets all --runs 5 --seed 42
"""

import csv
import json
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    SEED,
    MODELS,
    DATASETS,
    GPU_MAPPING,
    RESULTS_DIR,
    L2_THRESHOLD,
)
from load_datasets import load_all_datasets
from papillon import papillon_baseline
from evaluate import RoBERTaEvaluator, GPTEvaluator, paired_t_test

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str) -> Tuple:
    """Load model and tokenizer. Returns (model, tokenizer, device)."""
    config = MODELS[model_name]
    gpu_id = GPU_MAPPING.get(model_name, 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {model_name} ({config['tokenizer_type']}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config["hf_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["hf_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device} if torch.cuda.is_available() else None,
    )
    model.eval()

    # Set pad_token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate model response for a prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def run_baseline_config(
    model_name: str,
    dataset_name: str,
    prompts: List[str],
    roberta_eval: RoBERTaEvaluator,
    gpt_eval: Optional[GPTEvaluator] = None,
    seed: int = SEED,
) -> List[dict]:
    """
    Run baseline (no perturbation) for one model-dataset pair.

    Returns list of per-sample result dicts with L1 + L2 scores.
    """
    set_seed(seed)
    model, tokenizer, device = load_model(model_name)

    results = []
    for i, prompt in enumerate(prompts):
        # Algorithm 1: Q_base = Framework(Q_original). "Original" is the Papillon
        # baseline (no whitespace inserted), i.e. the prompt actually sent.
        base_prompt = papillon_baseline(prompt)
        response = generate_response(model, tokenizer, base_prompt)
        l1_label, l1_conf = roberta_eval.classify(response)

        l2_score = 0
        is_jb = False
        if l1_label == "harmful" and gpt_eval is not None:
            l2_result = gpt_eval.score(base_prompt, response)
            l2_score = l2_result["score"]
            is_jb = l2_score >= L2_THRESHOLD

        results.append({
            "prompt_id": i,
            "dataset": dataset_name,
            "model": model_name,
            "raw_prompt": prompt,
            "prompt": base_prompt,
            "response": response[:500],
            "L1_label": l1_label,
            "L1_confidence": round(l1_conf, 4),
            "L2_score": l2_score,
            "is_jailbreak": int(is_jb),
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  [{model_name}][{dataset_name}] {i + 1}/{len(prompts)}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def compute_asr(results: List[dict]) -> dict:
    """
    Compute ASR (fraction where L1=harmful AND L2 >= threshold).
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "asr_percent": 0.0, "jailbreak_count": 0}

    jb_count = sum(1 for r in results if r.get("is_jailbreak") == 1)
    asr = round(jb_count / total * 100, 2)

    return {
        "total": total,
        "jailbreak_count": jb_count,
        "asr_percent": asr,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline (no perturbation) experiment")
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model names or 'all' (default: all)",
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="Comma-separated dataset names or 'all' (default: all)",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/baseline/)",
    )
    args = parser.parse_args()

    # Parse model/dataset selection
    if args.models == "all":
        model_names = list(MODELS.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    if args.datasets == "all":
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

    # Setup output
    output_dir = args.output or os.path.join(RESULTS_DIR, "baseline")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BASELINE EXPERIMENT START")
    logger.info(f"Models: {model_names}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"Runs: {args.runs} | Seed: {args.seed}")
    logger.info("=" * 60)

    # Load datasets
    datasets = load_all_datasets(dataset_names)

    # Initialize evaluators (shared across runs)
    roberta_eval = RoBERTaEvaluator()
    gpt_eval = GPTEvaluator()

    # Aggregate summary
    all_summaries = {}

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        logger.info(f"\n{'='*40} Run {run_idx + 1}/{args.runs} {'='*40}")

        for model_name in model_names:
            for ds_name in dataset_names:
                if ds_name not in datasets:
                    logger.warning(f"Dataset {ds_name} not loaded; skipping")
                    continue

                prompts = datasets[ds_name]
                logger.info(f"  [{model_name}][{ds_name}] {len(prompts)} prompts")

                t0 = time.time()
                results = run_baseline_config(
                    model_name, ds_name, prompts, roberta_eval, gpt_eval, seed=run_seed,
                )
                elapsed = time.time() - t0

                summary = compute_asr(results)
                logger.info(
                    f"    ASR={summary['asr_percent']}% "
                    f"({summary['jailbreak_count']}/{summary['total']}) "
                    f"[{elapsed:.1f}s]"
                )

                # Save per-run CSV
                csv_path = os.path.join(
                    output_dir,
                    f"run{run_idx + 1}_{model_name}_{ds_name}.csv",
                )
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)

                # Accumulate summary
                key = f"{model_name}|{ds_name}"
                if key not in all_summaries:
                    all_summaries[key] = []
                all_summaries[key].append(summary)

    # Compute aggregate summaries with 95% CI
    agg = {}
    for key, summaries in all_summaries.items():
        asr_values = [s["asr_percent"] for s in summaries]
        n = len(asr_values)
        mean_asr = sum(asr_values) / n
        std_asr = (sum((v - mean_asr) ** 2 for v in asr_values) / (n - 1)) ** 0.5 if n > 1 else 0
        t_crit = 1.96 if n >= 30 else (2.776 if n == 5 else 2.0)
        ci_low = round(mean_asr - t_crit * std_asr / n**0.5, 2)
        ci_high = round(mean_asr + t_crit * std_asr / n**0.5, 2)
        agg[key] = {
            "mean_asr": round(mean_asr, 2),
            "ci_95_asr": f"[{ci_low}, {ci_high}]",
            "n_runs": n,
        }

    # Save summary JSON
    summary_path = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("BASELINE EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
