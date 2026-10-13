"""
Tokenizer Ablation — BBPE vs BPE vs Unigram vs TED–ASR correlation

Measures the relationship between tokenizer type and SWP effectiveness:
  - BBPE (Qwen-7B, 152K vocab) — finest granularity, most boundary disruption points
  - BPE (Llama2-7B, Gemma-7B, 32K/256K) — medium granularity
  - Unigram (T5-small, 32K) — coarsest granularity

Hypothesis: finer tokenization → more boundary disruption → higher SWP ASR.

Usage:
    python run_tokenizer_ablation.py --models Qwen-7B,Llama2-7B,Gemma-7B,T5-small --runs 5
"""

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
    GPU_MAPPING,
    RESULTS_DIR,
    L2_THRESHOLD,
    WHITESPACE_POOL,
)
from load_datasets import load_all_datasets
from evaluate import RoBERTaEvaluator, GPTEvaluator, compute_ted, spearman_r, paired_t_test

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def compute_ted_local(original: str, perturbed: str, tokenizer) -> float:
    """
    Compute Token Edit Distance:
    Levenshtein distance on token sequences, normalized by prompt token length.
    See evaluate.compute_ted for the implementation used project-wide.

    Local wrapper for backward compatibility within this script.
    """
    return compute_ted(original, perturbed, tokenizer)


def insert_zero_width(prompt: str, rng: random.Random) -> Tuple[str, str]:
    """Insert a zero-width character at a random intra-word position."""
    ws_chars = WHITESPACE_POOL["zero_width"]
    ws_entry = rng.choice(ws_chars)
    ch = ws_entry["char"]

    # Try to insert at a word boundary
    pos = rng.randint(1, len(prompt) - 1) if len(prompt) > 2 else 0
    return prompt[:pos] + ch + prompt[pos:], ws_entry["code"]


def run_tokenizer_ablation(
    model_names: List[str],
    prompts: List[str],
    roberta_eval: RoBERTaEvaluator,
    gpt_eval: GPTEvaluator,
    seed: int = SEED,
) -> Dict[str, dict]:
    """
    Run TED-ASR correlation experiment for each model/tokenizer.

    Returns {model_name: {ted_values: [...], asr_values: [...], spearman_rho: float}}
    """
    set_seed(seed)
    rng = random.Random(seed)

    results = {}

    for model_name in model_names:
        config = MODELS[model_name]
        gpu_id = GPU_MAPPING.get(model_name, 0)
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading {model_name} ({config['tokenizer_type']}, {config['vocab_size']} vocab)")

        tokenizer = AutoTokenizer.from_pretrained(config["hf_name"])
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_name"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device} if torch.cuda.is_available() else None,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ted_values = []
        asr_values = []
        jb_count = 0

        for i, prompt in enumerate(prompts):
            perturbed, ws_code = insert_zero_width(prompt, rng)

            # Compute TED
            ted = compute_ted_local(prompt, perturbed, tokenizer)

            # Generate response
            inputs = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )

            # Evaluate
            l1_label, _ = roberta_eval.classify(response)
            l2_score = 0
            is_jb = False
            if l1_label == "harmful":
                score_result = gpt_eval.score(perturbed, response)
                l2_score = score_result["score"]
                is_jb = l2_score >= L2_THRESHOLD
                if is_jb:
                    jb_count += 1

            ted_values.append(ted)
            asr_values.append(1 if is_jb else 0)

            if (i + 1) % 50 == 0:
                logger.info(f"  [{model_name}] {i + 1}/{len(prompts)} (jailbreaks: {jb_count})")

        # Compute Spearman correlation (Levenshtein-based TED)
        rho = spearman_r(ted_values, asr_values)

        results[model_name] = {
            "tokenizer_type": config["tokenizer_type"],
            "vocab_size": config["vocab_size"],
            "mean_ted": round(sum(ted_values) / len(ted_values), 2) if ted_values else 0,
            "asr_percent": round(jb_count / len(prompts) * 100, 2) if prompts else 0,
            "ted_values": ted_values,
            "spearman_rho": round(rho, 4),
            "n_samples": len(prompts),
        }

        del model, tokenizer
        torch.cuda.empty_cache()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tokenizer type ablation — TED vs ASR")
    parser.add_argument("--models", type=str,
                        default="Qwen-7B,Llama2-7B,Gemma-7B,T5-small",
                        help="Comma-separated model names")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]

    output_dir = args.output or os.path.join(RESULTS_DIR, "tokenizer_ablation")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TOKENIZER ABLATION START")
    logger.info(f"Models: {model_names}")
    logger.info(f"Runs: {args.runs} | Seed: {args.seed}")
    logger.info("=" * 60)

    roberta_eval = RoBERTaEvaluator()
    gpt_eval = GPTEvaluator()
    # Paper Table 5: tokenizer ablation is averaged across all eight datasets.
    datasets = load_all_datasets()
    prompts = [p for ds in datasets.values() for p in ds]

    all_results = {}

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        logger.info(f"\n{'='*10} Run {run_idx + 1}/{args.runs} {'='*10}")

        run_result = run_tokenizer_ablation(
            model_names, prompts, roberta_eval, gpt_eval, seed=run_seed,
        )

        for model_name, stats in run_result.items():
            if model_name not in all_results:
                all_results[model_name] = []
            all_results[model_name].append(stats)

    # Aggregate
    agg = {}
    for model_name, stats_list in all_results.items():
        if not stats_list:
            continue
        agg[model_name] = {
            "tokenizer_type": stats_list[0]["tokenizer_type"],
            "vocab_size": stats_list[0]["vocab_size"],
            "mean_asr": round(sum(s["asr_percent"] for s in stats_list) / len(stats_list), 2),
            "mean_ted": round(sum(s["mean_ted"] for s in stats_list) / len(stats_list), 2),
            "mean_rho": round(sum(s["spearman_rho"] for s in stats_list) / len(stats_list), 4),
            "n_runs": len(stats_list),
        }

    with open(os.path.join(output_dir, "tokenizer_ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"\nSummary saved to {output_dir}/tokenizer_ablation_summary.json")

    # Print comparison
    logger.info("\nTokenizer type comparison:")
    logger.info(f"{'Model':<15} {'Type':<10} {'Vocab':>8} {'ASR':>8} {'TED':>6} {'rho':>6}")
    logger.info("-" * 60)
    for model_name, stats in sorted(agg.items(), key=lambda x: x[1]["mean_asr"], reverse=True):
        logger.info(
            f"{model_name:<15} {stats['tokenizer_type']:<10} "
            f"{stats['vocab_size']:>8} {stats['mean_asr']:>7.1f}% "
            f"{stats['mean_ted']:>5.1f} {stats['mean_rho']:>6.4f}"
        )


if __name__ == "__main__":
    main()
