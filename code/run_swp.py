"""
SWP Main Experiment — Single Whitespace Perturbation Attack

Core jailbreaking pipeline:
1. Load dataset prompts
2. Insert a single Unicode whitespace character at a random position
3. Send perturbed prompt to each attack model
4. Collect responses
5. Two-level evaluation: L1 RoBERTa + L2 GPT-3.5

Usage:
    python run_swp.py --models all --datasets all --runs 5 --seed 42
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
    WHITESPACE_POOL,
    N_RUNS,
)
from load_datasets import load_all_datasets
from papillon import papillon_baseline
from evaluate import RoBERTaEvaluator, GPTEvaluator, is_jailbreak

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


# ─── Whitespace injection ───

def select_whitespace_char(rng: random.Random) -> dict:
    """Randomly select one whitespace character from the full pool."""
    all_chars = []
    for category, chars in WHITESPACE_POOL.items():
        for entry in chars:
            all_chars.append({"char": entry["char"], "name": entry["name"],
                              "code": entry["code"], "category": category})
    return rng.choice(all_chars)


def insert_whitespace(prompt: str, ws_entry: dict, position_idx: int) -> Tuple[str, str]:
    """
    Insert a single whitespace character into a prompt at the given index.
    If position_idx >= len(prompt), inserts at end.

    Returns (perturbed_text, position_label).
    """
    ch = ws_entry["char"]
    if position_idx >= len(prompt):
        return prompt + ch, "sent_end"
    elif position_idx == 0:
        return ch + prompt, "sent_start"
    else:
        # Determine position type
        pos_label = "intra_word"
        if prompt[position_idx - 1].isspace() or (position_idx < len(prompt) and prompt[position_idx].isspace()):
            pos_label = "inter_word"
        return prompt[:position_idx] + ch + prompt[position_idx:], pos_label


# ─── Model loading ───

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate model response for a perturbed prompt."""
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

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


# ─── Main experiment function ───

def run_swp_config(
    model_name: str,
    dataset_name: str,
    prompts: List[str],
    roberta_eval: RoBERTaEvaluator,
    gpt_eval: GPTEvaluator,
    seed: int = SEED,
) -> List[dict]:
    """
    Run SWP perturbation for one model-dataset pair.

    For each prompt:
    1. Randomly insert one whitespace character
    2. Send to model, collect response
    3. L1 RoBERTa + L2 GPT-3.5 evaluation

    Returns list of per-sample result dicts.
    """
    set_seed(seed)
    rng = random.Random(seed)
    model, tokenizer, device = load_model(model_name)

    results = []
    jailbreak_count = 0

    for i, prompt in enumerate(prompts):
        # Algorithm 1: Q_base = Framework(Q_original); the "Ours"/SWP condition
        # inserts one whitespace on top of the Papillon baseline.
        base_prompt = papillon_baseline(prompt)

        # Step 1: Select and insert whitespace
        ws_entry = select_whitespace_char(rng)
        pos_idx = rng.randint(0, len(base_prompt))
        perturbed_text, pos_label = insert_whitespace(base_prompt, ws_entry, pos_idx)

        # Step 2: Generate response
        response = generate_response(model, tokenizer, perturbed_text)

        # Step 3: Two-level evaluation
        l1_label, l1_conf = roberta_eval.classify(response)
        l2_score = 0
        if l1_label == "harmful":
            l2_result = gpt_eval.score(perturbed_text, response)
            l2_score = l2_result["score"]

        is_jb = l1_label == "harmful" and l2_score >= L2_THRESHOLD
        if is_jb:
            jailbreak_count += 1

        results.append({
            "prompt_id": i,
            "dataset": dataset_name,
            "model": model_name,
            "raw_prompt": prompt,
            "baseline_prompt": base_prompt,
            "perturbed_text": perturbed_text,
            "omega_char": ws_entry["code"],
            "omega_name": ws_entry["name"],
            "omega_category": ws_entry["category"],
            "position": pos_label,
            "pos_idx": pos_idx,
            "response": response[:500],
            "L1_label": l1_label,
            "L1_confidence": round(l1_conf, 4),
            "L2_score": l2_score,
            "is_jailbreak": int(is_jb),
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  [{model_name}][{dataset_name}] {i + 1}/{len(prompts)} "
                        f"(jailbreaks: {jailbreak_count})")

    # Cleanup GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()

    return results


def compute_summary(results: List[dict]) -> dict:
    """Compute ASR from per-sample results.

    ASR: fraction of samples where is_jailbreak == True.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "asr_percent": 0.0, "jailbreak_count": 0}

    jailbreak_count = sum(1 for r in results if r["is_jailbreak"])
    asr_pct = round(jailbreak_count / total * 100, 2)

    return {
        "total": total,
        "jailbreak_count": jailbreak_count,
        "asr_percent": asr_pct,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run SWP main experiment")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset names or 'all'")
    parser.add_argument("--runs", type=int, default=N_RUNS, help="Number of independent runs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--threshold", type=int, default=L2_THRESHOLD, help="L2 ASR threshold")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: results/swp/)")
    parser.add_argument("--no-gpt", action="store_true", help="Skip GPT-3.5 evaluation (L1 only)")
    args = parser.parse_args()

    if args.models == "all":
        model_names = list(MODELS.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    if args.datasets == "all":
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

    output_dir = args.output or os.path.join(RESULTS_DIR, "swp")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SWP EXPERIMENT START")
    logger.info(f"Models: {model_names}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"Runs: {args.runs} | Seed: {args.seed} | L2 Threshold: {args.threshold}")
    logger.info("=" * 60)

    # Load datasets
    datasets = load_all_datasets(dataset_names)

    # Initialize evaluators (shared)
    roberta_eval = RoBERTaEvaluator()
    gpt_eval = GPTEvaluator() if not args.no_gpt else None

    all_summaries = {}

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        logger.info(f"\n{'='*10} Run {run_idx + 1}/{args.runs} (seed={run_seed}) {'='*10}")

        for model_name in model_names:
            for ds_name in dataset_names:
                if ds_name not in datasets:
                    continue
                prompts = datasets[ds_name]
                logger.info(f"  [{model_name}][{ds_name}] {len(prompts)} prompts")

                t0 = time.time()
                results = run_swp_config(
                    model_name, ds_name, prompts,
                    roberta_eval, gpt_eval, seed=run_seed,
                )
                elapsed = time.time() - t0

                summary = compute_summary(results)
                logger.info(
                    f"    ASR={summary['asr_percent']}% "
                    f"({summary['jailbreak_count']}/{summary['total']}) [{elapsed:.0f}s]"
                )

                # Save per-run CSV
                csv_path = os.path.join(
                    output_dir,
                    f"run{run_idx + 1}_{model_name}_{ds_name}.csv",
                )
                if results:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)

                key = f"{model_name}|{ds_name}"
                if key not in all_summaries:
                    all_summaries[key] = []
                all_summaries[key].append(summary)

    # Aggregate over runs
    agg = {}
    for key, summaries in all_summaries.items():
        if not summaries:
            continue
        agg[key] = {
            "mean_asr": round(sum(s["asr_percent"] for s in summaries) / len(summaries), 2),
            "n_runs": len(summaries),
        }

    summary_path = os.path.join(output_dir, "swp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"\nAggregate summary saved to {summary_path}")
    logger.info("SWP EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
