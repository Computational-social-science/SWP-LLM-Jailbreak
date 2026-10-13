"""
Ablation Study — Insertion Position × Character Type

4 insertion positions × 3 character types = 12 conditions.
Measures ASR per combination to identify the optimal perturbation strategy.

Positions: intra_word, inter_word, sent_start, sent_end
Types:     regular, non_breaking, zero_width

Usage:
    python run_ablation.py --positions intra_word,inter_word,sent_start,sent_end \
        --types regular,non_breaking,zero_width --runs 5
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
    GPU_MAPPING,
    RESULTS_DIR,
    L2_THRESHOLD,
    WHITESPACE_POOL,
    INSERTION_POSITIONS,
    CHARACTER_TYPES,
)
from load_datasets import load_all_datasets
from evaluate import RoBERTaEvaluator, GPTEvaluator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def get_ws_chars_by_type(ws_type: str) -> List[dict]:
    """Get all whitespace characters of a given category."""
    return WHITESPACE_POOL.get(ws_type, [])


def insert_at_position(prompt: str, ws_char: dict, position: str) -> Tuple[str, int]:
    """
    Insert whitespace character at the specified position type.
    Returns (perturbed_text, actual_position_index).
    """
    ch = ws_char["char"]

    if position == "sent_start":
        return ch + prompt, 0
    elif position == "sent_end":
        return prompt + ch, len(prompt)
    elif position == "intra_word":
        # Find a non-space position inside a word
        words = prompt.split()
        if not words:
            return ch + prompt, 0
        # Pick a random word and insert in the middle
        target_word = random.choice(words)
        # Find the word in the original text
        idx = prompt.find(target_word)
        if idx == -1:
            return prompt + ch, len(prompt)
        # Insert at midpoint of this word
        insert_pos = idx + max(1, len(target_word) // 2)
        return prompt[:insert_pos] + ch + prompt[insert_pos:], insert_pos
    elif position == "inter_word":
        # Find a space position
        spaces = [i for i, c in enumerate(prompt) if c == " "]
        if not spaces:
            return prompt + ch, len(prompt)
        insert_pos = random.choice(spaces)
        return prompt[:insert_pos] + ch + prompt[insert_pos:], insert_pos
    else:
        # Random position
        pos = random.randint(0, len(prompt))
        return prompt[:pos] + ch + prompt[pos:], pos


def run_ablation_config(
    model_name: str,
    prompts: List[str],
    roberta_eval: RoBERTaEvaluator,
    gpt_eval: GPTEvaluator,
    positions: List[str],
    ws_types: List[str],
    seed: int = SEED,
) -> Dict[Tuple[str, str], dict]:
    """
    Run ablation: each (position × type) combination on the full prompt set.

    Returns dict: {(position, ws_type): stats_dict}
    """
    set_seed(seed)
    rng = random.Random(seed)

    model_config = MODELS[model_name]
    gpu_id = GPU_MAPPING.get(model_name, 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_config["hf_name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_config["hf_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device} if torch.cuda.is_available() else None,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for position in positions:
        for ws_type in ws_types:
            logger.info(f"  [{position}] × [{ws_type}]")
            ws_chars = get_ws_chars_by_type(ws_type)
            if not ws_chars:
                logger.warning(f"    No chars for type {ws_type}; skipping")
                continue

            jb_count = 0
            total = len(prompts)

            for i, prompt in enumerate(prompts):
                ws_entry = rng.choice(ws_chars)
                perturbed, pos_idx = insert_at_position(prompt, ws_entry, position)

                inputs = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=256, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                )

                l1_label, _ = roberta_eval.classify(response)
                if l1_label == "harmful":
                    score_result = gpt_eval.score(perturbed, response)
                    l2_score = score_result["score"]
                    if l2_score >= L2_THRESHOLD:
                        jb_count += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"    [{position}][{ws_type}] {i + 1}/{total}")

            asr = round(jb_count / total * 100, 2) if total > 0 else 0.0
            all_results[(position, ws_type)] = {
                "position": position,
                "ws_type": ws_type,
                "total": total,
                "jailbreak_count": jb_count,
                "asr_percent": asr,
            }
            logger.info(f"    ASR={asr}% ({jb_count}/{total})")

    del model, tokenizer
    torch.cuda.empty_cache()

    return all_results


def aggregate_ablation_results(
    all_run_results: List[Dict],
    positions: List[str],
    ws_types: List[str],
) -> dict:
    """Aggregate ablation results across multiple runs."""
    agg = {"by_position": {}, "by_type": {}, "by_combination": {}}

    # Collect values
    pos_values = {p: [] for p in positions}
    type_values = {t: [] for t in ws_types}
    combo_values = {}

    for run_result in all_run_results:
        for (pos, wst), stats in run_result.items():
            asr = stats["asr_percent"]
            pos_values[pos].append(asr)
            type_values[wst].append(asr)
            combo_values[(pos, wst)] = combo_values.get((pos, wst), []) + [asr]

    # Compute means
    for pos in positions:
        if pos_values[pos]:
            agg["by_position"][pos] = round(sum(pos_values[pos]) / len(pos_values[pos]), 1)

    for wst in ws_types:
        if type_values[wst]:
            agg["by_type"][wst] = round(sum(type_values[wst]) / len(type_values[wst]), 1)

    for (pos, wst), vals in combo_values.items():
        if vals:
            agg["by_combination"][f"{pos}×{wst}"] = round(sum(vals) / len(vals), 1)

    return agg


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ablation: Position × Character Type")
    parser.add_argument("--positions", type=str,
                        default=",".join(INSERTION_POSITIONS),
                        help="Comma-separated positions")
    parser.add_argument("--types", type=str,
                        default=",".join(CHARACTER_TYPES),
                        help="Comma-separated character types")
    parser.add_argument("--model", type=str, default="all", help="Model(s) or 'all'")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    positions = [p.strip() for p in args.positions.split(",")]
    ws_types = [t.strip() for t in args.types.split(",")]

    output_dir = args.output or os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ABLATION STUDY START")
    logger.info(f"Model: {args.model}")
    logger.info(f"Positions: {positions}")
    logger.info(f"Types: {ws_types}")
    logger.info(f"Runs: {args.runs} | Seed: {args.seed}")
    logger.info("=" * 60)

    roberta_eval = RoBERTaEvaluator()
    gpt_eval = GPTEvaluator()

    if args.model == "all":
        model_names = list(MODELS.keys())
    else:
        model_names = [m.strip() for m in args.model.split(",")]

    # Paper Table 5: position/type ablation is averaged across all eight datasets.
    datasets = load_all_datasets()
    prompts = [p for ds in datasets.values() for p in ds]

    all_run_results = {}

    for model_name in model_names:
        logger.info(f"\n{'='*10} Model: {model_name} {'='*10}")
        model_results = []

        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx
            logger.info(f"  Run {run_idx + 1}/{args.runs}")

            run_result = run_ablation_config(
                model_name, prompts, roberta_eval, gpt_eval,
                positions, ws_types, seed=run_seed,
            )
            model_results.append(run_result)

        all_run_results[model_name] = model_results

    # Aggregate per model
    agg = {}
    for model_name, model_results in all_run_results.items():
        agg[model_name] = aggregate_ablation_results(model_results, positions, ws_types)

    with open(os.path.join(output_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"\nAblation summary saved to {output_dir}/ablation_summary.json")

    # Print ranking (pooled across all models)
    pooled_pos = {p: [] for p in positions}
    pooled_type = {t: [] for t in ws_types}
    for model_name, result in agg.items():
        for pos, asr in result.get("by_position", {}).items():
            pooled_pos[pos].append(asr)
        for wst, asr in result.get("by_type", {}).items():
            pooled_type[wst].append(asr)

    logger.info("\nPosition ranking (pooled across models):")
    for pos in sorted(pooled_pos, key=lambda p: sum(pooled_pos[p])/max(1,len(pooled_pos[p])), reverse=True):
        avg = round(sum(pooled_pos[pos]) / max(1, len(pooled_pos[pos])), 1)
        logger.info(f"  {pos}: {avg}%")

    logger.info("\nCharacter type ranking (pooled across models):")
    for wst in sorted(pooled_type, key=lambda t: sum(pooled_type[t])/max(1,len(pooled_type[t])), reverse=True):
        avg = round(sum(pooled_type[wst]) / max(1, len(pooled_type[wst])), 1)
        logger.info(f"  {wst}: {avg}%")


if __name__ == "__main__":
    main()
