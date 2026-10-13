"""
Full Ablation Matrix — 36 Conditions (Supplementary)

Systematic sweep across:
  3 fine-tuning states × 4 insertion positions × 3 character types = 36 conditions
  × 10 independent runs = 360 experimental units

States:   θ_S (base), θ_S-GSM8K (gsm8k), θ_S-Alpaca (alpaca)
Positions: intra_word, inter_word, sent_start, sent_end
Types:     zero_width, non_breaking, regular

This is the largest single experiment, confirming that ranking orders
(position and type) are preserved across all conditions.

Usage:
    python run_full_ablation.py --model Llama-3.1-8B --states base,gsm8k,alpaca \
        --positions intra_word,inter_word,sent_start,sent_end \
        --types zero_width,non_breaking,regular --runs 10 --seed 42
"""

import csv
import json
import logging
import os
import random
import time
from itertools import product
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    SEED,
    RESULTS_DIR,
    WHITESPACE_POOL,
    INSERTION_POSITIONS,
    CHARACTER_TYPES,
)
from evaluate import LlamaGuardEvaluator, spearman_r, compute_ted

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Fine-tuning states for full ablation ───
# Override via env vars or --state-paths CLI arg:
#   export SWP_GSM8K_PATH="/path/to/gsm8k-ckpt"
#   python run_full_ablation.py --state-paths '{"gsm8k":"/path/to/ckpt",...}'

FULL_ABLATION_STATES = {
    "base": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Original aligned model",
    },
    "gsm8k": {
        "model_path": os.environ.get(
            "SWP_GSM8K_PATH",
            "path/to/Llama-3.1-8B-Instruct-gsm8k-100step",
        ),
        "description": "Fine-tuned 100 steps on GSM8K",
    },
    "alpaca": {
        "model_path": os.environ.get(
            "SWP_ALPACA_PATH",
            "path/to/Llama-3.1-8B-Instruct-alpaca-100step",
        ),
        "description": "Fine-tuned 100 steps on Alpaca",
    },
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


# ─── Whitespace injection helpers ───

def get_chars_by_type(ws_type: str) -> List[dict]:
    """Get all whitespace entries for a given type category."""
    return WHITESPACE_POOL.get(ws_type, [])


def inject_ws(prompt: str, ws_entry: dict, position: str, rng: random.Random) -> Tuple[str, int]:
    """
    Inject a whitespace character at the specified position type.

    Args:
        prompt: Original prompt text
        ws_entry: Whitespace character entry dict
        position: Target position type (intra_word, inter_word, sent_start, sent_end)
        rng: Random number generator

    Returns:
        (perturbed_text, actual_position_index)
    """
    ch = ws_entry["char"]

    if position == "sent_start":
        return ch + prompt, 0
    elif position == "sent_end":
        return prompt + ch, len(prompt)
    elif position == "intra_word":
        words = prompt.split()
        if not words:
            return ch + prompt, 0
        target_word = rng.choice(words)
        idx = prompt.find(target_word)
        if idx == -1:
            return prompt + ch, len(prompt)
        insert_pos = idx + max(1, len(target_word) // 2)
        return prompt[:insert_pos] + ch + prompt[insert_pos:], insert_pos
    elif position == "inter_word":
        spaces = [i for i, c in enumerate(prompt) if c.isspace()]
        if not spaces:
            return prompt + ch, len(prompt)
        insert_pos = rng.choice(spaces)
        return prompt[:insert_pos] + ch + prompt[insert_pos:], insert_pos
    else:
        pos = rng.randint(0, len(prompt))
        return prompt[:pos] + ch + prompt[pos:], pos


# ─── Main experiment ───

def run_full_ablation_condition(
    state_key: str,
    prompts: List[str],
    position: str,
    ws_type: str,
    seed: int = SEED,
    llama_guard_eval: Optional[LlamaGuardEvaluator] = None,
) -> List[dict]:
    """
    Run one condition of the full ablation matrix.
    Uses the Llama-Guard-3-8B binary judge (cross-paper protocol, Sec. 5.7.3).

    Args:
        state_key: Fine-tuning state key
        prompts: List of prompt strings
        position: Insertion position type
        ws_type: Whitespace character type
        seed: Random seed
        llama_guard_eval: Shared Llama-Guard evaluator

    Returns:
        List of per-prompt result dicts
    """
    set_seed(seed)
    rng = random.Random(seed)

    state_config = FULL_ABLATION_STATES[state_key]
    ws_chars = get_chars_by_type(ws_type)

    if not ws_chars:
        logger.warning(f"No chars for type {ws_type}; skipping")
        return []

    if llama_guard_eval is None:
        llama_guard_eval = LlamaGuardEvaluator()

    tokenizer = AutoTokenizer.from_pretrained(state_config["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        state_config["model_path"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    jb_count = 0
    ted_values = []

    for i, prompt in enumerate(prompts):
        ws_entry = rng.choice(ws_chars)
        perturbed, pos_idx = inject_ws(prompt, ws_entry, position, rng)

        # Compute TED (Levenshtein over tokens, normalized)
        ted = compute_ted(prompt, perturbed, tokenizer)
        ted_values.append(ted)

        # Generate response
        device = next(model.parameters()).device
        inputs = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )

        # Cross-paper protocol (Sec. 5.7.3): Llama-Guard-3-8B binary judge.
        is_safe, _ = llama_guard_eval.classify(perturbed, response)
        is_jb = not is_safe
        l1_label = "harmful" if is_jb else "safe"
        l2_score = 0  # binary judge; kept for schema compatibility
        if is_jb:
            jb_count += 1

        results.append({
            "prompt_id": i,
            "state": state_key,
            "position": position,
            "ws_type": ws_type,
            "omega_char": ws_entry["code"],
            "omega_pos": pos_idx,
            "TED": round(ted, 4),
            "perturbed_text": perturbed,
            "response": response[:500],
            "L1_label": l1_label,
            "L2_score": l2_score,
            "is_jailbreak": int(is_jb),
        })

        if (i + 1) % 50 == 0:
            logger.info(f"    [{state_key}][{position}][{ws_type}] "
                        f"{i + 1}/{len(prompts)} (jailbreaks: {jb_count})")

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full ablation matrix — 36 conditions")
    parser.add_argument("--model", type=str, default="Llama-3.1-8B",
                        help="Model specification (currently uses per-state paths)")
    parser.add_argument("--states", type=str, default="base,gsm8k,alpaca",
                        help="Comma-separated fine-tuning states")
    parser.add_argument("--positions", type=str, default=",".join(INSERTION_POSITIONS),
                        help="Comma-separated insertion positions")
    parser.add_argument("--types", type=str, default=",".join(CHARACTER_TYPES),
                        help="Comma-separated character types")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--state-paths", type=str, default=None,
                        help="JSON dict mapping state key -> model path")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Apply CLI overrides to model paths
    if args.state_paths:
        import json as _json
        overrides = _json.loads(args.state_paths)
        for state_key, path in overrides.items():
            if state_key in FULL_ABLATION_STATES:
                FULL_ABLATION_STATES[state_key]["model_path"] = path
                logger.info(f"  Override: {state_key} -> {path}")

    states = [s.strip() for s in args.states.split(",")]
    positions = [p.strip() for p in args.positions.split(",")]
    ws_types = [t.strip() for t in args.types.split(",")]

    conditions = list(product(states, positions, ws_types))
    logger.info(f"Conditions: {len(states)} states × {len(positions)} positions × {len(ws_types)} types "
                f"= {len(conditions)} conditions × {args.runs} runs "
                f"= {len(conditions) * args.runs} experimental units")

    output_dir = args.output or os.path.join(RESULTS_DIR, "full_ablation")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FULL ABLATION MATRIX START")
    logger.info(f"States: {states}")
    logger.info(f"Positions: {positions}")
    logger.info(f"Types: {ws_types}")
    logger.info(f"Runs: {args.runs} | Seed: {args.seed}")
    logger.info("=" * 60)

    # Load the AdvBench evaluation prompt set (520 harmful behaviors, as reported in the paper)
    from load_datasets import load_all_datasets
    datasets = load_all_datasets(["AdvBench"])
    prompts = datasets.get("AdvBench", [])[:520]

    llama_guard_eval = LlamaGuardEvaluator()

    all_summaries = {}

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        logger.info(f"\n{'='*10} Run {run_idx + 1}/{args.runs} (seed={run_seed}) {'='*10}")

        for state_key in states:
            for position in positions:
                for ws_type in ws_types:
                    logger.info(f"  [{run_idx + 1}] {state_key} | {position} | {ws_type}")

                    t0 = time.time()
                    results = run_full_ablation_condition(
                        state_key, prompts, position, ws_type,
                        seed=run_seed, llama_guard_eval=llama_guard_eval,
                    )
                    elapsed = time.time() - t0

                    jb_count = sum(1 for r in results if r["is_jailbreak"])
                    asr = round(jb_count / len(prompts) * 100, 2) if prompts else 0.0
                    logger.info(f"    ASR={asr}% ({jb_count}/{len(prompts)}) [{elapsed:.0f}s]")

                    key = f"{state_key}|{position}|{ws_type}"
                    if key not in all_summaries:
                        all_summaries[key] = []
                    all_summaries[key].append({
                        "asr": asr,
                        "jb_count": jb_count,
                        "total": len(prompts),
                    })

                    # Save per-run CSV
                    csv_path = os.path.join(
                        output_dir,
                        f"run{run_idx + 1}_{state_key}_{position}_{ws_type}.csv",
                    )
                    if results:
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=results[0].keys())
                            writer.writeheader()
                            writer.writerows(results)

    # Aggregate
    agg = {}
    for key, summaries in all_summaries.items():
        if not summaries:
            continue
        asr_values = [s["asr"] for s in summaries]
        agg[key] = {
            "mean_asr": round(sum(asr_values) / len(asr_values), 2),
            "std_asr": round(
                (sum((v - sum(asr_values) / len(asr_values)) ** 2 for v in asr_values) / len(asr_values)) ** 0.5, 2,
            ),
            "n_runs": len(summaries),
        }

    # Check ranking consistency
    pos_ranking_states = set()
    type_ranking_states = set()

    logger.info("\nRanking consistency check:")
    for state_key in states:
        pos_asrs = {
            pos: sum(
                s["mean_asr"]
                for key, s in agg.items()
                if key.startswith(f"{state_key}|{pos}|")
            ) / sum(
                1 for key in agg.keys() if key.startswith(f"{state_key}|{pos}|")
            )
            for pos in positions
        }
        pos_ranking = sorted(pos_asrs, key=pos_asrs.get, reverse=True)
        logger.info(f"  {state_key} position ranking: {' > '.join(pos_ranking)}")
        pos_ranking_states.add(" > ".join(pos_ranking))

        type_asrs = {
            t: sum(
                s["mean_asr"]
                for key, s in agg.items()
                if key.endswith(f"|{t}")
            )
            for t in ws_types
        }
        type_ranking = sorted(type_asrs, key=type_asrs.get, reverse=True)
        logger.info(f"  {state_key} type ranking:     {' > '.join(type_ranking)}")
        type_ranking_states.add(" > ".join(type_ranking))

    consistent_pos = len(pos_ranking_states) == 1
    consistent_type = len(type_ranking_states) == 1
    logger.info(f"\nPosition ranking preserved: {consistent_pos}")
    logger.info(f"Type ranking preserved:     {consistent_type}")

    # Compute TED–ASR Spearman ρ per state (from all available CSV rows)
    rho_by_state = {}
    for state_key in states:
        state_rows = []
        for fname in os.listdir(output_dir):
            if fname.endswith(".csv") and state_key in fname:
                from csv import DictReader
                with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                    state_rows.extend(list(DictReader(f)))
        if state_rows:
            ted_vals = [float(r.get("TED", 0)) for r in state_rows if r.get("TED")]
            asr_vals = [int(r.get("is_jailbreak", 0)) for r in state_rows if r.get("TED")]
            if len(ted_vals) > 1 and len(ted_vals) == len(asr_vals):
                rho = spearman_r(ted_vals, asr_vals)
                rho_by_state[state_key] = round(rho, 4)
                logger.info(f"  {state_key} TED–ASR ρ = {rho:.4f} (n={len(ted_vals)})")

    # Save summary
    output_summary = {
        "conditions": {"states": states, "positions": positions, "types": ws_types},
        "n_conditions": len(conditions),
        "n_runs": args.runs,
        "results": agg,
        "ranking_consistency": {
            "position_consistent": consistent_pos,
            "type_consistent": consistent_type,
        },
        "spearman_rho_TED_ASR": rho_by_state,
    }

    output_path = os.path.join(output_dir, "full_ablation_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    logger.info(f"\nFull ablation results saved to {output_path}")
    logger.info("FULL ABLATION MATRIX COMPLETE")


if __name__ == "__main__":
    main()
