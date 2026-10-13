"""
Aggregate experiment results from per-run CSVs into summary JSONs.

Outputs:
  - results/swp_summary.json       (pooled SWP results)
  - results/delta_summary.json     (delta ASR vs baseline)
  - results/subgroup_llama2.json   (Llama2 family analysis)
  - results/subgroup_7b.json       (7B cross-architecture analysis)

Usage:
    python aggregate_results.py results/swp/ --baseline results/baseline/baseline_summary.json
"""

import csv
import json
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

from config import RESULTS_DIR, MODELS, DATASETS
from evaluate import paired_t_test

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def load_csvs_from_dir(directory: str) -> List[dict]:
    """Load all CSV files from a directory into a list of rows."""
    rows = []
    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return rows

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        filepath = os.path.join(directory, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    return rows


def compute_asr(rows: List[dict], key_fn=None) -> dict:
    """
    Compute ASR (Attack Success Rate) from rows.

    Args:
        key_fn: Optional grouping function, e.g., lambda r: r["model"].
                If provided, returns nested dict {group: stats}.
    """
    if not rows:
        return {"total": 0, "asr_percent": 0.0}

    if key_fn is not None:
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r)

        result = {}
        for key, group_rows in groups.items():
            result[key] = compute_asr(group_rows)
        return result

    total = len(rows)
    jb_count = sum(1 for r in rows if r.get("is_jailbreak") == "1" or r.get("is_jailbreak") == 1)
    asr = round(jb_count / total * 100, 2) if total > 0 else 0.0

    return {
        "total": total,
        "jailbreak_count": jb_count,
        "asr_percent": asr,
    }


def load_baseline_summary(path: str) -> dict:
    """Load baseline summary JSON."""
    if not os.path.exists(path):
        logger.warning(f"Baseline summary not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_delta(swp_stats: dict, baseline_stats: dict) -> dict:
    """
    Compute delta ASR between SWP and baseline.
    Includes paired t-test p-values when per-run data is available.
    """
    result = {}
    swp_asrs = []
    base_asrs = []
    for key in swp_stats:
        if key not in baseline_stats:
            continue
        swp = swp_stats[key]
        base = baseline_stats[key]
        swp_asr = swp.get("asr_percent", swp.get("mean_asr", 0))
        base_asr = base.get("mean_asr", base.get("asr_percent", 0))
        swp_asrs.append(swp_asr)
        base_asrs.append(base_asr)
        result[key] = {
            "baseline_asr": base_asr,
            "swp_asr": swp_asr,
            "delta_asr": round(swp_asr - base_asr, 1),
        }

    # Overall paired t-test across all model-dataset pairs
    if len(swp_asrs) > 1 and len(swp_asrs) == len(base_asrs):
        t_result = paired_t_test(swp_asrs, base_asrs)
        result["_overall_paired_t_test"] = t_result
        logger.info(
            f"Overall paired t-test: t={t_result.get('t_statistic'):.4f}, "
            f"p={t_result.get('p_value', 'N/A')}, "
            f"95% CI = [{t_result.get('ci_95_lower', 'N/A')}, {t_result.get('ci_95_upper', 'N/A')}]"
        )

    return result


def aggregate_swp_results(
    swp_dir: str,
    baseline_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Main aggregation function.

    Reads all SWP CSVs from swp_dir, optionally compares with baseline,
    and writes summary JSONs to output_dir.
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    rows = load_csvs_from_dir(swp_dir)
    logger.info(f"Loaded {len(rows)} rows from {swp_dir}")

    if not rows:
        logger.warning("No data loaded; writing empty summaries")
        return {}

    # ─── SWP Summary ───
    # Group by model + dataset
    groups = defaultdict(list)
    for r in rows:
        key = f"{r.get('model', '')}|{r.get('dataset', '')}"
        groups[key].append(r)

    swp_summary = {}
    for key, group_rows in groups.items():
        swp_summary[key] = compute_asr(group_rows)

    # Save SWP summary
    swp_path = os.path.join(output_dir, "swp_summary.json")
    with open(swp_path, "w", encoding="utf-8") as f:
        json.dump(swp_summary, f, indent=2)
    logger.info(f"SWP summary saved to {swp_path}")

    # ─── Delta vs Baseline ───
    if baseline_path:
        baseline = load_baseline_summary(baseline_path)
        if baseline:
            delta = compute_delta(swp_summary, baseline)
            delta_path = os.path.join(output_dir, "delta_summary.json")
            with open(delta_path, "w", encoding="utf-8") as f:
                json.dump(delta, f, indent=2)
            logger.info(f"Delta summary saved to {delta_path}")

            # Compute overall mean delta
            all_deltas = [d["delta_asr"] for d in delta.values()]
            if all_deltas:
                mean_delta = round(sum(all_deltas) / len(all_deltas), 1)
                logger.info(f"Mean Δ ASR = +{mean_delta}%")

    # ─── Llama2 subgroup ───
    llama2_models = ["Llama2-7B", "Llama2-13B"]
    llama2_rows = [r for r in rows if r.get("model") in llama2_models]
    if llama2_rows:
        llama2_stats = compute_asr(llama2_rows, key_fn=lambda r: r["model"])
        llama2_path = os.path.join(output_dir, "subgroup_llama2.json")
        with open(llama2_path, "w", encoding="utf-8") as f:
            json.dump(llama2_stats, f, indent=2)
        logger.info(f"Llama2 subgroup saved to {llama2_path}")

    # ─── 7B cross-architecture ───
    b7_models = ["Llama2-7B", "Gemma-7B", "Qwen-7B"]
    b7_rows = [r for r in rows if r.get("model") in b7_models]
    if b7_rows:
        b7_stats = compute_asr(b7_rows, key_fn=lambda r: r["model"])
        b7_path = os.path.join(output_dir, "subgroup_7b.json")
        with open(b7_path, "w", encoding="utf-8") as f:
            json.dump(b7_stats, f, indent=2)
        logger.info(f"7B cross-architecture saved to {b7_path}")

    return swp_summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SWP experiment results")
    parser.add_argument("swp_dir", type=str, help="Directory containing SWP CSV files")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline_summary.json")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()

    aggregate_swp_results(args.swp_dir, args.baseline, args.output)


if __name__ == "__main__":
    main()
