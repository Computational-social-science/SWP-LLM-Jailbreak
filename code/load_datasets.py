"""
Dataset Loader — 8 Jailbreak Benchmarks

Loads all 8 benchmark datasets from HuggingFace / web sources,
caches them to a single pickle file for fast reload across experiments.

Datasets: AdvBench, HarmBench, JailbreakBench, XSTEST, Do-Not-Answer,
          Tech-HazardQA, StrongREJECT, AttackEval
Total: 11,100 samples.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

from config import DATASETS, DATA_CACHE_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(DATA_CACHE_DIR, "swp_datasets.pkl")


def _try_import_datasets():
    """Attempt to import HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        return True, load_dataset
    except ImportError:
        return False, None


def _load_advbench() -> List[str]:
    """Load AdvBench harmful prompts (520 samples: the harmful_behaviors half)."""
    logger.info("Loading AdvBench...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        logger.warning("datasets library not available; returning empty AdvBench")
        return []

    try:
        # The repository's other half, "harmful_strings" (574 rows, column "target"),
        # holds harmful *output* strings used as optimisation targets, not user
        # prompts, so it is deliberately excluded from the prompt set.
        ds = load_dataset("llm-attacks/llm-attacks", "harmful_behaviors", split="train")
        prompts = [item["goal"] for item in ds]
        return prompts[:520]
    except Exception as e:
        logger.error(f"Failed to load AdvBench: {e}")
        return []


def _load_harmbench() -> List[str]:
    """Load HarmBench harmful prompts (510 samples)."""
    logger.info("Loading HarmBench...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("walledai/HarmBench", split="train")
        prompts = [item["prompt"] if "prompt" in item else item["text"] for item in ds]
        return prompts[:510]
    except Exception as e:
        logger.error(f"Failed to load HarmBench: {e}")
        return []


def _load_jailbreakbench() -> List[str]:
    """Load JailbreakBench (200 samples)."""
    logger.info("Loading JailbreakBench...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("EddyLuo/JailbreakBench", split="train")
        prompts = [item["prompt"] for item in ds]
        return prompts[:200]
    except Exception as e:
        logger.error(f"Failed to load JailbreakBench: {e}")
        return []


def _load_xstest() -> List[str]:
    """Load XSTEST (450 samples)."""
    logger.info("Loading XSTEST...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("walledai/XSTest", "xstest_v2_prompts", split="train")
        prompts = [item["prompt"] for item in ds]
        return prompts[:450]
    except Exception as e:
        logger.error(f"Failed to load XSTEST: {e}")
        return []


def _load_donotanswer() -> List[str]:
    """Load Do-Not-Answer (939 samples)."""
    logger.info("Loading Do-Not-Answer...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("LibrAI/do-not-answer", split="train")
        prompts = [item["question"] for item in ds]
        return prompts[:939]
    except Exception as e:
        logger.error(f"Failed to load Do-Not-Answer: {e}")
        return []


def _load_techhazardqa() -> List[str]:
    """Load Tech-HazardQA (7,745 samples)."""
    logger.info("Loading Tech-HazardQA...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("SoftMINER-Group/TechHazardQA", split="train")
        prompts = [item["question"] if "question" in item else item["text"] for item in ds]
        return prompts[:7745]
    except Exception as e:
        logger.error(f"Failed to load Tech-HazardQA: {e}")
        return []


def _load_strongreject() -> List[str]:
    """Load StrongREJECT (346 samples)."""
    logger.info("Loading StrongREJECT...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("alexandrasouly/strongreject", split="train")
        prompts = [item["forbidden_prompt"] for item in ds]
        return prompts[:346]
    except Exception as e:
        logger.error(f"Failed to load StrongREJECT: {e}")
        return []


def _load_attackeval() -> List[str]:
    """Load AttackEval (390 samples)."""
    logger.info("Loading AttackEval...")
    ok, load_dataset = _try_import_datasets()
    if not ok:
        return []

    try:
        ds = load_dataset("AttackEval/AttackEval", split="train")
        prompts = [item["prompt"] for item in ds]
        return prompts[:390]
    except Exception as e:
        logger.error(f"Failed to load AttackEval: {e}")
        return []


# Registry of loaders
_LOADERS = {
    "AdvBench": _load_advbench,
    "HarmBench": _load_harmbench,
    "JailbreakBench": _load_jailbreakbench,
    "XSTEST": _load_xstest,
    "Do-Not-Answer": _load_donotanswer,
    "Tech-HazardQA": _load_techhazardqa,
    "StrongREJECT": _load_strongreject,
    "AttackEval": _load_attackeval,
}


def load_all_datasets(
    dataset_names: Optional[List[str]] = None,
    force_reload: bool = False,
) -> Dict[str, List[str]]:
    """
    Load specified datasets (or all if None). Returns dict of name -> list of prompt strings.
    Results are cached to data_cache/swp_datasets.pkl.

    Args:
        dataset_names: Subset of dataset names to load (None = all 8).
        force_reload: If True, skip cache and re-download.

    Returns:
        dict: {dataset_name: [prompt_str, ...]}
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    if not force_reload and os.path.exists(CACHE_PATH):
        logger.info(f"Loading cached datasets from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if dataset_names:
            return {k: v for k, v in cached.items() if k in dataset_names}
        return cached

    if dataset_names is None:
        dataset_names = list(_LOADERS.keys())

    results = {}
    total = 0
    for name in dataset_names:
        if name not in _LOADERS:
            logger.warning(f"Unknown dataset: {name}")
            continue
        prompts = _LOADERS[name]()
        results[name] = prompts
        total += len(prompts)
        logger.info(f"  {name}: {len(prompts)}")

    logger.info(f"TOTAL: {total} samples")

    # Cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Cache written to {CACHE_PATH}")

    return results


def get_dataset_samples(dataset_name: str) -> List[str]:
    """Load a single dataset by name."""
    return load_all_datasets([dataset_name]).get(dataset_name, [])


# ─── CLI ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and cache jailbreak benchmark datasets")
    parser.add_argument("--all", action="store_true", default=False, help="Load all 8 datasets")
    parser.add_argument("--dataset", type=str, help="Load a specific dataset by name")
    parser.add_argument("--force", action="store_true", help="Force re-download, skip cache")
    args = parser.parse_args()

    if args.dataset:
        datasets = load_all_datasets([args.dataset], force_reload=args.force)
    else:
        datasets = load_all_datasets(force_reload=args.force)

    for name, prompts in datasets.items():
        print(f"{name}: {len(prompts)} samples")
        print(f"  Example: {prompts[0][:80]}..." if prompts else "  (empty)")
