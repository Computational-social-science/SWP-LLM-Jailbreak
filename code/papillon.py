"""
Papillon baseline integration (Algorithm 1 of the paper).

The main experiment defines the "Original" condition as the Papillon baseline:
the benchmark prompt passed through Papillon's mutation operators once
(Gong et al., USENIX Security 2025):

    Q_base <- Framework(Q_original)     # single mutation-operator pass

"Ours"/SWP then inserts exactly one whitespace character on top of Q_base.
This module encapsulates that Framework() step so run_baseline.py and
run_swp.py share a single definition of the baseline prompt.

Important: `Framework()` here is a SINGLE mutation-operator pass, NOT Papillon's
full iterative fuzz search. Full fuzz is a strong SOTA attack (60-90% ASR) and
belongs in run_sota_comparison.py, not in the "Original" column.

Note: the cross-paper replication (run_cross_paper.py) does NOT apply this step;
there "Original" is the raw, unperturbed prompt (see Sec. 5.7.3 of the paper).

Mode (config.PAPILLON_BASELINE_MODE):
  - "single_mutation": one mutation-operator pass -> the "Original" baseline (default).
  - "full_fuzz":       full iterative fuzz -> SOTA attack; rejected here.

Backend (config.PAPILLON_BACKEND):
  - "stub":     identity placeholder used when the official repo is not run;
                keeps the pipeline runnable end-to-end without mutation.
  - "official": call the official Papillon repo's mutation operator to obtain
                the single-mutation baseline prompt for each question.
"""

import logging
import os
import subprocess

from config import (
    PAPILLON_BACKEND,
    PAPILLON_BASELINE_MODE,
    PAPILLON_REPO,
    PAPILLON_REPO_PATH,
)

logger = logging.getLogger(__name__)


def papillon_baseline(prompt: str) -> str:
    """Return Q_base = Framework(Q_original) for a raw benchmark prompt.

    `Framework()` is a SINGLE mutation-operator pass. If
    PAPILLON_BASELINE_MODE == "full_fuzz", that is the SOTA Papillon attack and
    belongs in run_sota_comparison.py, not in the "Original" baseline.
    """
    if PAPILLON_BASELINE_MODE == "full_fuzz":
        raise RuntimeError(
            "PAPILLON_BASELINE_MODE='full_fuzz' is the SOTA Papillon attack; "
            "it belongs in run_sota_comparison.py, not in the 'Original' baseline."
        )
    if PAPILLON_BACKEND == "official":
        return _official_papillon_baseline(prompt)
    logger.debug("Papillon backend=stub: returning prompt unchanged (single_mutation placeholder)")
    return prompt


def _official_papillon_baseline(prompt: str) -> str:
    """Obtain the single-mutation baseline prompt from the official Papillon repo.

    This applies ONE mutation-operator pass (role-play / contextualization /
    expand), NOT Papillon's full iterative fuzz search (which is a SOTA attack).

    Requires `config.PAPILLON_REPO_PATH` (a local clone of `PAPILLON_REPO`) and
    an `OPENAI_API_KEY` for the repo's LLM helper. Point the command below at the
    repo's mutation operator; the repo's `run.py` is the full fuzz pipeline and
    is NOT the baseline path.
    """
    repo = PAPILLON_REPO_PATH
    if not repo or not os.path.isdir(repo):
        raise RuntimeError(
            "PAPILLON_BACKEND='official' requires PAPILLON_REPO_PATH to point "
            f"to a local clone of {PAPILLON_REPO}."
        )
    key = os.environ.get("OPENAI_API_KEY", "")
    proc = subprocess.run(
        [
            "python", "run.py",
            "--openai_key", key,
            "--model_path", "gpt-3.5-turbo",
            "--target_model", "gpt-3.5-turbo",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Papillon run.py failed: {proc.stderr}")
    return _read_papillon_result(repo, prompt)


def _read_papillon_result(repo: str, prompt: str) -> str:
    """Read the baseline prompt for `prompt` from the official repo's output.

    Papillon writes attacked prompts to a result file under `repo`; adapt this to
    the repo's actual result layout (cannot be fully verified offline).
    """
    raise RuntimeError(
        "Papillon official backend: result parsing not yet wired. "
        "Read the repo's result file for `prompt` and return the baseline prompt."
    )
