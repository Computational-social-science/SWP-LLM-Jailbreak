# SWP — Reproducibility Instructions

This archive contains the code for all experiments reported in the paper. It does not include raw per-run records (see Notes).

## Environment

- Tested on: 3× NVIDIA RTX 4090 (24 GB), CUDA 12.2
- Python 3.10+ with PyTorch 2.1+ and Transformers 4.36+
- Multi-GPU mapping in `config.py` (GPU_MAPPING) can be adjusted for any setup
- Per-model VRAM requirements: ~1.2 GB (T5-small) to ~25.8 GB (Llama2-13B)
- The five models can fit on a single 24 GB GPU via sequential loading; multi-GPU speeds up runs

## Setup

```bash
pip install -r requirements.txt
```

## Experiment Scripts

All scripts accept `--help` for full argument documentation. Shared configuration (seeds, model registry, dataset registry, thresholds) lives in `config.py`.

### Core Pipeline

| Script | Description |
|--------|-------------|
| `build_whitespace_pool.py` | Validate the perturbation pool (30 entries: 25 `White_Space` + 5 zero-width); export pool as JSON |
| `load_datasets.py` | Load & cache 8 jailbreak benchmarks (11,100 prompts; AdvBench is the 520-item `harmful_behaviors` half) |
| `evaluate.py` | Two-level safety evaluation: L1 RoBERTa binary + L2 GPT-3.5 scoring |
| `papillon.py` | Papillon baseline (`Q_base = Framework(Q_original)`) for the "Original" condition; backend selected via `config.PAPILLON_BACKEND` |

### Experiments

| Script | Description | Output |
|--------|-------------|--------|
| `run_baseline.py` | Original condition (Papillon baseline, no whitespace; 5 models × 8 datasets) | `results/baseline/` |
| `run_swp.py` | Main SWP attack experiment | `results/swp/` |
| `aggregate_results.py` | Aggregate per-run CSVs → summary JSONs | `results/swp_summary.json` |
| `run_sota_comparison.py` | GCG / PAIR / AutoDAN comparison | `results/sota/` |
| `run_ablation.py` | Position × character type ablation (12 conditions) | `results/ablation/` |
| `run_tokenizer_ablation.py` | BBPE / BPE / Unigram tokenizer comparison | `results/tokenizer_ablation/` |
| `run_defense.py` | 5 defense configurations | `results/defense/` |
| `analyze_attention.py` | Attention shift analysis (Llama2-7B) | `results/attention/` |
| `visualize_bomb_attention.py` | Attention-shift mechanism visualization + batch statistics (single/batch, eval/no_eval). `--model` accepts the five target models plus `Qwen1.5-0.5B-Chat` (registered in `config.PROBE_MODELS`). See `experiment_attention.md` for the protocol. | `results/attention_viz/` |
| `run_cross_paper.py` | Cross-paper validation (Llama-3.1-8B) | `results/cross_paper/` |
| `run_full_ablation.py` | 36-condition full ablation matrix | `results/full_ablation/` |

### Figures

| Script | Output |
|--------|--------|
| `generate_fig_token.py` | `figures/fig_token_comparison.pdf` / `.png` |

## Quickstart

```bash
# 1. Build whitespace pool
python build_whitespace_pool.py

# 2. Load datasets
python load_datasets.py --all

# 3. Generate the tokenization figure
python generate_fig_token.py
```

## Reproducibility

- All seeds fixed to **42**. GPT-3.5 temperature = **0** (deterministic scoring).
- `generate_fig_token.py` reproduces the tokenization figure from the paper.

## Notes

- The `run_cross_paper.py` and `run_full_ablation.py` scripts assume fine-tuned checkpoints are available at the paths specified in their respective `FINETUNING_STATES` / `FULL_ABLATION_STATES` dictionaries.
- The main experiment's "Original" condition is the Papillon baseline (`Q_base = Framework(Q_original)`). By default `config.PAPILLON_BACKEND = "stub"` returns the raw prompt unchanged so the pipeline runs without the official Papillon repo; set it to `"official"` and wire `papillon.py` to `github.com/aaFrostnova/Papillon` to reproduce the exact baseline prompts. The cross-paper replication does not apply Papillon (its "Original" is the raw prompt).
- **The `"official"` backend is not wired end-to-end in this package**: `papillon.py::_read_papillon_result` raises `RuntimeError` because parsing the upstream repo's result layout could not be verified offline. The reported "Original" numbers were produced with the upstream mutation operator, which is not redistributed here. Running with the default `"stub"` therefore yields the *raw* prompt as the baseline, not the Papillon baseline.
- The mechanistic-probe tool and our extension of it are in `../logitscope/`; see `../THIRD_PARTY.md` for attribution and the list of files we changed.
- Per-run results are written as CSV under `results/` when the pipeline is run; this package contains code only, not raw records.
