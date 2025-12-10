## ReATT: minGPT experiments setup (Lab 4 - Transformers)

### What this directory is
- Training entrypoint: `projects/reatt/reatt.py`
- Dataset used by training: `mingpt/jsonl_dataset.py`
- Batch script for the supercomputer: `scripts/reatt_train.sbatch`
- Outputs: `out/reatt_batch_<JOBID>/` (when run via sbatch) or `./out/reatt` (local default)

### Goal
Run a tiny, reproducible minGPT baseline on JSONL data, then add and ablate architecture/training changes:
- SwiGLU (MLP)
- RMSNorm (norm layers)
- Rotary Position Embeddings (RoPE)
- Learning rate warmup
- Cosine learning rate schedule

## Dataset: `JSONLDataset`
- One document per JSONL line with a `text` field.
- Tokenizer: `GPT2TokenizerFast` (avoids protobuf issues and supports offline).
- Each sample returns a token window of length `block_size + 1`; we output `(x, y)` where `y` is `x` shifted by one.
- Short docs are resampled; long docs pick a random in-doc window.
- Config fields:
  - `data.file_path` (path to `.jsonl`)
  - `data.split` (`'train'`)
  - `data.test_size`
  - `data.block_size`

Example CLI overrides:
```bash
python -m projects.reatt.reatt \
  --data.file_path='/home/tbday/minGPT/pile_data_10_first_10.jsonl' \
  --data.block_size=256
```

## Config and CLI (CfgNode, alias CN)
`mingpt.utils.CfgNode` builds a nested config tree (e.g., `system`, `data`, `model`, `trainer`).
`config.merge_from_args(sys.argv[1:])` applies overrides like:
- `--model.model_type='gpt-nano'`
- `--trainer.batch_size=1`
- `--trainer.max_iters=1000`
- `--system.work_dir='./out/reatt_smoke'`

The attribute must exist in the config or an assertion will fire.

## Training: local run
Minimal smoke test:
```bash
python -m projects.reatt.reatt \
  --data.file_path='/home/tbday/minGPT/pile_data_10_first_10.jsonl' \
  --data.block_size=256 \
  --model.model_type='gpt-nano' \
  --trainer.batch_size=1 \
  --trainer.max_iters=1000 \
  --trainer.num_workers=0 \
  --system.work_dir='./out/reatt_smoke'
```

Logging and sampling:
- Loss CSV every `log_interval` steps: `loss.csv` in `system.work_dir`
- Optional periodic text samples (tokenizer-based) are printed to stdout
- Checkpoint saving is scaffolded; enable by uncommenting the save block in `reatt.py`

## Supercomputer run (sbatch)
One-time on a LOGIN node (with internet) to warm the HF cache:
```bash
module load python/3.11 || true
source /home/tbday/minGPT/.venv/bin/activate
export HF_HOME=/home/tbday/nobackup/autodelete/hf

pip install --upgrade 'protobuf<5' transformers tokenizers
python - <<'PY'
from transformers import GPT2TokenizerFast
GPT2TokenizerFast.from_pretrained('gpt2')
print("HF cache warmed.")
PY
```
Net effect: one-time cache warm-up on a login node with internet so your compute jobs can run fully offline using the cached GPT-2 tokenizer.

More specifically:
- module load python/3.11 || true: Tries to load the site’s Python 3.11 module; if the cluster doesn’t have modules or it fails, continue anyway.
- source /home/tbday/minGPT/.venv/bin/activate: Activates your project’s virtual environment so pip/python go into it.
- export HF_HOME=...: Sets the Hugging Face cache directory to a location you control on the cluster.
- pip install --upgrade 'protobuf<5' transformers tokenizers: Ensures compatible versions for GPT2TokenizerFast and avoids protobuf issues.
- The heredoc python block:
  - Imports GPT2TokenizerFast and calls from_pretrained('gpt2'), which downloads the tokenizer files (vocab.json, merges.txt, tokenizer.json, etc.) into HF_HOME.
  - Prints “HF cache warmed.” so you know the cache is ready.


Submit a job:
```bash
sbatch --export=ALL,MAX_ITERS=500,BLOCK_SIZE=256,BATCH_SIZE=1 /home/tbday/minGPT/scripts/reatt_train.sbatch
```

What the script does:
- Uses venv python; sets `HF_HOME` to `/home/tbday/nobackup/autodelete/hf`
- Forces offline mode: `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`
- Runs `python -m projects.reatt.reatt` with your overrides

Where to look:
- Stdout/err: `out/slurm/reatt-<JOBID>.out|.err`
- Artifacts: `out/reatt_batch_<JOBID>/loss.csv`, `config.json`, `args.txt`

## Where to implement each modification
Edit `mingpt/model.py`:
- SwiGLU:
  - In `Block`, replace the GELU MLP with a gated two-projection MLP (SiLU gate) when `config.swiglu` is true.
  - Suggestion: add `model.swiglu: bool` to `GPT` config.
- RMSNorm:
  - Add a small `RMSNorm` module and swap for `LayerNorm` in `Block` and final `ln_f` when `config.rmsnorm` is true.
- Rotary Positional Embeddings (RoPE):
  - In `CausalSelfAttention`, apply rotary to `q` and `k` just before attention matmul when `config.rope` is true.
  - You may keep token embeddings; consider bypassing learned `wpe` while RoPE is enabled.

Edit `mingpt/trainer.py`:
- Learning rate warmup:
  - Add `warmup_steps` and, if enabled, scale LR linearly from 0 to base LR over warmup.
- Cosine schedule:
  - If enabled, decay LR via cosine after warmup until `max_iters`.
  - Add flags to `Trainer` config (e.g., `use_cosine: bool`, `warmup_steps: int`).

Expose toggles via CLI by adding fields to `C.model`/`C.trainer` and overriding with `--model.swiglu=True`, etc.

## Ablations (7 runs)
1) Baseline (all flags off), log final NLL (cross-entropy loss).
2) One run per feature: enable exactly one of `swiglu`, `rmsnorm`, `rope`, `warmup`, `cosine`.
3) One run with all features on.
Use unique `--system.work_dir` per run and collect `loss.csv` for plotting.

## Summary of changes from this setup
- Added `JSONLDataset` that returns `(x, y)` token ids with random in-doc windows and resampling of short docs.
- Switched to `GPT2TokenizerFast` and made loading offline-aware to avoid protobuf/network issues on compute nodes.
- Added a simple CSV loss logger and tokenizer-based sampling in `reatt.py`.
- Created `scripts/reatt_train.sbatch` that sets up offline execution and uses your venv’s Python.
- Clarified how to use `CfgNode` (`CN`) and CLI overrides to control runs.

## Troubleshooting
- Network unreachable / tokenizer download:
  - Warm the HF cache on a login node (see above), then run with offline vars.
- Protobuf errors:
  - Using `GPT2TokenizerFast` avoids this dependency.
- OOM / slow training:
  - Lower `--data.block_size`, switch to `--model.model_type='gpt-nano'`, keep `--trainer.batch_size=1`.


