# Installation

IRIS runs on Linux with an NVIDIA GPU. The main pipeline lives in one conda env
(`iris`); two optional models (PowerPaint, TRELLIS) live in their own envs because
their dependencies conflict with the main stack — they are driven as subprocess
workers (see [ax.md](ax.md) §3). All paths resolve from the repo root or env vars
([../src/config.py](../src/config.py)), so the project is portable across machines.

> All models are open-weight; most auto-download from Hugging Face on first run.
> No API keys or paid services are needed at runtime.

## Reproducible setup (3 steps)

```bash
git clone <this-repo> IRIS && cd IRIS

# 1. create conda envs + clone third-party model repos (iris / trellis / powerpaint)
bash scripts/setup_envs.sh                 # or: setup_envs.sh iris  (just the main env)

# 2. fetch the weights that aren't auto-downloaded (PowerPaint; RORem instructions)
conda run -n iris python scripts/fetch_weights.py

# 3. run
conda run -n iris python src/pipeline.py --image data/test3.png --output_dir output
```

`setup_envs.sh` builds each env from the pinned `requirements-{iris,trellis,powerpaint}.txt`
and applies the small compat shims (TRELLIS kaolin stub; the xformers alias is in
code). TripoSR (fallback image-to-3D) and VGGT are installed into `iris`.

## What runs where

| Env | Purpose | Key pins |
|-----|---------|----------|
| `iris` | full pipeline (VLM, SAM3, depth, RORem, VGGT, Mask2Former, occupancy) | torch 2.5.1+cu121, transformers 5.9, diffusers 0.38 |
| `trellis` | image-to-3D worker (`--image3d trellis`) | torch 2.4.1+cu118, spconv-cu118, xformers 0.0.28 |
| `powerpaint` | removal A/B worker (`--remover powerpaint`) | torch 2.1.2, transformers 4.28, diffusers 0.27 |

## Configuration / overrides

All optional; defaults resolve under the repo. Set if your layout differs:

| Env var | Default |
|---------|---------|
| `IRIS_ROREM_CKPT` | `checkpoints/RORem` |
| `IRIS_POWERPAINT_CKPT` | `models/PowerPaint/checkpoints/ppt-v2-1` |
| `IRIS_TRELLIS_DIR` / `IRIS_POWERPAINT_DIR` / `IRIS_TRIPOSR_DIR` | `models/<name>` |
| `IRIS_TRELLIS_PYTHON` / `IRIS_POWERPAINT_PYTHON` | auto-detected conda env python |

The worker envs are auto-located via `CONDA_EXE` / common conda roots; override the
`*_PYTHON` vars if needed.

## Weights notes

- **RORem** (default remover) is not openly downloadable — get its SDXL-inpainting
  UNet checkpoint from <https://github.com/leeruibin/RORem>, place it at
  `checkpoints/RORem` (or set `IRIS_ROREM_CKPT`). Or just use `--remover lama`
  (no extra weights).
- **PowerPaint v2-1** auto-downloads + converts in `fetch_weights.py`.
- Everything else (Qwen3-VL, SAM3, Depth-Anything-V2, SDXL-inpainting, TRELLIS,
  VGGT, Mask2Former) auto-downloads from Hugging Face on first use.

## GPU notes

- Developed on a 24 GB RTX 3090; runs comfortably on larger GPUs (e.g. H100 80 GB),
  where the multi-view path and higher resolutions are unconstrained.
- On a power-constrained machine, sustained load can trip the PSU; cap with
  `sudo nvidia-smi -pl 200` and rely on `--resume` (per-phase / per-object
  checkpointing). Not needed on datacenter GPUs.

See [user_guide.md](user_guide.md) for usage and flags.
