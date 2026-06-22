# Installation

IRIS runs on Linux with an NVIDIA GPU. The main pipeline lives in one conda env
(`iris`); each image-to-3D backend (Amodal3R, TRELLIS) lives in its own env because
its dependencies conflict with the main stack — they are driven as subprocess
workers (see [ax.md](ax.md) §3). All paths resolve from the repo root or env vars
([../src/config.py](../src/config.py)), so the project is portable across machines.

> All models are open-weight; most auto-download from Hugging Face on first run.
> No API keys or paid services are needed at runtime.

## Reproducible setup (3 steps)

```bash
git clone <this-repo> IRIS && cd IRIS

# 1. create conda envs + clone third-party model repos (iris + image-to-3D backends)
bash scripts/setup_envs.sh                 # or: setup_envs.sh iris  (just the main env)

# 2. fetch the weights that aren't auto-downloaded (RORem instructions below)
conda run -n iris python scripts/fetch_weights.py

# 3. run (default backend = occlusion-aware Amodal3R, in the amodal3r env)
conda run -n iris python src/pipeline.py --image data/demoImage.png --output_dir output --image3d amodal3r
```

`setup_envs.sh` builds the `iris` env plus the image-to-3D backend env(s) with their
pinned dependencies, and applies the small compat shims (e.g. the TRELLIS xformers
alias is in code). VGGT is installed into `iris`.

## What runs where

| Env | Purpose | Key pins |
|-----|---------|----------|
| `iris` | full pipeline (VLM, SAM3, depth, RORem, VGGT, Mask2Former, occupancy) | torch 2.5.1+cu121, transformers 5.9, diffusers 0.38 |
| `amodal3r` | **Amodal3R** image-to-3D (`--image3d amodal3r`, default) | torch 2.4.0+cu118, spconv-cu118, xformers 0.0.27 |
| `trellis` | TRELLIS image-to-3D (`--image3d trellis`, baseline) | torch 2.4.1+cu118, spconv-cu118, xformers 0.0.28 |

Only the `iris` env plus the env for your chosen `--image3d` backend are required;
the default uses the `amodal3r` env.

## Configuration / overrides

All optional; defaults resolve under the repo. Set if your layout differs:

| Env var | Default |
|---------|---------|
| `IRIS_ROREM_CKPT` | `checkpoints/RORem` |
| `IRIS_VLM_ID` | `Qwen/Qwen3-VL-32B-Instruct` (set to the 8B id for a lighter run) |
| `IRIS_AMODAL3R_DIR` / `IRIS_TRELLIS_DIR` | `models/<name>` |
| `IRIS_AMODAL3R_PYTHON` / `IRIS_TRELLIS_PYTHON` | auto-detected conda env python |

The worker envs are auto-located via `CONDA_EXE` / common conda roots; override the
`*_PYTHON` vars if needed.

## Weights notes

- **RORem** (the object remover) is required — get its SDXL-inpainting UNet
  checkpoint from <https://github.com/leeruibin/RORem> and place it at
  `checkpoints/RORem` (or set `IRIS_ROREM_CKPT`). It loads on top of the base
  SDXL-inpainting pipeline (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`),
  which auto-downloads from Hugging Face.
- **SAM 3** (`facebook/sam3`) is a gated Hugging Face repo — accept its terms and
  export an `HF_TOKEN` so it (and any other gated weights) download.
- Everything else (Qwen3-VL, SAM3, Depth-Anything-V2, SDXL-inpainting, Amodal3R,
  TRELLIS, VGGT, Mask2Former) auto-downloads from Hugging Face on first use.

## GPU notes

- The default **32B VLM** is the memory peak (~65 GB, transient — loaded for
  discovery, freed before peeling), so it expects a large GPU (e.g. H100 80 GB).
  For a 24 GB card, set `IRIS_VLM_ID=Qwen/Qwen3-VL-8B-Instruct` (~16 GB); the rest
  of the pipeline fits comfortably.
- `--resume` (per-phase / per-object checkpointing) makes a run robust to crashes,
  so a failure costs one object rather than the whole run.

See [user_guide.md](user_guide.md) for usage and flags.
