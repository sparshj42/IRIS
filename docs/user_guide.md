# User Guide

## Run the full pipeline

```bash
conda activate iris
cd IRIS
python src/pipeline.py --image data/test3.png --output_dir output
```

Inputs: a single RGB image. Outputs land in `--output_dir`:
- `synthetic_views/` — the peeled, same-pose views
- `fused_pointcloud.ply` — scene + registered object clouds
- `labeled_pointcloud.ply` — semantically colored cloud
- `final_semantic_mesh.ply` — final watertight semantic mesh

## Flags

The pipeline uses the models chosen by the A/B evaluations in [ax.md](ax.md):
**Qwen3-VL-32B** (discovery), **SAM 3** (segmentation), **Depth-Anything-V2**
(occlusion ordering), **RORem** (removal), **Amodal3R** (occlusion-aware
image-to-3D, default), **VGGT** (multi-view), **Mask2Former** (background labeling).
Image-to-3D is swappable via `--image3d`; the alternatives are documented in
[attribution.md](attribution.md) and [ax.md](ax.md).

| Flag | Default | Purpose |
|------|---------|---------|
| `--image` | `data/test.png` | input RGB image (single-frame, IRIS's main mode) |
| `--scene_dir` | – | folder of images (multi-view); overrides `--image` |
| `--output_dir` | `output` | where artifacts are written |
| `--image3d` | `trellis` | per-object 3D backend: `amodal3r` (occlusion-aware, recommended), `trellis`, `wonder3d`, `tigon`, `splattn` |
| `--sparse_depth` | – | `.npy` of (row,col,metric_depth) ~500 px; makes the **whole reconstruction metric** (scales VGGT via median depth ratio) for the `<2 cm` KPI |
| `--skip_3d` | off | skip per-object image-to-3D; fused recon = VGGT scene only |
| `--resume` | off | skip phases whose outputs already exist (crash recovery) |
| `--stop_after_peeling` | off | stop after the peel phase |

> Set `IRIS_VLM_ID=Qwen/Qwen3-VL-8B-Instruct` for a lighter (~16 GB) discovery VLM
> on smaller GPUs; the 32B default expects a large GPU.

## Staged / crash-safe execution

On an unstable machine, split the run so each stage is short and resumable:

```bash
# Stage 1 — peel (saves views + object crops), then stop
python src/pipeline.py --image data/test3.png --stop_after_peeling \
       --output_dir output --resume

# Stage 2 — image-to-3D → VGGT → fusion → labeling → mesh
python src/pipeline.py --image data/test3.png --output_dir output --image3d amodal3r --resume
```

`--resume` reuses any completed phase and continues per-object peeling from the
last checkpoint, so a crash costs at most the object in flight. (Not needed on a
stable/datacenter GPU.)

## Metric reconstruction (sparse depth)

The problem provides ~500 sparse depth points per image. Generate the input from a
dataset's dense depth and run metric:

```bash
python scripts/make_sparse_depth.py --depth gt_depth.npy --n 500 --out sparse.npy
python src/pipeline.py --image rgb.png --sparse_depth sparse.npy --output_dir output
```

Without `--sparse_depth` the reconstruction is up-to-scale (fine for the
occlusion/semantic outputs; required only for the metric `<2 cm` KPI).

## Standalone steps

Each stage is also a standalone script (`src/step0_vlm.py` … `src/step9_mesh.py`)
that reads/writes intermediate artifacts, useful for debugging a single stage.
