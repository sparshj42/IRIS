# Attribution

IRIS is an original pipeline that orchestrates several open-source models and
projects. We did not fork a single base project; we integrate the following
upstream works, with gratitude to their authors. Each is used under its own
license.

## Models in the final pipeline

| Project | Used for | Link |
|---------|----------|------|
| Qwen3-VL | object discovery (VLM) | https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct |
| SAM 3 (Segment Anything 3) | grounded segmentation | https://huggingface.co/facebook/sam3 |
| Depth Anything V2 | monocular depth | https://github.com/DepthAnything/Depth-Anything-V2 |
| RORem | object removal | https://github.com/leeruibin/RORem |
| Stable Diffusion XL Inpainting | RORem base pipeline | https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 |
| Amodal3R | occlusion-aware image-to-3D (default backend; built on TRELLIS's gaussian decoder) | https://github.com/Sm0kyWu/Amodal3R |
| TRELLIS | image-to-3D (selectable baseline backend) | https://github.com/microsoft/TRELLIS |
| VGGT | pose-free multi-view reconstruction | https://github.com/facebookresearch/vggt |
| Mask2Former | 2D semantic segmentation (background "stuff" labels) | https://huggingface.co/facebook/mask2former-swin-large-ade-semantic |

## Models evaluated during development (not in final pipeline)

| Project | Outcome | Link |
|---------|---------|------|
| Wonder3D | image-to-3D via cross-domain multi-view diffusion, explored (not kept) | https://github.com/xxlong0/Wonder3D |
| TIGON | text+image image-to-3D, explored (not kept) | https://jumpat.github.io/tigon-page/ |
| TripoSR | image-to-3D, explored (TRELLIS preferred) | https://github.com/VAST-AI-Research/TripoSR |
| Grounding DINO + SAM 2 | segmentation, replaced by SAM 3 | https://github.com/IDEA-Research/GroundingDINO |
| LaMa (simple-lama-inpainting) | removal, explored (RORem preferred) | https://github.com/advimman/lama |
| PowerPaint v2 | removal, rejected (hallucinated) | https://github.com/open-mmlab/PowerPaint |
| Depth Anything 3 | multi-view, shelved (marginal vs VGGT) | https://github.com/ByteDance-Seed/depth-anything-3 |
| InstantMesh | image-to-3D, explored | https://github.com/TencentARC/InstantMesh |
| SplAttN | image-guided point-cloud completion, explored (fills viewpoint not scene occlusion) | https://github.com/zay002/SplAttN |
| GenPC | point-cloud completion, explored | https://github.com/Sangminhong/GenPC |

## Baselines compared against

| Project | Use | Link |
|---------|-----|------|
| Gen3DSR | divide-and-conquer single-view scene reconstruction (3DV 2025) — closest related work; benchmarked head-to-head | https://github.com/AndreeaDogaru/Gen3DSR |
| SceneComplete | open-world single-RGB-D scene completion for manipulation — related work | https://github.com/scenecomplete/SceneComplete |

## Key libraries

PyTorch, Hugging Face Transformers & Diffusers, Open3D, trimesh, scikit-image,
scikit-learn, SciPy, OpenCV, spconv, xformers. Full versions in
[installation.md](installation.md).

## What is original to IRIS

The following are our own contribution (see [architecture.md](architecture.md)):

- **Iterative depth-ordered occlusion peeling** with the *nearest-object-cannot-be-occluded* guarantee, and the **same-pose synthetic-view formulation** for multi-view reconstruction.
- **Occlusion-graph peel ordering** — a pairwise occlusion graph built from boundary depth + physical-support cues, topologically sorted (replacing a naive global depth sort).
- **Gravity-aligned registration & fusion** — gravity-align + yaw-search + asymmetric weld ICP + floor-contact + object-relative 3D-inpainting graft of occluded geometry, with the whole reconstruction rotated to a gravity-aligned output frame.
- **Instance-aware semantic labeling** — labels propagated from the SAM3 instance masks + VLM open-vocabulary names (background "stuff" from Mask2Former), instead of a closed coarse taxonomy.
- **Free / occupied / occluded occupancy** output and the **end-to-end orchestration**.
