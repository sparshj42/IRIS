# Attribution

IRIS is an original pipeline that orchestrates several open-source models and
projects. We did not fork a single base project; we integrate the following
upstream works, with gratitude to their authors. Each is used under its own
license.

## Models in the final pipeline

| Project | Used for | Link |
|---------|----------|------|
| Qwen3-VL | object discovery (VLM) | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct |
| SAM 3 (Segment Anything 3) | grounded segmentation | https://huggingface.co/facebook/sam3 |
| Depth Anything V2 | monocular depth | https://github.com/DepthAnything/Depth-Anything-V2 |
| RORem | object removal | https://github.com/leeruibin/RORem |
| Stable Diffusion XL Inpainting | RORem base pipeline | https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 |
| TRELLIS | image-to-3D | https://github.com/microsoft/TRELLIS |
| VGGT | pose-free multi-view reconstruction | https://github.com/facebookresearch/vggt |
| Mask2Former | 2D semantic segmentation | https://huggingface.co/facebook/mask2former-swin-large-ade-semantic |

## Models evaluated during development (not in final pipeline)

| Project | Outcome | Link |
|---------|---------|------|
| TripoSR | image-to-3D, kept as light fallback (TRELLIS preferred) | https://github.com/VAST-AI-Research/TripoSR |
| Grounding DINO + SAM 2 | segmentation, replaced by SAM 3 | https://github.com/IDEA-Research/GroundingDINO |
| LaMa (simple-lama-inpainting) | removal, kept as low-power fallback | https://github.com/advimman/lama |
| PowerPaint v2 | removal, rejected (hallucinated) | https://github.com/open-mmlab/PowerPaint |
| Depth Anything 3 | multi-view, shelved (marginal vs VGGT) | https://github.com/ByteDance-Seed/depth-anything-3 |
| InstantMesh | image-to-3D, explored | https://github.com/TencentARC/InstantMesh |

## Key libraries

PyTorch, Hugging Face Transformers & Diffusers, Open3D, trimesh, scikit-image,
scikit-learn, SciPy, OpenCV, spconv, xformers. Full versions in
[installation.md](installation.md).

## What is original to IRIS

The **iterative depth-ordered occlusion peeling**, the **same-pose synthetic-view
formulation for multi-view reconstruction**, the **mask-guided ICP registration**
of single-image-3D objects into the VGGT scene, and the **end-to-end
orchestration** are our own contribution. See [architecture.md](architecture.md).
