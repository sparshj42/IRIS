# IRIS — Iterative Reconstruction via Incremental Scene-peeling

**Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments**

- **Problem Statement Number** — 9
- **Problem Statement Title** — Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments
- **Team name** — Team PIL
- **Team members** — Sparsh Pradeep Jain, Arghyadip Bagchi
- **Institute** — Indian Institute of Technology, Kanpur

## What IRIS does

A robot's sensors only see what's directly in front of them; everything occluded
becomes a blind spot that most maps silently collapse into "free space." IRIS
attacks this with a simple geometric guarantee — **the nearest object can't be
occluded** — and peels the scene one object at a time, nearest first: detect the
front object, reconstruct it in 3D, then *erase* it from the image to reveal what
was behind, and repeat. The peeled images are **same-pose synthetic views** (the
camera never moves; each view just has fewer objects), fused into a complete,
semantically-labeled 3D reconstruction where occluded geometry is actively
recovered rather than left as holes.

```
RGB → VLM discovery → [peel: SAM3 segment · DepthAnythingV2 order · RORem remove]
    → TRELLIS per-object 3D → VGGT scene recon → ICP fusion
    → Mask2Former labeling → Marching-Cubes semantic mesh
```

## Documentation (`docs/`)

- [architecture.md](docs/architecture.md) — technical stack, pipeline, implementation
- [installation.md](docs/installation.md) — setup & model downloads
- [user_guide.md](docs/user_guide.md) — how to run, flags, staged execution
- [ax.md](docs/ax.md) — **[required]** open-weight models & agentic development workflow
- [attribution.md](docs/attribution.md) — upstream projects & what's original
- [kpis.md](docs/kpis.md) — evaluation plan & metrics

## Quick start

```bash
conda activate iris
python src/pipeline.py --image data/test3.png --output_dir output
# outputs: output/{synthetic_views, fused_pointcloud.ply,
#          labeled_pointcloud.ply, final_semantic_mesh.ply}
```

## Models used (all open-weight, run locally)

| Stage | Model |
|-------|-------|
| Object discovery | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Segmentation | [SAM 3](https://huggingface.co/facebook/sam3) |
| Depth | [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) |
| Object removal | [RORem](https://github.com/leeruibin/RORem) on [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| Image-to-3D | [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) |
| Multi-view recon | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) |
| Semantic labeling | [Mask2Former Swin-L ADE](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic) |

See [docs/attribution.md](docs/attribution.md) for models evaluated but not kept,
and for what is original to IRIS.

## Datasets

- **Used for evaluation:** ScanNet / ScanNet++ (indoor RGB-D with GT 3D + labels),
  NYU Depth V2 (depth), S3DIS (semantic labels). See [docs/kpis.md](docs/kpis.md).
- **Published:** none.

## Submission artefacts

- **Source code** — [src/](src/) (orchestrator `pipeline.py` + standalone `stepN_*.py`)
- **Models published** — none (no new model trained)
- **Final Presentation (PDF)** — _TODO: add public Google Drive link_
- **Demo Video** — _TODO: add YouTube link_
- **Setup & Reproducibility Video** — _TODO: add YouTube link_

## License

See [LICENSE](LICENSE).
