# IRIS — Iterative Reconstruction via Incremental Scene-peeling

**Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments**

- **Problem Statement Number** — 9
- **Problem Statement Title** — Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments
- **Team name** — Team PIL
- **Team members** — Sparsh Pradeep Jain, Arghyadip Bagchi
- **Institute** — Indian Institute of Technology, Kanpur

---

## What IRIS does

A robot's sensors only see what's directly in front of them; everything occluded
becomes a blind spot that most maps silently collapse into "free space." IRIS
attacks this with a simple geometric guarantee — **the nearest object cannot be
occluded** — and peels the scene one object at a time, nearest first: detect the
front object, reconstruct it in 3D, then *erase* it from the image to reveal what
was behind, and repeat. The peeled images are **same-pose synthetic views** (the
camera never moves; each view just has fewer objects), fused into a complete,
semantically-labeled 3D reconstruction. Unobserved volume is explicitly flagged as
**occluded (unknown)** rather than falsely "free" — the free / occupied / occluded
distinction the problem requires.

```
RGB → VLM discovery (Qwen3-VL) → for each object, nearest-first:
        SAM3 segment · DepthAnythingV2 + occlusion-graph order · RORem erase
      → per-object image-to-3D (Amodal3R / TRELLIS)
      → VGGT scene recon on same-pose views → gravity-aligned register + fuse
      → instance-aware semantic labeling → Marching-Cubes mesh
      → free / occupied / occluded occupancy grid
```

See [docs/architecture.md](docs/architecture.md) for the full technical breakdown.

---

## Project Artefacts

- **Source code** — [src/](src/) — the self-contained orchestrator [`pipeline.py`](src/pipeline.py), the VLM discovery step it runs as a subprocess (`step0_vlm.py`), the occupancy module (`step10_occupancy.py`), and the per-backend image-to-3D workers (`*_worker.py`). Run via `python src/pipeline.py` (see [docs/user_guide.md](docs/user_guide.md)).
- **Documentation** — [docs/](docs/): [architecture](docs/architecture.md) · [installation](docs/installation.md) · [user guide](docs/user_guide.md) · [evaluation/KPIs](docs/kpis.md) · [ax.md](docs/ax.md) · [attribution](docs/attribution.md)

- **Models Used** (all open-weight, run locally):

  | Stage | Model |
  |-------|-------|
  | Object discovery (VLM) | [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) (8B variant also supported) |
  | Instance segmentation | [SAM 3](https://huggingface.co/facebook/sam3) |
  | Monocular depth (peel ordering) | [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) |
  | Object removal / inpainting | [RORem](https://github.com/leeruibin/RORem) on [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
  | Image-to-3D (occlusion-aware, default) | [Amodal3R](https://github.com/Sm0kyWu/Amodal3R) |
  | Image-to-3D (baseline) | [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) |
  | Multi-view scene reconstruction | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) |
  | Semantic labeling (background stuff) | [Mask2Former Swin-L ADE](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic) |

- **Models Published** — none (no new model trained; IRIS is a training-free composition of open-weight models).
- **Datasets Used** — [ScanNet / ScanNet++](http://www.scan-net.org/) (indoor RGB-D, GT 3D + semantics), [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [S3DIS](http://buildingparser.stanford.edu/dataset.html) — for evaluation only.
- **Datasets Published** — none.

---

## Final Presentation

_TODO: add public Google Drive link to the final presentation (PDF/slides)._

## Full Submission Demo Video

https://youtu.be/wY5HgBuFqpU

## Setup & Result Reproducibility Video

_TODO: add YouTube link (public/unlisted)._

---

## Attribution

IRIS is a **training-free orchestration** of open-source, open-weight models; the
novel contribution is the iterative occlusion-peeling pipeline, the occlusion-graph
peel ordering, the same-pose multi-view fusion + gravity-aligned registration, the
instance-aware semantic labeling, and the free/occupied/occluded occupancy output.
All upstream projects and their licenses are credited in
**[docs/attribution.md](docs/attribution.md)** — including models we evaluated but
did not keep. Agentic-development tooling and process are documented in
**[docs/ax.md](docs/ax.md)**.

## License

Released under the [MIT License](LICENSE).
