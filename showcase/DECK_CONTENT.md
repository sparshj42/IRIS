# IRIS — Final Presentation (slide-by-slide content)

Drop each slide's content into your deck template. Visuals referenced are in `showcase/`.
Map to rubric: **Technical 30 · Innovation 25 · Feasibility 20 · Alignment 15 · Documentation 10**.

---

### Slide 1 — Title
**IRIS — Iterative Reconstruction via Incremental Scene-peeling**
Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments
Problem Statement #9 · Team PIL · Sparsh Pradeep Jain, Arghyadip Bagchi · IIT Kanpur

### Slide 2 — The problem
- A camera/robot only sees front surfaces; everything behind an object is a **blind spot**.
- Most 3D maps silently collapse those blind spots into "free space" — dangerous for navigation/manipulation.
- Goal: from a **single RGB image**, recover a **complete, semantically-labeled 3D scene**, and explicitly mark what is **occupied / free / unknown(occluded)**.

### Slide 3 — Key insight
> **The nearest object cannot be occluded.**
- So reconstruct **front-to-back**: detect the nearest object, reconstruct it, **erase it** from the image to reveal what was behind, and repeat.
- The erased images are **same-pose synthetic views** — the camera never moves; each view just has fewer objects. No camera motion needed.

### Slide 4 — The mechanism (HERO)
- Visual: `showcase/01_peeling_sequence.png`
- Walk through: input → erase chair → erase book → erase table … each removal reveals previously-occluded floor/shelf via RORem inpainting.
- This is the core novelty: turning one image into a consistent multi-view signal by *peeling*, not by moving the camera.

### Slide 5 — Pipeline
```
RGB → VLM discovery (Qwen3-VL-32B)
    → per object, nearest-first: SAM3 segment · DepthAnythingV2 + occlusion-graph order · RORem erase
    → per-object image-to-3D (Amodal3R / TRELLIS / Wonder3D)
    → VGGT multi-view recon on same-pose views → gravity-aligned register + fuse
    → instance-aware semantic labeling → marching-cubes mesh
    → FREE / OCCUPIED / OCCLUDED occupancy grid
```
- Training-free composition of open-weight models; modular — each stage swappable.

### Slide 6 — Output that targets the PS (HERO)
- Visual: `showcase/03_occupancy.png` (free=green · occupied=red · occluded/unknown=blue)
- This is the deliverable the problem asks for: unobserved volume flagged **unknown**, not falsely free.

### Slide 7 — Results
- **10-scene ScanNet benchmark (single RGB image each):** mean visible-surface **F1 0.87** (reaching **0.95**) and **median reconstruction accuracy 1.8 cm — clears both the 0.85 F1 benchmark and the 2 cm accuracy target**. Pose-anchored alignment (input camera pose; GT used only for scoring). Per-scene results listed in `docs/kpis.md`.
- **Best scenes are benchmark-grade from one view:** scene0250 / scene0400 F1 **0.95**, scene0300 0.93, scene0100 0.91, with 1.0–1.7 cm median accuracy.
- Per-object reconstruction: `showcase/04_per_object_3d.png` (recognizable bottle/toolbox/mouse from occlusion-aware image-to-3D).
- Runs end-to-end on real **ScanNet** frames and tabletop scenes on a single GPU.
- (Be honest: full-scene fusion is a single-view frustum; objects are completed, background is partial.)

### Slide 8 — Innovation (contributions)
1. **Iterative occlusion peeling** with the nearest-cannot-be-occluded guarantee → same-pose multi-view signal.
2. **Occlusion-graph peel ordering** (boundary-depth + support cues, topologically sorted) vs naive global depth sort.
3. **Instance-aware semantic labeling** from SAM3 masks + VLM open-vocab names (not a closed coarse taxonomy).
4. **Free/occupied/occluded occupancy** output for downstream robotics.
5. **Gravity-aligned, multi-backend** (Amodal3R / TRELLIS / Wonder3D) image-to-3D.

### Slide 9 — Comparison to prior work
- Closest: **Gen3DSR** (3DV'25, divide-and-conquer single-view) and **SceneComplete** (single RGB-D).
- We ran Gen3DSR head-to-head on the same ScanNet frame:
  - IRIS ~**4 min** vs Gen3DSR ~**23 min** (feed-forward generation vs per-object optimization).
  - Gen3DSR's released code **crashed** on a cluttered real frame (needed robustness patches); IRIS **degrades gracefully** (skips/places, never crashes).
  - IRIS additionally emits the **occupancy/occluded-volume** output neither targets.

### Slide 10 — Feasibility & practicality
- **Training-free** — no fine-tuning; works on diverse scenes out of the box.
- Single GPU; crash-resilient (`--resume`, per-object checkpointing); modular backends.
- All models **open-weight**; reproducible from a clean clone (see `docs/installation.md`).

### Slide 11 — Closing
- IRIS = a practical, training-free system that recovers occluded geometry from one image and exposes the unknown.
- Repo · demo video · reproducibility video links.
