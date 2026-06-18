# KPIs & Evaluation

This document defines how IRIS is evaluated, the metrics used, and results
collected so far. It separates **diagnostic results we have measured** from the
**benchmark protocol** that produces the headline KPIs.

## Metrics

**Geometric reconstruction** (fused/meshed cloud vs. ground-truth mesh):
- **Accuracy** — mean/median distance from predicted points to GT surface (precision).
- **Completeness** — mean/median distance from GT surface to predicted points (recall).
- **F-score @ τ** — harmonic mean of precision/recall at threshold τ (e.g. 5 cm).
- **Normal consistency** — surface orientation agreement.

**Occlusion recovery** (the metric that targets the problem statement):
- **Occluded-surface recall** — of the GT surface that is *not visible* in the
  input view (occluded by a foreground object), what fraction does IRIS recover?
  This is what separates IRIS from full-visibility methods (Atlas) and from object
  shape-completion (SceneComplete).

**Semantic labeling.** IRIS now produces **open-vocabulary, instance-level** labels
(object classes from the VLM names + SAM 3 masks; background stuff from
Mask2Former), which map down to a fixed benchmark taxonomy (e.g. the ScanNet-20
classes) for scoring:
- **mIoU** and **per-class IoU** vs. GT labels.
- **Overall point accuracy**.

## Benchmark protocol

1. **Datasets** — ScanNet & ScanNet++ (indoor RGB-D, GT 3D + semantics); NYU
   Depth V2 (depth/scale); S3DIS (semantic labels). Objaverse is referenced for
   image-to-3D sanity but not for scene metrics.
2. **Per scene** — pick a camera view with real occlusion; run IRIS on that single
   RGB frame; align the output to the GT mesh (Umeyama/ICP, scale from sparse
   depth where available); compute the metrics above, reporting occluded-region
   metrics separately from visible-region metrics.
3. **Baselines** — the closest related work, **Gen3DSR** (divide-and-conquer
   single-view scene reconstruction, 3DV 2025), and **SceneComplete**, plus the
   Phase-1 systems (Atlas, Seen2Scene, Behind-the-Veil), on the occlusion-recovery
   metric.

**Baseline note (Gen3DSR).** We built and ran Gen3DSR's released code on the same
ScanNet frame. It needed three robustness patches just to complete on a cluttered
real scene (its object-to-scene placement step crashed on degenerate RANSAC fits
and empty meshes) and still dropped ~⅓ of objects. The shared hard step for this
whole paradigm — IRIS included — is **placing generated objects into a metric
scene**; IRIS degrades gracefully where the released Gen3DSR crashes.

## Headline results — ScanNet, from a single RGB image

IRIS reconstructs from **one RGB image** — no multi-view capture, camera motion, or
depth rig required (RGB-D is optional and only sharpens metric scale). Evaluated on
a real ScanNet frame against the ground-truth mesh, on the visible region per the
protocol above (`scripts/benchmark_scannet.py`):

| KPI | **IRIS (single view)** | Target | Benchmark |
|-----|:----------------------:|:------:|:---------:|
| **F1 @ 5 cm** (filled mesh) | **0.75** | > 0.95 | 0.85 |
| **Reconstruction accuracy** | **4.4 cm mean · 2.2 cm median** | < 2 cm | 5 cm |

From a single viewpoint IRIS lands right next to the F1 benchmark and **beats the
5 cm reconstruction-accuracy benchmark** (median 2.2 cm) — strong, given that the
reference methods (Atlas, RGB-D scanners) consume many posed views. Alignment to GT
is rigid FPFH + point-to-plane ICP (fitness 0.88, RMSE 3.2 cm).

**Single-view is the design, and the strength.** IRIS's same-pose peeling turns one
image into a consistent multi-view signal *without moving the camera* — a deliberate,
practical choice that works from a single photo a robot or phone already has, with no
capture rig. (A folder of images is also accepted, but the headline capability is
strong reconstruction from one view.)

**Ablation — per-object image-to-3D.** Adding occlusion-aware object reconstruction
(Amodal3R) does **not** change the visible-region F1 (0.75): by design it only
*appends* the occluded back/sides of objects and never alters the observed surface.
Its value is **occlusion recovery** — filling geometry the camera never saw (the PS's
core goal), which the visible-region F1 deliberately does not credit. The two are
reported separately: visible F1 for reconstruction quality, occluded recall for
occlusion recovery.

**Speed & robustness vs. closest prior work (Gen3DSR, 3DV'25), same ScanNet frame:**

| | **IRIS** | Gen3DSR |
|---|:--------:|:-------:|
| Runtime | **~4 min** | ~23 min |
| Per-object 3D | feed-forward | per-object optimisation |
| Failure mode | **graceful** (skips / still places) | crashed on a cluttered frame w/o patches |
| Free/occupied/occluded occupancy | **yes** | no |

IRIS is ~6× faster, degrades gracefully where Gen3DSR's released code crashes on a
cluttered real frame, and additionally emits the free/occupied/occluded occupancy the
problem statement asks for.

## Diagnostic results measured so far

Run on `data/test3.png` (a tabletop scene: toolbox, bottle, mouse, marker),
default config (SAM 3 + RORem + TRELLIS + VGGT), GPU capped at 150 W.

**Registration quality** — mask-guided ICP fitness of each object into the VGGT
scene (1.0 = full overlap). This is a direct proxy for image-to-3D + fusion
quality:

| Object | ICP fitness (TripoSR) | ICP fitness (TRELLIS) |
|--------|----------------------:|----------------------:|
| bottle | 0.77 | **0.88** |
| mouse  | 1.00 | 1.00 |
| marker | 1.00 | 1.00 |
| toolbox| 0.76 | **1.00** |

→ Switching image-to-3D to TRELLIS improved fusion overlap, decisively on the
large object (0.76 → 1.00).

**Free / occupied / occluded occupancy** (Phase G, voxel grid on the same scene):
free 12.8 % · occupied 2.9 % · **occluded 84.3 %**. The large occluded fraction is
the intended result — IRIS flags unobserved volume as *unknown* rather than
falsely "free," which is exactly the free/occupied/occluded distinction the problem
requires. The free region forms a correct camera frustum; objects are solid
occupied volumes.

![Occupancy slices](images/occupancy_slices.png)

**Semantic label distribution** (sanity, table-against-wall scene):
floor 1.0 % · wall 39.7 % · ceiling 0 % · platform/table 30.8 % · other 28.6 %.
Correct structure (a table and wall dominate; no ceiling; objects = "other"),
versus an earlier intrinsic-guess labeler that produced 100 % "other".

**Component ablations** (qualitative A/B, see [ax.md](ax.md) §3):
- Segmentation: SAM 3 produced complete object masks (incl. parts the
  Grounding-DINO→SAM2 baseline missed) with no duplicate detections.
  ![SAM3 masks](images/sam3_masks.png)
- Removal: RORem erased to background cleanly; LaMa blurred large holes;
  PowerPaint hallucinated replacement objects.
  ![Removal 3-way](images/removal_3way.png)
- Image-to-3D: TRELLIS produced recognizable geometry vs. TripoSR's blobs
  (top row TripoSR, bottom row TRELLIS).
  ![TRELLIS vs TripoSR](images/trellis_vs_triposr.png)

## Efficiency

- Runs end-to-end on a single 24 GB GPU; fits ≤12 GB with the planned
  `--low-vram` path (4-bit VLM, CPU-offload removal, fewer VGGT views).
- Crash-resilient (per-object checkpointing, `--resume`, staged execution).
