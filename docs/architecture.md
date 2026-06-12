# IRIS — Technical Architecture

**IRIS: Iterative Reconstruction via Incremental Scene-peeling**
Occlusion-aware 3D scene reconstruction in partially observable environments.

## Core idea

The closest object to the camera at any instant cannot be occluded — it must be
fully visible. IRIS exploits this geometric guarantee to **peel the scene one
object at a time, nearest first**: detect the front object, reconstruct it,
*remove* it from the image to reveal what was behind, and repeat. The sequence of
"peeled" images are **same-pose synthetic views** — the camera never moves, but
each view shows fewer objects, effectively seeing *through* the scene from a fixed
viewpoint. These are fused into a complete, semantically labeled 3D reconstruction
in which occluded geometry is actively recovered rather than left as holes.

## Pipeline (data flow)

```
RGB image
   │
   ▼
[A] Object discovery ............ Qwen3-VL  → list of object names (JSON)
   │
   ▼
[B] Per-object peel loop (nearest → farthest), per iteration:
     • segment object on current image .... SAM 3            → mask
     • order by nearest-point depth ....... Depth Anything V2
     • save white-bg object crop          (for phase B2)
     • remove object → reveal background .. RORem (SDXL inpaint, crop@512)
     • store result as a synthetic view
   │
   ▼
[B2] Image-to-3D (deferred) ...... TRELLIS  → per-object point cloud
   │
   ▼
[C] Multi-view reconstruction .... VGGT     → scene point cloud + per-pixel
   │                                           world points & confidence
   ▼
[D] Registration + fusion ........ mask-guided ICP (Open3D) → fused cloud
   │
   ▼
[E] Semantic labeling ............ Mask2Former + multi-view KD-tree voting
   │                                (labels via VGGT world points)
   ▼
[F] Mesh extraction .............. Marching Cubes (scikit-image) → semantic mesh
   │
   ▼
[G] Occupancy classification ..... ray-cast voxel grid → FREE / OCCUPIED / OCCLUDED
```

Implementation: a single orchestrator [src/pipeline.py](../src/pipeline.py); each
stage is also runnable standalone as `src/stepN_*.py`.

## Stage details

**[A] Object discovery.** Qwen3-VL-8B is prompted (descriptive, singular, JSON
output) for a list of distinct movable objects. Run in a subprocess so its ~16 GB
of VRAM is fully released before the multi-model peel phase.

**[B] Depth ordering + peeling.** SAM 3 segments each named object
(text-promptable; one model replaces the older Grounding-DINO→SAM2 two-step).
Depth Anything V2 gives a relative depth map; each object's *nearest* point
(max disparity over its mask) sets the peel order, with a support-aware
reordering so objects resting on a surface are peeled before the surface. Removal
uses RORem on a padded square **crop** around the mask (512²), feather-composited
back so only masked pixels change — no cumulative blur across peels.

**[B2] Image-to-3D (deferred).** Each saved object crop → TRELLIS → a point cloud
(from TRELLIS's 3D-Gaussian positions). Deferred to its own phase, after the peel
models are freed, so the ~8–10 GB TRELLIS worker has the GPU to itself.

**[C] Multi-view reconstruction.** All synthetic views go to VGGT, which returns
per-view per-pixel world points + confidence. Confidence-thresholded points form
the scene cloud. (Novel use: same-pose views with progressively fewer objects,
rather than the multi-position views these models expect.)

**[D] Registration + fusion.** TRELLIS objects live in their own canonical frame.
For each object we take the VGGT world points under its mask (where it sits in the
scene), seed a scale+centroid alignment, and refine with point-to-point ICP. This
mask-guided initialization is why fitness reaches 0.88–1.0 (a naive
identity-initialized ICP failed at 0.0).

**[E] Semantic labeling.** Mask2Former produces a 2D ADE20K segmentation per view,
mapped to IRIS classes (floor / wall / ceiling / platform / other). Because VGGT
already gives each pixel a world point, we label those points directly and vote
onto the fused cloud with a KD-tree — avoiding the camera-intrinsic guesswork that
made an earlier version label everything "other."

**[F] Mesh extraction.** The labeled cloud is voxelized into an occupancy grid and
Marching Cubes extracts a watertight mesh; vertex labels/colors are transferred
from the nearest labeled point.

**[G] Occupancy classification.** A voxel grid is ray-cast from the camera through
VGGT's per-pixel world points: voxels a ray traverses before a surface are **free**,
the surface voxel is **occupied**, voxels behind it are **occluded**. Because the
synthetic views are same-pose, each peeled view re-casts the same rays but reaches
deeper, so space behind a removed object is re-marked free/occupied — *peeling
resolves occlusion*. Reconstructed objects are filled to solid **occupied** volumes
(convex-hull fill). Whatever remains unobserved is **occluded** — the honest
"unknown" that distinguishes IRIS from mappers that assume free space.

![Occupancy slices](images/occupancy_slices.png)

## Engineering notes

- **Multi-env isolation.** PowerPaint and TRELLIS need dependencies incompatible
  with the main env, so they run in dedicated conda envs behind subprocess workers
  (see [ax.md](ax.md) §3). The main pipeline stays on one consistent stack.
- **VRAM budgeting.** Models load and free per phase; heavy models (VLM, TRELLIS)
  run isolated. Peak fits 24 GB; a `--low-vram` path (4-bit VLM, CPU-offload
  removal, fewer VGGT views) is the route to ≤12 GB.
- **Crash resilience.** Per-object peel checkpointing + `--resume` + staged
  execution make the run robust to the build machine's power instability.

## Result

Final labeled reconstruction (tabletop scene): wall (gray), table/platform (blue),
objects (green) registered onto the table.

![Final labeled reconstruction](images/final_reconstruction.png)

## Outputs (`output/`)

- `synthetic_views/` — the peeled same-pose views
- `object_crops/` — per-object inputs to TRELLIS
- `fused_pointcloud.ply` — scene + registered objects
- `labeled_pointcloud.ply` — semantically colored cloud
- `final_semantic_mesh.ply` — final watertight semantic mesh
