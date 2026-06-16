# AX: Open-Weight Models & Agentic Development

This document covers two things the AX track asks for: (1) the open-weight models
and OSS that make up IRIS, and (2) how we used agentic development tooling to
build it — including, honestly, what worked and what did not.

A note up front on scope: **IRIS is not itself an agentic system.** It is a fixed,
deterministic computer-vision pipeline — there is no LLM agent reasoning or
planning at *runtime*. So the parts of the AX prompt about runtime multi-agent
orchestration, MCP servers, or in-solution tool-chaining do not apply, and we
have not invented them. What *does* apply, and what we describe below, is (a) that
every model in the solution is open-weight and runs locally, and (b) that the
*development* of IRIS was a heavily agentic, experiment-driven process.

---

## 1. Open-weight models used (all run locally; no runtime API calls)

| Stage | Model | Hugging Face ID | License |
|-------|-------|-----------------|---------|
| Object discovery (VLM) | Qwen3-VL-32B-Instruct (8B also supported) | `Qwen/Qwen3-VL-32B-Instruct` | Apache-2.0 |
| Grounded segmentation | SAM 3 | `facebook/sam3` | SAM license (open weights) |
| Monocular depth | Depth Anything V2 (Large) | `depth-anything/Depth-Anything-V2-Large-hf` | Apache-2.0 |
| Object removal | RORem (SDXL-inpainting UNet) | base `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | OpenRAIL / open weights |
| Image-to-3D (default, occlusion-aware) | Amodal3R | `Sm0kyWu/Amodal3R` | open weights |
| Image-to-3D (alternatives) | TRELLIS · Wonder3D · TIGON | `microsoft/TRELLIS-image-large` etc. | MIT / open |
| Pose-free multi-view recon | VGGT-1B | `facebook/VGGT-1B` | open weights |
| 2D semantic segmentation | Mask2Former (Swin-L, ADE20K) | `facebook/mask2former-swin-large-ade-semantic` | open weights |
| Mesh extraction | Marching Cubes | (algorithm, via scikit-image) | — |

Models evaluated but **not** kept (documented under §3): TripoSR
(`stabilityai/TripoSR`), Grounding DINO + SAM2
(`IDEA-Research/grounding-dino-base`, `facebook/sam2-hiera-large`), LaMa
(`simple-lama-inpainting`), PowerPaint v2 (`JunhaoZhuang/PowerPaint-v2-1`),
Depth Anything 3 (`depth-anything/DA3-LARGE`).

**Runtime compliance:** the delivered pipeline calls no commercial/closed API. All
inference is local on the above open-weight checkpoints.

## 2. Key OSS libraries / projects

- **PyTorch** 2.5.1 (cu121) — core DL runtime
- **Hugging Face Transformers** 5.9 — VLM, SAM 3, Depth Anything, Mask2Former
- **Diffusers** 0.38 — SDXL-inpainting pipeline for RORem
- **TRELLIS** (microsoft/TRELLIS) — image-to-3D, gaussian decode path
- **VGGT** (facebook) — multi-view reconstruction
- **Open3D** 0.19, **trimesh** 4.12 — point clouds, ICP registration, meshing
- **scikit-image** 0.24 (Marching Cubes), **scikit-learn** 1.7 (KD-tree voting), **SciPy**, **OpenCV**
- **spconv-cu118**, **xformers** — TRELLIS sparse backbone

Attribution to the upstream projects we build on is in
[docs/attribution.md](attribution.md).

## 3. Agentic development workflow — what worked, what didn't

IRIS was built through an **agentic coding harness (Claude Code)** driving an
experiment-first loop: for every pipeline stage we wired the candidate model
behind a flag, ran it on a real test image, rendered the output, and **kept the
winner on evidence** rather than on the paper's claims. The harness chained file
edits, shell runs, GPU jobs, and image rendering, and carried project state across
many sessions via a persistent file-based memory.

### Tool use / workflow patterns that worked
- **Flag-gated A/B swaps.** Every stage kept its old model selectable
  (`--segmenter`, `--remover`, `--image3d`, `--recon`) so comparisons were
  one-command and nothing regressed silently.
- **Pinned-env subprocess workers.** Several SOTA models needed mutually
  incompatible dependencies (PowerPaint wants transformers 4.28 / diffusers 0.27;
  TRELLIS wants torch 2.4 / numpy<2; DA3 wants numpy<2). Rather than break the
  main `iris` env, each ran in its **own conda env behind a line-protocol
  subprocess worker** (`src/powerpaint_worker.py`, `src/trellis_worker.py`). This
  isolation pattern was the single most useful piece of engineering — it let us
  trial four extra models without dependency hell.
- **Crash-resilient staging.** The build machine (a desktop RTX 3090) suffered
  repeated *hard power-offs* under sustained GPU load. We mitigated with a GPU
  power cap, per-object peel checkpointing, and splitting the run into resumable
  stages, so a crash cost one object instead of the whole run.
- **Persistent memory.** Decisions (which model won and *why*) were written to a
  durable memory so later sessions didn't re-litigate settled questions.

### Evidence-based model decisions (the "what worked")
- **Segmentation:** SAM 3 replaced Grounding DINO + SAM2 — single text-promptable
  model, complete masks (it captured an object's cap that the two-step missed),
  no duplicate detections.
- **Removal:** RORem won a 3-way A/B over LaMa and PowerPaint. RORem cleanly
  erases to background; LaMa blurs large holes; the crop+feather compositing fix
  stopped cumulative blur across peels.
- **Image-to-3D:** TRELLIS clearly beat TripoSR (recognizable object geometry vs
  blobs) and *measurably* improved downstream registration (toolbox ICP fitness
  0.76 → 1.00).

### What did NOT work (honest negatives)
- **PowerPaint hallucinated replacement objects** even when set up correctly in
  its native pinned env (verified the task tokens loaded). On tight object-shaped
  masks it regenerates content instead of erasing — unsuitable for "scene
  revelation." A genuine dead end we kept documented rather than hidden.
- **Higher-resolution removal didn't fix the large-object "ghost."** Running RORem
  at 1024 for a frame-filling object left the same faint silhouette and slightly
  worse shadows — it's an inherent "no surrounding context" limit of inpainting,
  not a resolution problem. Reverted to 512.
- **Depth Anything 3 was evaluated and shelved.** Its scene cloud was marginally
  cleaner than VGGT but not decisively better, and not worth the extra env. A
  comparison we ran specifically so we could *stop* considering it.
- **The unobservable object backs.** IRIS's same-pose synthetic views mean no
  camera ever sees object backs, so image-to-3D must hallucinate them. We
  discussed depth-grounded alternatives and chose to keep TRELLIS, accepting that
  generative completion is the cost of full object shapes — a known limitation,
  stated plainly.

### Later developments (H100 pass)
With more compute we re-ran the same evidence-first loop on every stage:
- **VLM 8B → 32B.** A/B showed a *tie* on easy scenes but the 32B found ~2× the
  objects on a cluttered ScanNet frame (the 8B wastes its budget on duplicate
  detections), so 32B is the default; 8B stays selectable.
- **Peel order → occlusion graph.** Replaced the naive global depth sort with a
  pairwise occlusion graph (boundary-depth + physical-support) + topological sort.
  On a cluttered frame the old sort led the peel with a floor cable; the new one
  correctly leads with the foreground chair.
- **Occlusion-aware image-to-3D.** Added **Amodal3R** (consumes the occluder mask)
  as the default, plus **Wonder3D** (multi-view diffusion + visual-hull carve) —
  each as a pinned-env subprocess worker, the same isolation pattern as before.
- **Gravity-aligned output + instance semantics.** The recon is rotated so the
  estimated floor normal points up (fixing "tilted/floating" outputs in a viewer),
  and labels now come from the SAM3 instance masks + VLM names rather than a coarse
  closed taxonomy.

### What did NOT work, part 2 (baselines)
- **Reproducing Gen3DSR (3DV 2025), the closest related work, was instructive.**
  Its *released* code did **not** run to completion on a cluttered real ScanNet
  frame: it crashed at its own object-to-scene placement step (RANSAC scale-fit
  with no consensus, then zero depth-overlap) and on degenerate object meshes. It
  needed three robustness patches from us just to finish, and even then dropped ~⅓
  of objects. The lesson that shaped IRIS: **placing generated objects into a
  metric scene is the fragile, unsolved step for this whole divide-and-conquer
  paradigm** — Gen3DSR fails *loudly* (crash), IRIS degrades *gracefully* (still
  produces a scene). That graceful-degradation property is a deliberate design goal.

### What did not apply
We did not use multi-agent orchestration, MCP servers, or any agentic component
*inside* the solution, because IRIS does not need runtime agency — it is a
deterministic pipeline. We report this rather than dress it up.
