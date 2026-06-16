# IRIS — Video Scripts

Two YouTube videos are required (public or unlisted). Keep both tight.
Screen-record + voiceover; no editing polish needed.

---

## A) Full Submission Demo Video  (~3–4 min)

**Goal:** show the idea and the results. Audience = judges.

1. **(0:00) Hook + problem (30s).** "A camera only sees front surfaces — everything behind is a blind spot most 3D maps wrongly fill as free space." Show `showcase/01_peeling_sequence.png` input frame.
2. **(0:30) The insight (30s).** "The nearest object can't be occluded — so we peel front-to-back." Play through `01_peeling_sequence.png`: each object erased reveals what's behind. *This is the money shot.*
3. **(1:00) Pipeline (40s).** Show the pipeline diagram (Slide 5). One line per stage: discover → segment+order → erase → image-to-3D → VGGT fuse → label → occupancy.
4. **(1:40) Results (60s).**
   - Open `output_scannet0030/` reconstruction in a 3D viewer (rotate the objects).
   - Show `showcase/04_per_object_3d.png` (clean completed objects).
   - Show `showcase/03_occupancy.png` — "free / occupied / occluded — we flag the unknown."
5. **(2:40) Comparison (40s).** "Closest prior work Gen3DSR: we ran it head-to-head — IRIS is ~6× faster and degrades gracefully where its released code crashes, and we add the occupancy output."
6. **(3:20) Close (20s).** Contributions in one line + repo link.

---

## B) Setup & Result Reproducibility Video  (~4–6 min)

**Goal:** prove a grader can reproduce from a clean clone. Audience = reproducer.

1. **(0:00) Clone + env (60s).** `git clone …/IRIS`, then `bash scripts/setup_envs.sh` (or show `docs/installation.md`). Note: conda envs `iris` + per-backend envs; HF_TOKEN for gated SAM3.
2. **(1:00) Weights (30s).** `python scripts/fetch_weights.py` (RORem etc.); HF models auto-download on first use.
3. **(1:30) Run (90s).** Live-run on a provided image:
   ```
   HF_TOKEN=<token> conda run -n iris python src/pipeline.py \
       --image data/test3.png --output_dir output_demo --image3d amodal3r
   ```
   Talk through the phase prints as they appear (discovery → peel order → peeling → recon → labeling → occupancy).
4. **(3:00) Outputs (90s).** Open `output_demo/`:
   - `fused_pointcloud.ply` / `final_semantic_mesh.ply` in a viewer
   - `occupancy_render.png`
   - `synthetic_views/` (the peeled images)
5. **(4:30) Close (30s).** "Reproduced end-to-end on a single GPU." Mention `--resume`, `--skip_3d`, backend flags.

---

### Recording tips
- Use the **ScanNet scene0030 frame** for the demo (most dramatic peeling), and `data/test3.png` for the clean reproducibility run.
- For 3D viewing, MeshLab / the VSCode glTF viewer / online glTF viewer all open the `.ply`/`.glb`.
- Upload to YouTube (NOT Google Drive — guidelines forbid Drive for video). Paste links into README's three artifact sections.
