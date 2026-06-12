"""
IRIS: Iterative Reconstruction via Incremental Scene-peeling
  Phase A : Object Discovery            (Qwen3-VL)
  Phase B : Depth ordering + iterative occlusion peeling
              - re-evaluate mask on current image state  (SAM3)
              - depth ordering                           (DepthAnything V2)
              - object removal revealing what's behind   (RORem)
  Phase B2: Image-to-3D (deferred)      (TRELLIS; skip with --skip_3d)
  Phase C : Scene Reconstruction        (VGGT on same-pose synthetic views)
  Phase D : Rigid Registration + Fusion (mask-guided init + ICP, Open3D)
  Phase E : Semantic Labeling           (Mask2Former, multi-view voting)
  Phase F : Mesh Generation             (Marching Cubes)
  Phase G : Occupancy                   (free / occupied / occluded ray-cast)

Models are loaded/freed in phases. TRELLIS runs in a pinned-env subprocess worker.
"""

import argparse
import gc
import os
import pickle
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src/ on path
import config

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="IRIS pipeline")
parser.add_argument("--image", default="data/test.png", help="input RGB image")
parser.add_argument("--sparse_depth", default=None,
                    help="optional .npy of (row, col, metric_depth) rows for scale recovery")
parser.add_argument("--output_dir", default="output")
parser.add_argument("--resume", action="store_true",
                    help="skip phases whose outputs already exist (crash recovery)")
parser.add_argument("--skip_3d", action="store_true",
                    help="skip per-object image-to-3D (TRELLIS); fused recon = VGGT scene")
parser.add_argument("--stop_after_peeling", action="store_true",
                    help="exit after Phase B (for fast removal A/B comparison)")
args = parser.parse_args()

# per-output-dir so parallel runs (different --output_dir) don't collide
RECORDS_PATH = os.path.join(args.output_dir, "object_records.pkl")

IMAGE_PATH = args.image
OUTPUT_DIR = args.output_dir
VIEWS_DIR = os.path.join(OUTPUT_DIR, "synthetic_views")
MESH_DIR = os.path.join(OUTPUT_DIR, "meshes")
CROPS_DIR = os.path.join(OUTPUT_DIR, "object_crops")   # for the deferred TRELLIS phase
os.makedirs(VIEWS_DIR, exist_ok=True)
os.makedirs(MESH_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

ROREM_CKPT = config.ROREM_CKPT
SDXL_BASE = config.SDXL_BASE
ROREM_PROMPT = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
ROREM_NEG_PROMPT = "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background"

VGGT_CONF_THRESHOLD = 0.5

IRIS_LABEL_TO_ID = {"floor": 0, "wall": 1, "ceiling": 2, "platform": 3, "other": 4}
ADE20K_TO_IRIS = {
    3: "floor", 28: "floor", 54: "floor",
    0: "wall", 8: "wall",
    5: "ceiling",
    14: "platform", 15: "platform", 33: "platform",
}
LABEL_COLORS = {
    0: [0.6, 0.4, 0.2],   # floor - brown
    1: [0.8, 0.8, 0.8],   # wall - gray
    2: [0.9, 0.9, 0.7],   # ceiling - light yellow
    3: [0.2, 0.6, 0.9],   # platform - blue
    4: [0.2, 0.8, 0.2],   # other - green
}


def free_cuda(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def metric_scale_from_sparse(world_points_v0, extr_v0, sparse_path, orig_hw):
    """Metric scale for the (arbitrary-scale) VGGT reconstruction from the problem's
    sparse depth (~500 px). sparse_path = .npy of (row, col, metric_depth) in the
    ORIGINAL image. Returns s = median(metric_depth / VGGT_camera_depth) at those
    pixels; multiply world points + camera translations by s to make recon metric."""
    sparse = np.load(sparse_path)
    H0, W0 = orig_hw
    Hv, Wv = world_points_v0.shape[:2]
    rows = np.clip((sparse[:, 0] * Hv / H0).astype(int), 0, Hv - 1)
    cols = np.clip((sparse[:, 1] * Wv / W0).astype(int), 0, Wv - 1)
    z_metric = sparse[:, 2]
    R, t = extr_v0[:, :3], extr_v0[:, 3]
    P = world_points_v0[rows, cols]                       # (N,3) world points
    z_vggt = (P @ R.T + t)[:, 2]                           # VGGT-scale camera depth
    valid = (z_vggt > 1e-6) & np.isfinite(z_vggt) & (z_metric > 0)
    return float(np.median(z_metric[valid] / z_vggt[valid]))


class TrellisWorker:
    """Drives TRELLIS image->3D in the pinned `trellis` conda env (torch 2.4/cu118,
    spconv, xformers) via a subprocess. Input: a white-bg object crop PNG. Output:
    a point cloud (TRELLIS gaussian xyz)."""

    def __init__(self):
        import subprocess
        import tempfile
        pybin = config.conda_env_python("trellis")
        self.tmp = tempfile.mkdtemp(prefix="trellisworker_")
        self.log = open(os.path.join(self.tmp, "worker.log"), "w")
        env = dict(os.environ, ATTN_BACKEND="xformers", SPCONV_ALGO="native")
        self.proc = subprocess.Popen(
            [pybin, "-u", "src/trellis_worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env,
        )
        for line in self.proc.stdout:
            if line.strip() == "@@READY":
                break
            if self.proc.poll() is not None:
                raise RuntimeError("TRELLIS worker exited before READY; see " + self.log.name)

    def pointcloud(self, crop_path: str, n: int = 10000) -> np.ndarray:
        import json
        out = os.path.join(self.tmp, "pc.npy")
        self.proc.stdin.write(json.dumps({"image": crop_path, "out": out, "n": n}) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("@@OK"):
                return np.load(out)
            if line.startswith("@@ERR"):
                raise RuntimeError("TRELLIS worker: " + line)
            if self.proc.poll() is not None:
                raise RuntimeError("TRELLIS worker died; see " + self.log.name)
        raise RuntimeError("TRELLIS worker stdout closed unexpectedly")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


image = Image.open(IMAGE_PATH).convert("RGB")
W, H = image.size
print(f"Image loaded: {image.size}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE A: OBJECT DISCOVERY (VLM)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase A] Object discovery (Qwen3-VL)")
print("=" * 60)

# The 8B VLM is run in its own subprocess so the OS fully reclaims its ~16 GB
# of VRAM on exit (device_map="auto" leaves accelerate hooks that don't free
# cleanly in-process, which OOMs the multi-model Phase B that follows).
if not (args.resume and os.path.exists("detected_objects.txt")):
    import subprocess
    step0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step0_vlm.py")
    subprocess.run([sys.executable, step0, "--image", IMAGE_PATH,
                    "--out", "detected_objects.txt"], check=True)

with open("detected_objects.txt") as f:
    object_names = [l.strip() for l in f if l.strip()]
print(f"Object types: {object_names}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B: DEPTH ORDERING + ITERATIVE OCCLUSION PEELING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase B] Depth ordering + iterative occlusion peeling")
print("=" * 60)

RESUME_B = args.resume and os.path.exists(RECORDS_PATH)

if not RESUME_B:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # Segmentation: SAM3 (text-promptable, one model)
    from transformers import Sam3Model, Sam3Processor
    sam3_model = Sam3Model.from_pretrained(
        "facebook/sam3", torch_dtype=torch.bfloat16).to("cuda").eval()
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf", torch_dtype=torch.float16).to("cuda").eval()

    # Removal: RORem (SDXL-inpainting UNet fine-tuned for object removal)
    from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
    rorem_pipe = AutoPipelineForInpainting.from_pretrained(SDXL_BASE, torch_dtype=torch.float16)
    rorem_unet = UNet2DConditionModel.from_pretrained(ROREM_CKPT).to("cuda", dtype=torch.float16)
    rorem_pipe.unet = rorem_unet
    rorem_pipe = rorem_pipe.to("cuda")
    print("Loaded SAM3 + DepthAnything + RORem")


def segment(img: Image.Image, prompt_names: list) -> list:
    """SAM3: one forward per text concept → all instances of that concept, with
    complete masks. Returns a list of per-instance dicts {label, score, mask}."""
    h, w = np.array(img).shape[:2]
    instances = []
    for name in prompt_names:
        inputs = sam3_processor(images=img, text=name, return_tensors="pt")
        inputs = {k: (v.to(device="cuda", dtype=torch.bfloat16) if torch.is_floating_point(v)
                      else v.to("cuda")) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sam3_model(**inputs)
        res = sam3_processor.post_process_instance_segmentation(
            outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        masks, scores = res.get("masks"), res.get("scores")
        if masks is None:
            continue
        for m, s in zip(masks, scores):
            mb = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(np.float32)
            instances.append({"label": name, "score": float(s), "mask": mb})
    return instances


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


def consolidate_instances(instances: list, img_area: int) -> list:
    """Tame instance explosion from DINO:
    - dedupe same-label detections that overlap heavily (IoU > 0.5, keep best score)
    - merge same-label instances into one union mask when they are individually
      small (e.g. 10 toy blocks → one 'blocks' peel); keep large ones separate
    - drop speck instances (< 0.05% of the image)"""
    MERGE_AREA_FRAC = 0.02
    MIN_AREA_FRAC = 0.0005

    by_label = {}
    for inst in instances:
        by_label.setdefault(inst["label"], []).append(inst)

    result = []
    for label, group in by_label.items():
        # dedupe overlapping duplicates
        group = sorted(group, key=lambda x: -x["score"])
        kept = []
        for inst in group:
            if all(mask_iou(inst["mask"], k["mask"]) <= 0.5 for k in kept):
                kept.append(inst)

        areas = [float((i["mask"] > 0.5).sum()) for i in kept]
        small = [i for i, a in zip(kept, areas) if a < MERGE_AREA_FRAC * img_area]
        large = [i for i, a in zip(kept, areas) if a >= MERGE_AREA_FRAC * img_area]

        # Large instances are always kept individually (e.g. the two chairs)
        result.extend(large)

        # Merge small same-label instances only when SPATIALLY CLUSTERED
        # (connected components of their combined, dilated mask). Toys piled on a
        # table merge into one peel; toys across the room on a shelf stay separate.
        if small:
            union = np.zeros_like(kept[0]["mask"], dtype=np.uint8)
            for i in small:
                union |= (i["mask"] > 0.5).astype(np.uint8)
            link = cv2.dilate(union, np.ones((25, 25), np.uint8), iterations=2)
            n_comp, comp = cv2.connectedComponents(link)
            groups = 0
            for c in range(1, n_comp):
                region = comp == c
                grp = [i for i in small if (region & (i["mask"] > 0.5)).any()]
                if not grp:
                    continue
                gmask = np.zeros_like(kept[0]["mask"])
                for i in grp:
                    gmask = np.maximum(gmask, i["mask"])
                if (gmask > 0.5).sum() >= MIN_AREA_FRAC * img_area:
                    result.append({"label": label, "score": max(i["score"] for i in grp),
                                   "mask": gmask, "merged": len(grp) > 1})
                    groups += 1
            if len(small) > 1:
                print(f"  '{label}': {len(small)} small instances -> {groups} clustered group(s)")
    return result


def get_depth_map(img: Image.Image) -> np.ndarray:
    inputs = depth_processor(images=img, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = depth_model(**inputs)
    d = out.predicted_depth.squeeze().cpu().float().numpy()
    return cv2.resize(d.astype(np.float32), img.size, interpolation=cv2.INTER_LINEAR)


def recover_metric_scale(depth_rel: np.ndarray, sparse_path: str) -> np.ndarray:
    """Least-squares fit of scale/shift aligning relative depth to sparse metric depth.
    sparse file: .npy array of shape (N, 3) with rows (row, col, metric_depth)."""
    sparse = np.load(sparse_path)
    rows = sparse[:, 0].astype(int)
    cols = sparse[:, 1].astype(int)
    z = sparse[:, 2]
    d = depth_rel[rows, cols]
    A = np.stack([d, np.ones_like(d)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, z, rcond=None)
    print(f"  Scale recovery: scale={s:.4f}, shift={t:.4f} from {len(z)} sparse points")
    return s * depth_rel + t


def remove_object(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """image + mask → image with object removed.

    Inpainting runs on a padded square CROP around the mask (not the whole frame),
    so the model sees the region at high effective resolution. Only the masked
    pixels are feather-composited back, so the rest of the image is untouched —
    no cumulative blur across peels. The crop is filled by RORem (SDXL-inpainting
    UNet fine-tuned for object removal)."""
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((25, 25), np.uint8)          # generous dilation to catch soft shadows
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

    # Square crop around the mask with 35% context padding, clamped to the image
    ys, xs = np.where(mask_uint8 > 0)
    if len(ys) == 0:
        return img
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2
    half = min(max(half * 1.35, 192), max(h, w) / 2)   # ≥384px context window
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))

    crop = Image.fromarray(img_np[y0:y1, x0:x1])
    crop_mask = Image.fromarray(mask_uint8[y0:y1, x0:x1])
    cw, ch = crop.size

    result = rorem_pipe(
        prompt=ROREM_PROMPT, negative_prompt=ROREM_NEG_PROMPT,
        height=512, width=512,
        image=crop.resize((512, 512)),
        mask_image=crop_mask.resize((512, 512)),
        guidance_scale=1.0, num_inference_steps=25, strength=0.99,
    ).images[0].resize((cw, ch), Image.LANCZOS)

    # Feathered composite of the inpainted crop region only
    alpha_full = np.zeros((h, w), dtype=np.float32)
    alpha_full[y0:y1, x0:x1] = mask_uint8[y0:y1, x0:x1] / 255.0
    alpha_full = cv2.GaussianBlur(alpha_full, (31, 31), 0)[..., None]
    result_full = img_np.astype(np.float32).copy()
    result_full[y0:y1, x0:x1] = np.array(result).astype(np.float32)
    out = img_np.astype(np.float32) * (1 - alpha_full) + result_full * alpha_full
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def build_object_crop(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Masked object on a white background, cropped to its bbox (+pad), 512x512 —
    the input to TRELLIS image-to-3D (blueprint: fresh mask → image-to-3D)."""
    mask_binary = (mask > 0.5).astype(np.uint8)
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 20
    rmin, rmax = max(0, rmin - pad), min(img.size[1], rmax + pad)
    cmin, cmax = max(0, cmin - pad), min(img.size[0], cmax + pad)
    img_np = np.array(img).copy()
    img_np[mask_binary == 0] = 255
    return Image.fromarray(img_np[rmin:rmax, cmin:cmax]).resize((512, 512))


if RESUME_B:
    with open(RECORDS_PATH, "rb") as f:
        object_records = pickle.load(f)
    view_paths = [IMAGE_PATH] + sorted(
        os.path.join(VIEWS_DIR, f) for f in os.listdir(VIEWS_DIR) if f.endswith(".png"))
    print(f"[resume] {len(object_records)} object records, {len(view_paths)} views")
else:
    # Clear stale synthetic views from previous runs — but never when resuming
    # from a peel checkpoint, whose saved views ARE the restore state
    if not (args.resume and os.path.exists(os.path.join(OUTPUT_DIR, "peel_ckpt.pkl"))):
        for d in (VIEWS_DIR, CROPS_DIR):
            for f in os.listdir(d):
                if f.endswith(".png"):
                    os.remove(os.path.join(d, f))

    # ── Initial segmentation + depth ordering ────────────────────────────────
    print("\nInitial segmentation...")
    instances = segment(image, object_names)
    instances = consolidate_instances(instances, H * W)
    # Unique id per instance so duplicates ("chairs" x2) are peeled separately
    label_counts = {}
    for inst in instances:
        label_counts[inst["label"]] = label_counts.get(inst["label"], 0) + 1
        n = label_counts[inst["label"]]
        inst["uid"] = inst["label"] if n == 1 else f"{inst['label']} {n}"
    print(f"Created {len(instances)} instance masks")

    print("Computing depth map...")
    depth_map = get_depth_map(image)
    if args.sparse_depth:
        depth_map = recover_metric_scale(depth_map, args.sparse_depth)
    np.save("depth_map.npy", depth_map)

    # DepthAnything outputs disparity-like values: HIGHER = CLOSER.
    # Nearest point of each instance = max over its mask; peel largest first.
    for inst in instances:
        mb = inst["mask"] > 0.5
        inst["nearest"] = depth_map[mb].max() if mb.sum() > 0 else -np.inf
    sorted_instances = sorted(instances, key=lambda x: x["nearest"], reverse=True)

    # Support-aware reordering: an object resting ON or inside another (mask mostly
    # within the other's bbox, and smaller) must be peeled BEFORE its support —
    # otherwise the inpainter is forced to hallucinate a surface under the
    # still-unmasked supported object (e.g. blocks on the table).
    def rests_on(a, b):
        am, bm = a["mask"] > 0.5, b["mask"] > 0.5
        if am.sum() == 0 or bm.sum() == 0 or am.sum() >= bm.sum():
            return False
        ys, xs = np.where(bm)
        inside = am[ys.min():ys.max() + 1, xs.min():xs.max() + 1].sum()
        return inside / am.sum() > 0.7

    changed = True
    while changed:
        changed = False
        for i in range(len(sorted_instances)):
            for j in range(i):
                if rests_on(sorted_instances[i], sorted_instances[j]):
                    sorted_instances.insert(j, sorted_instances.pop(i))
                    changed = True
                    break
            if changed:
                break

    print("Peeling order (nearest first, supported objects before supports):")
    for i, inst in enumerate(sorted_instances, 1):
        print(f"  {i}. {inst['uid']} (nearest-point disparity: {inst['nearest']:.3f})")

    # ── Iterative occlusion peeling ──────────────────────────────────────────
    # Checkpoint after EVERY object so a machine crash mid-peel only costs the
    # object in flight: --resume restores the image state from the last saved
    # view and skips already-peeled objects (segmentation/depth are
    # deterministic, so the peel order reproduces identically).
    print("\nIterative occlusion peeling...")
    PEEL_CKPT = os.path.join(OUTPUT_DIR, "peel_ckpt.pkl")
    current_image = image.copy()
    view_paths = [IMAGE_PATH]            # view 0 = original image
    object_records = []                  # (uid, view_idx of the fresh mask, mask, local pc)
    done_uids = set()
    if args.resume and os.path.exists(PEEL_CKPT):
        with open(PEEL_CKPT, "rb") as f:
            _ck = pickle.load(f)
        done_uids = set(_ck["done_uids"])
        view_paths = _ck["view_paths"]
        object_records = _ck["records"]
        current_image = Image.open(view_paths[-1]).convert("RGB")
        print(f"[resume] peel checkpoint: {len(done_uids)} object(s) already peeled")

    for idx, inst in enumerate(sorted_instances):
        uid = inst["uid"]
        if uid in done_uids:
            print(f"\n  Peeling [{idx+1}/{len(sorted_instances)}]: {uid} [resume: already done]")
            continue
        print(f"\n  Peeling [{idx+1}/{len(sorted_instances)}]: {uid}")

        # Re-evaluate mask on the CURRENT image state (blueprint step 3a),
        # matching the right instance by IoU with the initial mask
        fresh = segment(current_image, [inst["label"]])
        mask, best_iou = inst["mask"], 0.0
        if inst.get("merged") and fresh:
            # merged instance (e.g. all blocks): compare against the fresh union
            union = np.zeros_like(fresh[0]["mask"])
            for f in fresh:
                union = np.maximum(union, f["mask"])
            iou = mask_iou(union, inst["mask"])
            if iou > best_iou:
                mask, best_iou = union, iou
        else:
            for f in fresh:
                iou = mask_iou(f["mask"], inst["mask"])
                if iou > best_iou:
                    mask, best_iou = f["mask"], iou
        if best_iou >= 0.3:
            print(f"    Fresh mask from current image state (IoU {best_iou:.2f})")
        else:
            mask = inst["mask"]
            print("    Re-detection mismatch; using initial mask")

        # Image-to-3D with the fresh mask (blueprint step 3b): defer to Phase B2.
        # Save the object crop; TRELLIS turns it into a point cloud later.
        if not args.skip_3d:
            crop = build_object_crop(current_image, mask)
            if crop is not None:
                i = len(object_records)            # crop i ↔ object_records[i]
                crop.save(os.path.join(CROPS_DIR, f"{i:02d}.png"))
                # view index of the state this mask was computed on = len(view_paths) - 1
                object_records.append((uid, len(view_paths) - 1,
                                       (mask > 0.5).astype(np.uint8), None))
                print("    Saved object crop for deferred TRELLIS 3D")

        # Object removal revealing what's behind (blueprint step 3c)
        current_image = remove_object(current_image, mask)
        view_path = os.path.join(VIEWS_DIR, f"view_{idx+1:02d}_{uid.replace(' ', '_')}.png")
        current_image.save(view_path)
        view_paths.append(view_path)
        print(f"    Saved synthetic view: {view_path}")

        # crash checkpoint: persist progress after every object
        done_uids.add(uid)
        with open(PEEL_CKPT, "wb") as f:
            pickle.dump({"done_uids": list(done_uids), "view_paths": view_paths,
                         "records": object_records}, f)

    print(f"\nPeeling complete: {len(view_paths)} views, {len(object_records)} object point clouds")
    with open(RECORDS_PATH, "wb") as f:
        pickle.dump([(n, v, m, p) for n, v, m, p in object_records], f)
    if os.path.exists(PEEL_CKPT):
        os.remove(PEEL_CKPT)   # full record saved; per-object checkpoint no longer needed

    free_cuda(sam3_model, sam3_processor, depth_model, depth_processor, rorem_pipe)

if args.stop_after_peeling:
    print(f"\nStopped after peeling (--stop_after_peeling). "
          f"Synthetic views in {VIEWS_DIR}/")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B2: DEFERRED IMAGE-TO-3D (TRELLIS)
# Runs after the iris-env models are freed, so the ~8-10 GB TRELLIS worker has the
# GPU to itself. Resumable: each object's point cloud is checkpointed into the
# records file, so a crash only costs the object in flight.
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_3d:
    need = [i for i, rec in enumerate(object_records) if rec[3] is None]
    if need:
        print("\n" + "=" * 60)
        print(f"[Phase B2] TRELLIS image-to-3D for {len(need)} object(s)")
        print("=" * 60)
        object_records = list(object_records)
        trellis = TrellisWorker()
        try:
            for i in need:
                uid, vi, m, _ = object_records[i]
                pc = trellis.pointcloud(os.path.join(CROPS_DIR, f"{i:02d}.png"))
                object_records[i] = (uid, vi, m, pc)
                with open(RECORDS_PATH, "wb") as f:
                    pickle.dump(object_records, f)   # checkpoint after each object
                print(f"  {uid}: TRELLIS point cloud {pc.shape}")
        finally:
            trellis.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE C: MULTI-VIEW SCENE RECONSTRUCTION (VGGT)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase C] Multi-view scene reconstruction (VGGT)")
print("=" * 60)

PM_PATH = os.path.join(OUTPUT_DIR, "vggt_pointmaps.npz")
if args.resume and os.path.exists(PM_PATH) and "extrinsics" in np.load(PM_PATH, allow_pickle=True):
    data = np.load(PM_PATH, allow_pickle=True)
    world_points, conf = data["world_points"], data["conf"]
    extrinsics, intrinsics = data["extrinsics"], data["intrinsics"]
    view_paths = [str(p) for p in data["view_paths"]]
    print(f"[resume] Loaded VGGT point maps: {world_points.shape}")
else:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda", dtype=torch.float32).eval()
    images_t = load_and_preprocess_images(view_paths).to("cuda", dtype=torch.float32)
    with torch.no_grad():
        predictions = vggt(images_t)

    world_points = predictions["world_points"].squeeze(0).cpu().float().numpy()   # (V, Hv, Wv, 3)
    conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()      # (V, Hv, Wv)
    # camera poses (for occupancy ray-casting in Phase G)
    extri, intri = pose_encoding_to_extri_intri(predictions["pose_enc"], images_t.shape[-2:])
    extrinsics = extri.squeeze(0).cpu().float().numpy()   # (V, 3, 4) world-to-camera
    intrinsics = intri.squeeze(0).cpu().float().numpy()   # (V, 3, 3)

    # Sparse-depth metric scaling: make the reconstruction metric so it can be
    # measured against the <2 cm KPI (VGGT alone is up-to-scale).
    metric_scale = 1.0
    if args.sparse_depth:
        metric_scale = metric_scale_from_sparse(
            world_points[0], extrinsics[0], args.sparse_depth, image.size[::-1])
        world_points = (world_points * metric_scale).astype(np.float32)
        extrinsics = extrinsics.copy()
        extrinsics[:, :, 3] *= metric_scale               # keep camera centres consistent
        print(f"Metric scale from sparse depth: x{metric_scale:.4f} (recon now metric)")

    print(f"VGGT point maps: {world_points.shape}")
    np.savez_compressed(PM_PATH, world_points=world_points, conf=conf,
                        extrinsics=extrinsics, intrinsics=intrinsics,
                        metric_scale=metric_scale, view_paths=np.array(view_paths))
    free_cuda(vggt, images_t, predictions)

V, Hv, Wv = conf.shape
scene_mask = conf > VGGT_CONF_THRESHOLD
scene_pc = world_points[scene_mask]
print(f"Scene point cloud: {scene_pc.shape}")
np.save(os.path.join(OUTPUT_DIR, "scene_pointcloud.npy"), scene_pc)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE D: RIGID REGISTRATION + FUSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase D] Object registration + fusion")
print("=" * 60)

import open3d as o3d

scene_diag = np.linalg.norm(scene_pc.max(0) - scene_pc.min(0))


def register_object(obj_pc: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """Scale+centroid init from mask-region scene points, then ICP refine."""
    src_diag = np.linalg.norm(obj_pc.max(0) - obj_pc.min(0))
    tgt_diag = np.linalg.norm(target_pts.max(0) - target_pts.min(0))
    scale = tgt_diag / max(src_diag, 1e-8)
    init = obj_pc * scale + (target_pts.mean(0) - obj_pc.mean(0) * scale)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(init)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts)
    threshold = 0.05 * scene_diag
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    src.transform(reg.transformation)
    print(f"    ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
    return np.asarray(src.points)


FUSED_PATH = os.path.join(OUTPUT_DIR, "fused_pointcloud.npy")
OBJECTS_PATH = os.path.join(OUTPUT_DIR, "registered_objects.pkl")
if args.resume and os.path.exists(FUSED_PATH) and os.path.getmtime(FUSED_PATH) > os.path.getmtime(PM_PATH):
    fused_pc = np.load(FUSED_PATH)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    registered_objects = pickle.load(open(OBJECTS_PATH, "rb")) if os.path.exists(OBJECTS_PATH) else []
    print(f"[resume] Loaded fused point cloud: {fused_pc.shape}")
else:
    fused = [scene_pc]
    registered_objects = []   # per-object registered clouds, for occupancy solidification
    for obj_name, view_idx, mask, obj_pc in object_records:
        print(f"\n  Registering: {obj_name} (view {view_idx})")
        mask_v = cv2.resize(mask, (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0.5
        valid = mask_v & scene_mask[view_idx]
        target_pts = world_points[view_idx][valid]
        if len(target_pts) < 50:
            print(f"    Only {len(target_pts)} mask points in scene, skipping registration")
            continue
        aligned = register_object(obj_pc, target_pts)
        fused.append(aligned)
        registered_objects.append(aligned)
        print(f"    Added {len(aligned)} points")

    with open(OBJECTS_PATH, "wb") as f:
        pickle.dump(registered_objects, f)
    fused_pc = np.concatenate(fused, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    pcd = pcd.voxel_down_sample(voxel_size=0.005 * scene_diag)
    fused_pc = np.asarray(pcd.points)
    print(f"\nFused point cloud (downsampled): {fused_pc.shape}")
    np.save(FUSED_PATH, fused_pc)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "fused_pointcloud.ply"), pcd)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE E: SEMANTIC LABELING (Mask2Former, multi-view voting)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase E] Semantic labeling (Mask2Former)")
print("=" * 60)

from sklearn.neighbors import KDTree

LABELS_PATH = os.path.join(OUTPUT_DIR, "labeled_pointcloud_labels.npy")
if args.resume and os.path.exists(LABELS_PATH) and os.path.getmtime(LABELS_PATH) > os.path.getmtime(FUSED_PATH):
    labels = np.load(LABELS_PATH)
    print(f"[resume] Loaded labels: {labels.shape}")
else:
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    m2f_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic", torch_dtype=torch.float16).to("cuda").eval()

    ref_pts, ref_labels = [], []
    for v, path in enumerate(view_paths):
        img_v = Image.open(path).convert("RGB")
        inputs = m2f_processor(images=img_v, return_tensors="pt")
        inputs = {k: v_.to(device="cuda", dtype=torch.float16) if torch.is_floating_point(v_) else v_.to("cuda")
                  for k, v_ in inputs.items()}
        with torch.no_grad():
            out = m2f(**inputs)
        seg = m2f_processor.post_process_semantic_segmentation(
            out, target_sizes=[(Hv, Wv)])[0].cpu().numpy()
        iris_map = np.full(seg.shape, IRIS_LABEL_TO_ID["other"], dtype=np.int32)
        for ade_id, iris_label in ADE20K_TO_IRIS.items():
            iris_map[seg == ade_id] = IRIS_LABEL_TO_ID[iris_label]

        valid = scene_mask[v]
        ref_pts.append(world_points[v][valid])
        ref_labels.append(iris_map[valid])
        print(f"  Labeled view {v}: {os.path.basename(path)}")

    ref_pts = np.concatenate(ref_pts, axis=0)
    ref_labels = np.concatenate(ref_labels, axis=0)

    tree = KDTree(ref_pts)
    dist, ind = tree.query(fused_pc, k=5)
    labels = np.full(len(fused_pc), IRIS_LABEL_TO_ID["other"], dtype=np.int32)
    radius = 0.02 * scene_diag
    for i in range(len(fused_pc)):
        near = ind[i][dist[i] < radius]
        if len(near) > 0:
            labels[i] = np.bincount(ref_labels[near]).argmax()

    print("\nLabel distribution:")
    for name, lid in IRIS_LABEL_TO_ID.items():
        n = int((labels == lid).sum())
        print(f"  {name}: {n} points ({100*n/len(labels):.1f}%)")

    np.save(os.path.join(OUTPUT_DIR, "labeled_pointcloud_points.npy"), fused_pc)
    np.save(LABELS_PATH, labels)
    colors = np.array([LABEL_COLORS[l] for l in labels])
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "labeled_pointcloud.ply"), pcd)

    free_cuda(m2f, m2f_processor)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE F: MESH GENERATION (Marching Cubes)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase F] Semantic mesh generation (Marching Cubes)")
print("=" * 60)

from skimage import measure
from scipy import ndimage

GRID = 128
mins = fused_pc.min(0)
maxs = fused_pc.max(0)
span = (maxs - mins).max()
voxel = span / (GRID - 1)

idx = np.clip(((fused_pc - mins) / voxel).astype(int), 0, GRID - 1)
occ = np.zeros((GRID, GRID, GRID), dtype=np.float32)
occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
occ = ndimage.binary_dilation(occ, iterations=1).astype(np.float32)
occ = ndimage.gaussian_filter(occ, sigma=1.0)

verts, faces, _, _ = measure.marching_cubes(occ, level=0.5)
verts_world = verts * voxel + mins
print(f"Mesh: {len(verts_world)} vertices, {len(faces)} faces")

vert_tree = KDTree(fused_pc)
_, vi = vert_tree.query(verts_world, k=1)
vert_labels = labels[vi[:, 0]]
vert_colors = np.array([LABEL_COLORS[l] for l in vert_labels])

mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_world)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
mesh_o3d.compute_vertex_normals()
mesh_path = os.path.join(OUTPUT_DIR, "final_semantic_mesh.ply")
o3d.io.write_triangle_mesh(mesh_path, mesh_o3d)
np.save(os.path.join(OUTPUT_DIR, "final_mesh_vertex_labels.npy"), vert_labels)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE G: FREE / OCCUPIED / OCCLUDED OCCUPANCY (occlusion-aware classification)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase G] Occupancy classification (free / occupied / occluded)")
print("=" * 60)

from step10_occupancy import compute_occupancy, render as render_occ, FREE, OCCUPIED, OCCLUDED

occ_grid, occ_min, occ_voxel, occ_dims = compute_occupancy(
    world_points, conf, extrinsics, fused_pc,
    object_clouds=registered_objects, grid=160)
occ_total = int(np.prod(occ_dims))
for lab, name in [(FREE, "free"), (OCCUPIED, "occupied"), (OCCLUDED, "occluded")]:
    n = int((occ_grid == lab).sum())
    print(f"  {name:9s}: {n:>9d}  ({100*n/occ_total:5.1f}%)")
np.save(os.path.join(OUTPUT_DIR, "occupancy_grid.npy"), occ_grid)
np.savez(os.path.join(OUTPUT_DIR, "occupancy_meta.npz"),
         grid_min=occ_min, voxel=occ_voxel, dims=occ_dims)
render_occ(occ_grid, occ_min, occ_voxel, os.path.join(OUTPUT_DIR, "occupancy_render.png"))

print(f"\n{'='*60}")
print("IRIS pipeline complete!")
print(f"  Synthetic views:    {VIEWS_DIR}/")
print(f"  Object meshes:      {MESH_DIR}/")
print(f"  Fused point cloud:  {OUTPUT_DIR}/fused_pointcloud.ply")
print(f"  Labeled cloud:      {OUTPUT_DIR}/labeled_pointcloud.ply")
print(f"  Semantic mesh:      {mesh_path}")
print(f"  Occupancy grid:     {OUTPUT_DIR}/occupancy_grid.npy (+ occupancy_render.png)")
print("=" * 60)
