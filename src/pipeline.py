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
parser.add_argument("--scene_dir", default=None,
                    help="folder of images (multi-view); overrides --image")
parser.add_argument("--sparse_depth", default=None,
                    help="optional .npy of (row, col, metric_depth) rows for scale recovery")
parser.add_argument("--depth", default=None,
                    help="optional dense metric depth map (.npy, aligned to --image) — "
                         "RGB-D mode: used directly for peel ordering instead of "
                         "DepthAnythingV2, and for metric scale (skips the depth model)")
parser.add_argument("--output_dir", default="output")
parser.add_argument("--resume", action="store_true",
                    help="skip phases whose outputs already exist (crash recovery)")
parser.add_argument("--skip_3d", action="store_true",
                    help="skip per-object image-to-3D; fused recon = VGGT scene")
parser.add_argument("--image3d", choices=["trellis", "tigon", "amodal3r", "splattn", "wonder3d"], default="trellis",
                    help="per-object 3D backend: image-only TRELLIS (default), text+image "
                         "TIGON (Phase-A label as prompt), occlusion-aware AMODAL3R (SAM3 mask), "
                         "or SPLATTN point-cloud completion (completes the VGGT partial in place, "
                         "no registration)")
parser.add_argument("--stop_after_peeling", action="store_true",
                    help="exit after Phase B (for fast removal A/B comparison)")
args = parser.parse_args()

# RGB-D mode: derive sparse metric points from the dense --depth so the existing
# metric-scale machinery (Phase C) makes the reconstruction metric for free.
if args.depth and not args.sparse_depth:
    import numpy as _np
    os.makedirs(args.output_dir, exist_ok=True)
    _d = _np.load(args.depth).astype(_np.float32)
    _ys, _xs = _np.where(_d > 1e-3)
    if len(_ys):
        _sel = _np.random.choice(len(_ys), min(3000, len(_ys)), replace=False)
        _sp = _np.stack([_ys[_sel], _xs[_sel], _d[_ys[_sel], _xs[_sel]]], 1).astype(_np.float32)
        _spp = os.path.join(args.output_dir, "_depth_sparse.npy")
        _np.save(_spp, _sp)
        args.sparse_depth = _spp        # reuse sparse → VGGT metric scaling

# per-output-dir so parallel runs (different --output_dir) don't collide
RECORDS_PATH = os.path.join(args.output_dir, "object_records.pkl")

if args.scene_dir:
    import glob as _glob
    _exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_list = sorted(p for p in _glob.glob(os.path.join(args.scene_dir, "*"))
                        if p.endswith(_exts))
    if not image_list:
        raise SystemExit(f"No images found in {args.scene_dir}")
    IMAGE_PATH = image_list[0]
else:
    image_list = [args.image]
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

# Semantic taxonomy. Phase E labels OBJECTS from the SAM3 instance masks + VLM
# names (open-vocab, mapped down to these canonical classes) and BACKGROUND STUFF
# (floor/wall/ceiling/...) from Mask2Former — instead of collapsing every object
# into "other". This set is our own choice; remap CLASS_COLORS / NAME_TO_CLASS /
# ADE20K_TO_IRIS to a fixed benchmark taxonomy (ScanNet-20, S3DIS-13) when grading.
CLASS_COLORS = {                       # name -> RGB (0-1)
    "floor":   [0.55, 0.40, 0.22],     # brown
    "wall":    [0.80, 0.80, 0.80],     # light gray
    "ceiling": [0.90, 0.90, 0.70],     # pale yellow
    "window":  [0.60, 0.85, 0.95],     # pale cyan
    "curtain": [0.70, 0.55, 0.85],     # lavender
    "door":    [0.50, 0.35, 0.25],     # dark brown
    "chair":   [0.90, 0.20, 0.20],     # red
    "table":   [0.20, 0.45, 0.90],     # blue
    "sofa":    [0.95, 0.45, 0.75],     # pink
    "bed":     [0.55, 0.25, 0.60],     # purple
    "cabinet": [0.85, 0.55, 0.20],     # orange
    "shelf":   [0.95, 0.80, 0.25],     # gold
    "screen":  [0.10, 0.70, 0.70],     # teal  (monitor/tv/laptop/phone)
    "book":    [0.95, 0.35, 0.10],     # vermilion (book/folder/paper)
    "box":     [0.40, 0.26, 0.13],     # umber
    "bottle":  [0.20, 0.80, 0.40],     # green
    "lamp":    [1.00, 0.95, 0.55],     # bright yellow
    "bag":     [0.45, 0.30, 0.65],     # indigo
    "appliance": [0.30, 0.55, 0.55],   # slate
    "plant":   [0.25, 0.65, 0.25],     # leaf green
    "tool":    [0.95, 0.75, 0.10],     # amber (hammer/screwdriver/wrench/...)
    "other":   [0.55, 0.55, 0.55],     # neutral gray
}
IRIS_CLASSES = list(CLASS_COLORS.keys())
IRIS_LABEL_TO_ID = {n: i for i, n in enumerate(IRIS_CLASSES)}
LABEL_COLORS = {i: CLASS_COLORS[n] for n, i in IRIS_LABEL_TO_ID.items()}

# ADE20K (Mask2Former) ids -> our STUFF classes only. Objects come from instances,
# not from ADE, so we deliberately map only background structure here.
ADE20K_TO_IRIS = {
    0: "wall", 1: "wall",
    3: "floor", 28: "floor", 54: "floor", 78: "floor",
    5: "ceiling",
    8: "window",
    18: "curtain",
    14: "door",
}

# VLM open-vocab name -> canonical class, by keyword (first match wins). Order
# matters: more specific words before generic ones.
_NAME_RULES = [
    (("ceiling",), "ceiling"), (("floor", "rug", "carpet"), "floor"),
    (("curtain", "drape", "blind"), "curtain"), (("window",), "window"),
    (("door",), "door"), (("wall",), "wall"),
    (("office chair", "chair", "stool", "seat"), "chair"),
    (("desk", "table", "platform", "counter"), "table"),
    (("sofa", "couch"), "sofa"), (("bed", "mattress"), "bed"),
    (("bookshelf", "bookcase", "shelf", "rack"), "shelf"),
    (("cabinet", "drawer", "nightstand", "wardrobe", "dresser"), "cabinet"),
    (("monitor", "screen", "tv", "television", "laptop", "computer", "phone",
      "mouse", "keyboard", "remote", "speaker", "tablet"), "screen"),
    (("book", "folder", "paper", "magazine", "notebook", "document"), "book"),
    (("box", "carton", "case", "toolbox", "container"), "box"),
    (("bottle", "cup", "mug", "glass", "can", "jar", "vase"), "bottle"),
    (("lamp", "light"), "lamp"),
    (("bag", "backpack", "purse", "cloth", "towel"), "bag"),
    (("hammer", "screwdriver", "wrench", "drill", "plier", "saw", "spanner",
      "knife", "scissor", "tool"), "tool"),
    (("plant", "flower", "tree"), "plant"),
    (("fridge", "microwave", "oven", "fan", "heater", "appliance",
      "printer", "machine"), "appliance"),
]


def name_to_class(name: str) -> str:
    n = name.lower()
    for keys, cls in _NAME_RULES:
        if any(k in n for k in keys):
            return cls
    return "other"


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


class Image3DWorker:
    """Drives an image->3D model in its own pinned conda env via a persistent
    subprocess speaking a line-based "@@" protocol. Input: a white-bg object crop
    PNG (+ optional text label). Output: a point cloud (gaussian xyz).

    Backends (all built on the TRELLIS gaussian decoder, same get_xyz contract):
      - "trellis":  image-only TRELLIS-image-large, `trellis` env, cwd = repo root.
      - "tigon"  :  text+image TIGON, `tigon` env, cwd = models/TIGON (its code uses
                    relative ./mix_e2e_pipe and ./external paths). The object label
                    from Phase A is passed as the text condition.
      - "amodal3r": occlusion-aware Amodal3R (TRELLIS fork), runs in the `tigon` env
                    (shares its deps). Takes the object's mask in addition to the
                    crop; reconstructs the complete object, tolerating occlusion.
    """

    _CFG = {
        "trellis":  dict(env="trellis",  script="src/trellis_worker.py",  cwd=None),
        "tigon":    dict(env="tigon",    script="src/tigon_worker.py",    cwd=config.TIGON_DIR),
        "amodal3r": dict(env="tigon",    script="src/amodal3r_worker.py", cwd=None),
        # SplAttN is point-cloud completion (not image-to-3D): it completes an
        # object's VGGT partial in place, so it has no register step. cwd is its
        # repo so config_55 / models.SplAttN import.
        "splattn":  dict(env="splattn",  script="src/splattn_worker.py",  cwd=config.SPLATTN_DIR),
        # Wonder3D: cross-domain multi-view diffusion (6 ortho views) + visual-hull
        # carve to a point cloud. cwd is its repo (relative ./mvdiffusion imports).
        "wonder3d": dict(env="wonder3d", script="src/wonder3d_worker.py", cwd=config.WONDER3D_DIR),
    }

    def __init__(self, backend: str = "trellis"):
        import subprocess
        import tempfile
        if backend not in self._CFG:
            raise ValueError(f"unknown image3d backend {backend!r}; choose from {list(self._CFG)}")
        cfg = self._CFG[backend]
        self.backend = backend
        pybin = config.conda_env_python(cfg["env"])
        self.tmp = tempfile.mkdtemp(prefix=f"{backend}worker_")
        self.log = open(os.path.join(self.tmp, "worker.log"), "w")
        env = dict(os.environ, ATTN_BACKEND="xformers", SPCONV_ALGO="native")
        # prepend the worker env's bin so its tools (e.g. ninja, needed by
        # knn_cuda's JIT build for splattn) are found — the subprocess otherwise
        # inherits the parent (iris) env's PATH, not the worker env's.
        env["PATH"] = os.path.dirname(pybin) + os.pathsep + env.get("PATH", "")
        # worker script path must be absolute since the worker may run from a
        # different cwd (TIGON resolves ./external and ./mix_e2e_pipe relatively).
        script = os.path.join(config.REPO_ROOT, cfg["script"])
        self.proc = subprocess.Popen(
            [pybin, "-u", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env, cwd=cfg["cwd"],
        )
        for line in self.proc.stdout:
            if line.strip() == "@@READY":
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"{backend} worker exited before READY; see " + self.log.name)

    def pointcloud(self, crop_path: str, n: int = 10000, text: str = "",
                   mask_path: str = None) -> np.ndarray:
        import json
        out = os.path.join(self.tmp, "pc.npy")
        # absolute paths so they resolve regardless of the worker's cwd
        req = {"image": os.path.abspath(crop_path), "out": out, "n": n, "text": text}
        if mask_path is not None:                       # amodal3r consumes the mask
            req["mask"] = os.path.abspath(mask_path)
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("@@OK"):
                return np.load(out)
            if line.startswith("@@ERR"):
                raise RuntimeError(f"{self.backend} worker: " + line)
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.backend} worker died; see " + self.log.name)
        raise RuntimeError(f"{self.backend} worker stdout closed unexpectedly")

    def complete(self, partial: np.ndarray, scene_up: np.ndarray) -> np.ndarray:
        """SplAttN path: complete an object's VGGT partial in place (scene coords)."""
        import json
        pin = os.path.join(self.tmp, "partial.npy")
        out = os.path.join(self.tmp, "completed.npy")
        np.save(pin, partial.astype(np.float32))
        req = {"partial": pin, "up": [float(x) for x in scene_up], "out": out}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("@@OK"):
                return np.load(out)
            if line.startswith("@@ERR"):
                raise RuntimeError(f"{self.backend} worker: " + line)
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.backend} worker died; see " + self.log.name)
        raise RuntimeError(f"{self.backend} worker stdout closed unexpectedly")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# used for sparse-depth metric scaling in Phase C (first image's size)
image = Image.open(IMAGE_PATH).convert("RGB")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE A: OBJECT DISCOVERY (VLM)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase A] Object discovery (Qwen3-VL)")
print("=" * 60)

# The 8B VLM is run in its own subprocess so the OS fully reclaims its ~16 GB
# of VRAM on exit (device_map="auto" leaves accelerate hooks that don't free
# cleanly in-process, which OOMs the multi-model Phase B that follows).
import subprocess
step0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step0_vlm.py")


def _freest_gpu():
    """Pick the GPU with the most free memory, so the big VLM lands on a clear
    card (this box has 8×H100; GPU 0 may be shared with other processes)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        free = [int(x) for x in out.split()]
        return str(int(np.argmax(free))) if free else "0"
    except Exception:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "0")


_vlm_env = dict(os.environ, CUDA_VISIBLE_DEVICES=_freest_gpu())
print(f"  VLM ({config.VLM_ID}) on GPU {_vlm_env['CUDA_VISIBLE_DEVICES']}")
_all_names = []
for _img_path in image_list:
    _stem = os.path.splitext(os.path.basename(_img_path))[0]
    _det_file = os.path.join(OUTPUT_DIR, f"detected_{_stem}.txt")
    if not (args.resume and os.path.exists(_det_file)):
        subprocess.run([sys.executable, step0, "--image", _img_path,
                        "--out", _det_file], check=True, env=_vlm_env)
    with open(_det_file) as _f:
        _all_names.extend(l.strip() for l in _f if l.strip())
object_names = list(dict.fromkeys(_all_names))   # union, preserve order, deduplicate
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

# W, H used later only for sparse-depth (single-image); multi-image uses first image
W, H = image.size


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

    # Cross-label NMS: the VLM names the same object differently across images
    # ("blue bottle" vs "blue water bottle", "black mouse" vs "gray mouse"), and
    # those synonyms are all prompted on every image — so the same physical object
    # gets segmented multiple times under different labels. Geometry settles it:
    # if two detections cover the same pixels, they are one object — keep the
    # highest-scoring, drop the rest. (Mask geometry, so no name matching needed.)
    XLABEL_IOU = 0.5
    result = sorted(result, key=lambda x: -x["score"])
    deduped = []
    for inst in result:
        dup_of = next((k for k in deduped if mask_iou(inst["mask"], k["mask"]) > XLABEL_IOU), None)
        if dup_of is None:
            deduped.append(inst)
        elif inst["label"] != dup_of["label"]:
            print(f"  cross-label dup: '{inst['label']}' ≈ '{dup_of['label']}' "
                  f"(IoU>{XLABEL_IOU}) — dropped")
    return deduped


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


def build_occlusion_order(instances, depth_map, band_px=6, margin_frac=0.04):
    """Front-to-back peel order via a pairwise occlusion graph + topological sort.

    Peeling must remove an occluder *before* what it hides, so the ordering
    question is fundamentally pairwise (who is in front of whom *where they
    meet*) — not a global depth sort. DepthAnythingV2 outputs DISPARITY
    (higher = nearer), verified empirically.

    For each adjacent mask pair we compare the median disparity in their shared
    contact band (sampled just *inside* each, away from the unreliable border):
    the nearer side there is the occluder → directed edge occluder→occluded.
    Two extra signals: a physical-support edge (a small object resting in
    another's footprint is peeled first), and, when no edge is decisive,
    objects fall back to their global nearest-point disparity. A topological
    sort then yields the peel order; cycles from boundary noise are broken at
    their weakest (smallest-margin) edge.

    Returns instances reordered front (peel first) → back.
    """
    n = len(instances)
    if n <= 1:
        return list(instances)

    H, W = depth_map.shape
    spread = float(np.percentile(depth_map, 95) - np.percentile(depth_map, 5)) + 1e-9
    tau = margin_frac * spread                      # min disparity gap to call an edge
    masks = [inst["mask"] > 0.5 for inst in instances]
    k = np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)
    dil = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in masks]
    ero = [cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for m in masks]

    centroid_y = [float(np.where(m)[0].mean()) if m.sum() else 0.0 for m in masks]

    def rests_on(ai, bi):
        """a rests ON TOP of b: a is smaller, makes REAL mask contact with b
        (not mere bbox overlap), and the contact is at/below a's centre
        (gravity — a sits above where it touches b)."""
        am, bm = masks[ai], masks[bi]
        if am.sum() == 0 or bm.sum() == 0 or am.sum() >= bm.sum():
            return False
        contact = dil[ai] & bm
        if contact.sum() < max(12, 0.05 * np.sqrt(am.sum())):
            return False
        return float(np.where(contact)[0].mean()) >= centroid_y[ai]

    # edges[a].add(b)  ==  "a occludes b, peel a first"; weight = decision margin
    edges = {i: set() for i in range(n)}
    weight = {}
    for a in range(n):
        for b in range(a + 1, n):
            # contact band: a's interior touching b, and b's interior touching a
            a_band = ero[a] & dil[b]
            b_band = ero[b] & dil[a]
            occ = None
            if a_band.sum() >= 8 and b_band.sum() >= 8:
                da = float(np.median(depth_map[a_band]))
                db = float(np.median(depth_map[b_band]))
                if abs(da - db) >= tau:                       # decisive depth call
                    occ, hid, w, why = ((a, b, abs(da - db), "depth") if da > db
                                        else (b, a, abs(da - db), "depth"))
            if occ is None:                                   # depth indecisive → support tiebreak
                if rests_on(a, b):
                    occ, hid, w, why = a, b, tau, "support"
                elif rests_on(b, a):
                    occ, hid, w, why = b, a, tau, "support"
            if occ is not None:
                edges[occ].add(hid)
                weight[(occ, hid)] = w
                if os.environ.get("IRIS_DEBUG_OCCLUSION"):
                    print(f"    edge: {instances[occ]['uid']} -> {instances[hid]['uid']} "
                          f"({why}, w={w:.1f})")

    # Kahn topological sort; ties + free choices broken by nearest-point disparity
    nearest = {i: inst["nearest"] for i, inst in enumerate(instances)}
    indeg = {i: 0 for i in range(n)}
    for a in edges:
        for b in edges[a]:
            indeg[b] += 1

    order = []
    avail = [i for i in range(n) if indeg[i] == 0]
    placed = set()
    while len(order) < n:
        if not avail:                               # cycle: drop weakest remaining edge
            rem = [(weight[(a, b)], a, b) for a in edges for b in edges[a]
                   if a not in placed and b not in placed]
            if not rem:
                avail = [i for i in range(n) if i not in placed]
            else:
                _, a, b = min(rem)
                edges[a].discard(b)
                indeg[b] -= 1
                if indeg[b] == 0:
                    avail.append(b)
                continue
        # among available, peel the nearest (highest disparity) first
        avail.sort(key=lambda i: nearest[i], reverse=True)
        cur = avail.pop(0)
        if cur in placed:
            continue
        order.append(cur)
        placed.add(cur)
        for b in edges[cur]:
            indeg[b] -= 1
            if indeg[b] == 0 and b not in placed:
                avail.append(b)

    return [instances[i] for i in order]


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


def build_object_crop(img: Image.Image, mask: np.ndarray):
    """Masked object on a white background, cropped to its bbox (+pad) and
    letterboxed into a square at 512x512 — the input to image-to-3D. Returns
    (crop, mask_crop): the RGB crop and the letterboxed SAM3 mask (L, 0/255) in
    the same frame, so occlusion-aware backends (Amodal3R) get the exact object
    region without re-deriving it from the white background (which would drop
    white object parts, e.g. a white bottle cap).

    The square padding is essential: a direct resize to 512x512 squishes a
    non-square bbox (e.g. a tall bottle, 2.4:1) into a square, and the image-to-3D
    model then reconstructs the distorted proportions. Padding to a square first
    preserves the object's true aspect ratio."""
    mask_binary = (mask > 0.5).astype(np.uint8)
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not rows.any():
        return None, None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 20
    rmin, rmax = max(0, rmin - pad), min(img.size[1], rmax + pad)
    cmin, cmax = max(0, cmin - pad), min(img.size[0], cmax + pad)
    img_np = np.array(img).copy()
    img_np[mask_binary == 0] = 255
    crop = img_np[rmin:rmax, cmin:cmax]
    mcrop = (mask_binary[rmin:rmax, cmin:cmax] * 255).astype(np.uint8)
    # letterbox both into a square canvas (preserve aspect ratio), then resize
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    mcanvas = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    mcanvas[y0:y0 + h, x0:x0 + w] = mcrop
    crop_img = Image.fromarray(canvas).resize((512, 512))
    mask_img = Image.fromarray(mcanvas).resize((512, 512), Image.NEAREST)
    return crop_img, mask_img


if RESUME_B:
    with open(RECORDS_PATH, "rb") as f:
        _saved = pickle.load(f)
    if isinstance(_saved, dict):
        object_records = _saved["records"]
        view_paths = _saved["view_paths"]
    else:
        # backward compat: old single-image format (plain list)
        object_records = _saved
        view_paths = [IMAGE_PATH] + sorted(
            os.path.join(VIEWS_DIR, fp) for fp in os.listdir(VIEWS_DIR) if fp.endswith(".png"))
    print(f"[resume] {len(object_records)} object records, {len(view_paths)} views")
else:
    # Clear stale synthetic views from previous runs — but never when resuming
    # from a peel checkpoint, whose saved views ARE the restore state
    PEEL_CKPT = os.path.join(OUTPUT_DIR, "peel_ckpt.pkl")
    if not (args.resume and os.path.exists(PEEL_CKPT)):
        for d in (VIEWS_DIR, CROPS_DIR):
            for fp in os.listdir(d):
                if fp.endswith(".png"):
                    os.remove(os.path.join(d, fp))

    PEEL_CKPT = os.path.join(OUTPUT_DIR, "peel_ckpt.pkl")
    view_paths = []
    object_records = []
    # peel checkpoint tracks progress across images
    _done_img_idx = 0
    _done_uids_by_img = {}   # img_idx -> set of peeled uids
    _ckpt_view_paths = None
    if args.resume and os.path.exists(PEEL_CKPT):
        with open(PEEL_CKPT, "rb") as f:
            _ck = pickle.load(f)
        _done_img_idx = _ck.get("done_img_idx", 0)
        _done_uids_by_img = _ck.get("done_uids_by_img", {})
        view_paths = _ck["view_paths"]
        object_records = _ck["records"]
        _ckpt_view_paths = view_paths
        print(f"[resume] peel checkpoint: img {_done_img_idx}/{len(image_list)}, "
              f"{sum(len(v) for v in _done_uids_by_img.values())} object(s) peeled")

    for img_idx, img_path in enumerate(image_list):
        img_stem = os.path.splitext(os.path.basename(img_path))[0]

        if img_idx < _done_img_idx:
            print(f"\n[img {img_idx+1}/{len(image_list)}] {img_stem} [resume: already done]")
            continue

        print(f"\n{'='*60}")
        print(f"[img {img_idx+1}/{len(image_list)}] Peeling: {img_stem}")
        print(f"{'='*60}")

        cur_img = Image.open(img_path).convert("RGB")
        cur_H, cur_W = np.array(cur_img).shape[:2]

        # ── Initial segmentation + depth ordering ────────────────────────────
        print("\nInitial segmentation...")
        instances = segment(cur_img, object_names)
        instances = consolidate_instances(instances, cur_H * cur_W)
        label_counts = {}
        for inst in instances:
            label_counts[inst["label"]] = label_counts.get(inst["label"], 0) + 1
            n = label_counts[inst["label"]]
            inst["uid"] = inst["label"] if n == 1 else f"{inst['label']} {n}"
        print(f"Created {len(instances)} instance masks")

        if args.depth and img_idx == 0:
            # RGB-D mode: use the provided metric depth directly (no depth model).
            # Peel ordering wants "higher = nearer" (disparity), so invert the
            # metric depth (nearer = smaller metres -> larger inverse depth).
            print("Using provided RGB-D depth (skipping DepthAnythingV2)...")
            dm = np.load(args.depth).astype(np.float32)
            if dm.shape[:2] != (cur_H, cur_W):
                dm = cv2.resize(dm, (cur_W, cur_H), interpolation=cv2.INTER_NEAREST)
            depth_map = np.where(dm > 1e-3, 1.0 / np.clip(dm, 1e-3, None), 0.0)
        else:
            print("Computing depth map...")
            depth_map = get_depth_map(cur_img)
            if args.sparse_depth and img_idx == 0:
                depth_map = recover_metric_scale(depth_map, args.sparse_depth)
        np.save(os.path.join(OUTPUT_DIR, f"depth_map_{img_stem}.npy"), depth_map)

        for inst in instances:
            mb = inst["mask"] > 0.5
            inst["nearest"] = depth_map[mb].max() if mb.sum() > 0 else -np.inf
        sorted_instances = build_occlusion_order(instances, depth_map)

        print("Peeling order (occlusion graph + topological sort, front first):")
        for i, inst in enumerate(sorted_instances, 1):
            print(f"  {i}. {inst['uid']} (nearest-point disparity: {inst['nearest']:.3f})")

        # ── Iterative occlusion peeling ──────────────────────────────────────
        print("\nIterative occlusion peeling...")
        # base view for this image (view_paths may already contain prior images)
        view_paths.append(img_path)
        current_image = cur_img.copy()
        done_uids = set(_done_uids_by_img.get(img_idx, set()))
        # if resuming mid-image, restore current image state from last saved view
        if done_uids and _ckpt_view_paths:
            current_image = Image.open(view_paths[-1]).convert("RGB")

        for idx, inst in enumerate(sorted_instances):
            uid = inst["uid"]
            if uid in done_uids:
                print(f"\n  Peeling [{idx+1}/{len(sorted_instances)}]: {uid} [resume: already done]")
                continue
            print(f"\n  Peeling [{idx+1}/{len(sorted_instances)}]: {uid}")

            fresh = segment(current_image, [inst["label"]])
            mask, best_iou = inst["mask"], 0.0
            if inst.get("merged") and fresh:
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

            # Always record the instance mask (uid + view + mask) — Phase E uses it
            # to label points by object. The crop / image-to-3D work stays gated.
            record_mask = (mask > 0.5).astype(np.uint8)
            if not args.skip_3d:
                crop, crop_mask = build_object_crop(current_image, mask)
                if crop is not None:
                    i = len(object_records)
                    crop.save(os.path.join(CROPS_DIR, f"{i:02d}.png"))
                    crop_mask.save(os.path.join(CROPS_DIR, f"{i:02d}_mask.png"))
                    print("    Saved object crop (+SAM3 mask) for deferred image-to-3D")
            object_records.append((uid, len(view_paths) - 1, record_mask, None))

            current_image = remove_object(current_image, mask)
            view_path = os.path.join(
                VIEWS_DIR, f"{img_stem}_view_{idx+1:02d}_{uid.replace(' ', '_')}.png")
            current_image.save(view_path)
            view_paths.append(view_path)
            print(f"    Saved synthetic view: {view_path}")

            done_uids.add(uid)
            _done_uids_by_img[img_idx] = done_uids
            with open(PEEL_CKPT, "wb") as f:
                pickle.dump({"done_img_idx": img_idx, "done_uids_by_img": _done_uids_by_img,
                             "view_paths": view_paths, "records": object_records}, f)

        print(f"\nPeeled {img_stem}: {len(sorted_instances)} objects")
        _done_img_idx = img_idx + 1

    print(f"\nPeeling complete: {len(view_paths)} views, {len(object_records)} object point clouds")
    with open(RECORDS_PATH, "wb") as f:
        pickle.dump({"records": object_records, "view_paths": view_paths}, f)
    if os.path.exists(PEEL_CKPT):
        os.remove(PEEL_CKPT)

    free_cuda(sam3_model, sam3_processor, depth_model, depth_processor, rorem_pipe)

if args.stop_after_peeling:
    print(f"\nStopped after peeling (--stop_after_peeling). "
          f"Synthetic views in {VIEWS_DIR}/")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B2: DEFERRED IMAGE-TO-3D (TRELLIS or TIGON)
# Runs after the iris-env models are freed, so the ~8-10 GB worker has the GPU to
# itself. Resumable: each object's point cloud is checkpointed into the records
# file, so a crash only costs the object in flight. With --image3d tigon, the
# object's Phase-A label is passed as a text prompt (TIGON is text+image cond).
# ═══════════════════════════════════════════════════════════════════════════════
# SPLATTN is point-cloud completion, not image-to-3D — it runs in Phase D on the
# VGGT partials, so skip the per-object image-to-3D worker here.
if not args.skip_3d and args.image3d != "splattn":
    # records without a saved crop (object too small to crop) carry no image-to-3D;
    # they still get an instance mask for Phase E labeling, just skip them here.
    need = [i for i, rec in enumerate(object_records) if rec[3] is None
            and os.path.exists(os.path.join(CROPS_DIR, f"{i:02d}.png"))]
    if need:
        backend = args.image3d
        print("\n" + "=" * 60)
        print(f"[Phase B2] {backend.upper()} image-to-3D for {len(need)} object(s)")
        print("=" * 60)
        object_records = list(object_records)
        worker = Image3DWorker(backend)
        try:
            for i in need:
                uid, vi, m, _ = object_records[i]
                # uid is the Phase-A object label (e.g. "purple water bottle");
                # TIGON conditions on it as a text prompt, TRELLIS ignores it.
                # amodal3r additionally consumes the SAM3 mask (saved alongside).
                mask_path = (os.path.join(CROPS_DIR, f"{i:02d}_mask.png")
                             if backend in ("amodal3r", "wonder3d") else None)
                pc = worker.pointcloud(os.path.join(CROPS_DIR, f"{i:02d}.png"),
                                       text=uid, mask_path=mask_path)
                object_records[i] = (uid, vi, m, pc)
                with open(RECORDS_PATH, "wb") as f:
                    pickle.dump(object_records, f)   # checkpoint after each object
                print(f"  {uid}: {backend} point cloud {pc.shape}")
        finally:
            worker.close()

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


def _R_axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    """Rotation matrix for `ang` radians about unit `axis` (Rodrigues)."""
    a = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _R_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation aligning unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:                                   # parallel or antiparallel
        return np.eye(3) if c > 0 else _R_axis_angle(np.array([1.0, 0, 0]), np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def scene_up_vector(scene_pc: np.ndarray):
    """Gravity/up direction = the FLOOR/TABLE plane normal, plus the floor height.

    Picking the *dominant* plane is fragile: scenes with a large background wall
    (facing the camera) make the WALL dominant, so the up vector ends up along the
    view axis and the whole reconstruction tilts. Instead we extract several planes
    and keep the most *vertical* one — for a roughly-level camera (VGGT's y-down
    convention) the floor normal aligns with the ±Y axis, whereas a camera-facing
    wall's normal lies along ±Z. Among sufficiently large planes, max |n.y| wins."""
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_pc)
        rem = pcd
        best = None   # (verticality, normal, inlier_pts)
        for _ in range(4):                       # peel off up to 4 dominant planes
            if len(rem.points) < 300:
                break
            plane, inl = rem.segment_plane(0.02 * scene_diag, 3, 200)
            n = np.asarray(plane[:3], float); n /= (np.linalg.norm(n) + 1e-9)
            pts = np.asarray(rem.points)[inl]
            if len(inl) / len(scene_pc) > 0.05:  # only planes of meaningful size
                vert = abs(float(n[1]))          # alignment with the vertical (±Y)
                if best is None or vert > best[0]:
                    best = (vert, n.copy(), pts.copy())
            rem = rem.select_by_index(inl, invert=True)
        if best is None:
            raise RuntimeError("no plane")
        n, fpts = best[1], best[2]
        if np.dot(n, scene_pc.mean(0) - fpts.mean(0)) < 0:
            n = -n
        floor_h = float(np.median(fpts @ n))     # floor level along up
        return n, floor_h
    except Exception:
        return np.array([0.0, 0.0, 1.0]), float(np.percentile(scene_pc[:, 2], 5))


def register_object(obj_pc: np.ndarray, target_pts: np.ndarray,
                    scene_up: np.ndarray, floor_h: float = None) -> np.ndarray:
    """Pose a complete generated object so its visible surface coincides with the
    observed scene points (`target_pts`) — i.e. the object is dropped in as an
    *extension* of the existing segment, completing the occluded geometry.

    The previous version compared the full object's bbox diagonal to the diagonal
    of the partial visible patch (shrinking the object) and ran symmetric ICP from
    the object's canonical pose (wrong orientation). This version instead:
      1. trims observed outliers (mask bleed / depth noise);
      2. fixes scale from the patch's lateral extent (below);
      3. tilts the canonical object so its up-axis matches scene gravity (the floor
         normal), fixing 2 of 3 rotation DoF;
      4. searches the remaining yaw about the up-axis (24 angles), and at each yaw
         slides the object (translation-only ICP) so observed points land on its
         surface; keeps the yaw with the smallest observed→object residual.
    The chosen pose minimizes the observed→object distance (every observed point
    lands on the object surface).

    Two design choices keep it from cheating:
      - Scale is fixed BEFORE refine from the observed patch's lateral extent (its
        two largest principal spreads; the depth spread of the thin front shell is
        unreliable). Letting ICP also solve scale lets it shrink the object to
        nestle inside the patch.
      - The refine is translation + yaw only. Gravity fixes the tilt; a full-SO(3)
        ICP would tip the object over to nestle into the partial segment."""
    from scipy.spatial import cKDTree

    # trim observed outliers (mask bleed onto neighbours, VGGT depth noise) that
    # would inflate the scale estimate and drag the fit — keep points within a
    # robust radius of the median.
    med = np.median(target_pts, axis=0)
    rad = np.linalg.norm(target_pts - med, axis=1)
    keep = rad < (np.median(rad) + 2.5 * (np.median(np.abs(rad - np.median(rad))) + 1e-9))
    if keep.sum() >= 50:
        target_pts = target_pts[keep]

    def _spread(P):                                 # principal std (descending)
        w = np.linalg.eigvalsh(np.cov((P - P.mean(0)).T))
        return np.sqrt(np.clip(w, 0, None))[::-1]
    te, oe = _spread(target_pts), _spread(obj_pc)
    s0 = (te[0] + te[1]) / max(oe[0] + oe[1], 1e-8)  # match the two lateral axes

    obj_c = (obj_pc - obj_pc.mean(0)) * s0           # scaled, centered object
    tgt_mean = target_pts.mean(0)
    R_tilt = _R_align(np.array([0.0, 0.0, 1.0]), scene_up)   # canonical +Z → gravity

    best = None   # (resid, aligned_pts)
    for yaw in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        R0 = _R_axis_angle(scene_up, yaw) @ R_tilt
        src = obj_c @ R0.T + tgt_mean
        # translation-only asymmetric ICP: slide the (upright) object so observed
        # points land on its surface — never re-rotates, so tilt stays gravity-true
        for _ in range(10):
            tree = cKDTree(src)
            _, idx = tree.query(target_pts)
            src = src - (src[idx] - target_pts).mean(0)
        resid = cKDTree(src).query(target_pts)[0].mean()
        if best is None or resid < best[0]:
            best = (resid, src)

    aligned = best[1]

    # Floor contact: tabletop objects rest on the floor, but the visible segment
    # (and a slightly squat generated object) often leaves the base floating just
    # above it — e.g. the bottle's diameter is right but it doesn't reach the
    # ground. Stretch the object along the up-axis, anchored at its observed top,
    # so the base touches the floor — extends height, preserves lateral size.
    if floor_h is not None:
        h = aligned @ scene_up
        top_h, base_h = np.percentile(h, 98), np.percentile(h, 2)
        height = top_h - base_h
        gap = base_h - floor_h
        if height > 1e-6 and 0 < gap < 0.6 * height:      # floating just above floor
            factor = (top_h - floor_h) / (top_h - base_h)
            new_h = top_h - (top_h - h) * factor
            aligned = aligned + (new_h - h)[:, None] * scene_up
            print(f"    floor contact: stretched base to floor (gap {gap/scene_diag:.4f} → 0)")

    print(f"    fit-to-segment: weld residual {best[0] / scene_diag:.4f} (scene-rel)")
    return aligned


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
    colored_objects = []      # (placed xyz, rgb) per object → colored objects-only mesh
    scene_up, floor_h = scene_up_vector(scene_pc)
    print(f"  scene up (floor normal): {np.round(scene_up, 3)}, floor_h {floor_h:.3f}")
    # SPLATTN completes each object's VGGT partial in place (no register step).
    splattn = Image3DWorker("splattn") if args.image3d == "splattn" else None
    try:
        for obj_name, view_idx, mask, obj_pc in object_records:
            verb = "Completing" if splattn else "Registering"
            print(f"\n  {verb}: {obj_name} (view {view_idx})")
            mask_v = cv2.resize(mask, (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0.5
            valid = mask_v & scene_mask[view_idx]
            target_pts = world_points[view_idx][valid]
            if len(target_pts) < 50:
                print(f"    Only {len(target_pts)} mask points in scene, skipping")
                continue
            if splattn:
                # trim VGGT outliers (depth noise / mask bleed floating off the
                # object) before completion — they inflate the canonicalization
                # scale and make the completion spread into a cloud. Same robust
                # radius trim register_object applies internally.
                med = np.median(target_pts, axis=0)
                rad = np.linalg.norm(target_pts - med, axis=1)
                keep = rad < (np.median(rad) + 2.5 * (np.median(np.abs(rad - np.median(rad))) + 1e-9))
                clean = target_pts[keep] if keep.sum() >= 50 else target_pts
                print(f"    trimmed {len(target_pts) - len(clean)}/{len(target_pts)} outlier pts")
                placed = splattn.complete(clean, scene_up)
                print(f"    SplAttN completed in place: {placed.shape}")
            else:
                if obj_pc is None:
                    print(f"    No object point cloud (--skip_3d), skipping registration")
                    continue
                obj_rgb = obj_pc[:, 3:6] if obj_pc.shape[1] >= 6 else None   # gaussian colour
                obj_pc = obj_pc[:, :3]
                aligned = register_object(obj_pc, target_pts, scene_up, floor_h)
                if obj_rgb is not None:    # full placed object + colour (objects-only render)
                    colored_objects.append((aligned.copy(), obj_rgb))
                # 3D inpainting: keep the observed front exactly (real → true size &
                # position) and graft on ONLY the generated geometry that isn't
                # already observed — the occluded back/sides. The generated object
                # just fills the gap; it never overwrites the real visible surface.
                from scipy.spatial import cKDTree
                d_obs = cKDTree(target_pts).query(aligned)[0]
                # "new" = generated geometry not already observed (the occluded
                # back/sides). Threshold must be OBJECT-relative: a scene-relative
                # one (0.02*scene_diag) is larger than a small object when a big
                # floor inflates scene_diag, so the whole completion gets rejected.
                obj_diag = float(np.linalg.norm(aligned.max(0) - aligned.min(0))) + 1e-9
                new_pts = aligned[d_obs > 0.03 * obj_diag]
                placed = np.concatenate([target_pts, new_pts], axis=0)
                print(f"    kept {len(target_pts)} observed + grafted {len(new_pts)} generated")
            fused.append(placed if splattn else new_pts)   # scene_pc already holds the observed front
            registered_objects.append(placed)
    finally:
        if splattn:
            splattn.close()

    with open(OBJECTS_PATH, "wb") as f:
        pickle.dump(registered_objects, f)
    fused_pc = np.concatenate(fused, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    pcd = pcd.voxel_down_sample(voxel_size=0.005 * scene_diag)
    # denoise: drop floaters (VGGT depth noise + per-view background ghosting) that
    # would otherwise become spurious blobs in the marching-cubes mesh / occupancy.
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    fused_pc = np.asarray(pcd.points)
    print(f"\nFused point cloud (downsampled + denoised): {fused_pc.shape} "
          f"(removed {n_before - len(fused_pc)} outliers)")

    # ── Gravity-align the whole reconstruction ──────────────────────────────
    # VGGT's world frame is arbitrary (often tilted ~45°), so the output looks
    # tilted / objects look "floating" in a viewer even when registration is
    # correct. Rotate everything so the estimated floor normal (scene_up) points
    # up (+Y). Camera extrinsics rotate too so Phase G occupancy stays valid.
    def _R_to(a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
        v = np.cross(a, b); s = float(np.linalg.norm(v)); c = float(a @ b)
        if s < 1e-9:
            return np.eye(3) if c > 0 else -np.eye(3)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    R_g = _R_to(np.asarray(scene_up, float), np.array([0.0, 1.0, 0.0]))
    fused_pc = fused_pc @ R_g.T
    registered_objects = [np.asarray(o) @ R_g.T for o in registered_objects]
    world_points = world_points @ R_g.T                       # for Phase E/G
    extrinsics = np.asarray(extrinsics, float).copy()
    extrinsics[:, :, :3] = extrinsics[:, :, :3] @ R_g.T       # world2cam rot
    print(f"  gravity-aligned output (scene_up {np.round(scene_up,2)} -> +Y)")

    np.save(FUSED_PATH, fused_pc)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "fused_pointcloud.ply"), pcd)

    # ── Colored objects-only reconstruction (SceneComplete-style) ────────────
    # Compose the placed, COLORED per-object gaussians (no floor/wall) into a
    # clean colored point cloud + Poisson surface mesh — a complete textured
    # object scene, distinct from the semantic mesh.
    if colored_objects:
        co_xyz = np.concatenate([a for a, _ in colored_objects], 0) @ R_g.T
        co_rgb = np.clip(np.concatenate([c for _, c in colored_objects], 0), 0, 1)
        cpcd = o3d.geometry.PointCloud()
        cpcd.points = o3d.utility.Vector3dVector(co_xyz)
        cpcd.colors = o3d.utility.Vector3dVector(co_rgb)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "objects_colored.ply"), cpcd)
        try:
            cpcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.03 * scene_diag, max_nn=30))
            cpcd.orient_normals_consistent_tangent_plane(20)
            omesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cpcd, depth=9)
            dens = np.asarray(dens)
            omesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))  # trim balloon
            omesh = omesh.crop(cpcd.get_axis_aligned_bounding_box())
            omesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(os.path.join(OUTPUT_DIR, "objects_mesh.ply"), omesh)
            print(f"  colored objects mesh: {len(omesh.vertices)} verts "
                  f"(objects_colored.ply + objects_mesh.ply)")
        except Exception as e:
            print(f"  colored objects mesh skipped: {e}")

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

    # Objects are labeled from the SAM3 instance masks + VLM names (open-vocab →
    # canonical class); background structure (floor/wall/ceiling/...) from
    # Mask2Former. Per view we build a class map: Mask2Former stuff first, then
    # paint the instance masks on top (objects take precedence). -1 = unlabeled.
    recs_by_view = {}
    for uid, vidx, m, _ in object_records:
        recs_by_view.setdefault(vidx, []).append((uid, m))

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
        cmap = np.full(seg.shape, -1, dtype=np.int32)                 # unlabeled
        for ade_id, iris_label in ADE20K_TO_IRIS.items():            # background stuff
            cmap[seg == ade_id] = IRIS_LABEL_TO_ID[iris_label]
        for uid, m in recs_by_view.get(v, []):                       # objects (priority)
            om = cv2.resize(m, (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0.5
            cmap[om] = IRIS_LABEL_TO_ID[name_to_class(uid)]

        valid = scene_mask[v] & (cmap >= 0)
        ref_pts.append(world_points[v][valid])
        ref_labels.append(cmap[valid])
        print(f"  Labeled view {v}: {os.path.basename(path)} ({int(valid.sum())} ref pts)")

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

# Finer grid → the dilation/smoothing no longer bridges across depth-discontinuity
# gaps (e.g. a tall object's edge to the wall behind it), which is what produced
# the "skirt" / webbing artifacts behind foreground objects in single-view recon.
GRID = 160
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

mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_world)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

# drop tiny disconnected fragments (meshing artifacts), keeping the main surface
# and any object-sized components — threshold is RELATIVE so real objects survive.
tri_clusters, n_tri, _ = mesh_o3d.cluster_connected_triangles()
n_tri = np.asarray(n_tri)
small = n_tri < 0.01 * n_tri.max()
mesh_o3d.remove_triangles_by_mask(small[np.asarray(tri_clusters)])
mesh_o3d.remove_unreferenced_vertices()
verts_world = np.asarray(mesh_o3d.vertices)
print(f"Mesh: {len(verts_world)} vertices, {len(mesh_o3d.triangles)} faces "
      f"({int(small.sum())} fragment component(s) removed)")

vert_tree = KDTree(fused_pc)
_, vi = vert_tree.query(verts_world, k=1)
vert_labels = labels[vi[:, 0]]
vert_colors = np.array([LABEL_COLORS[l] for l in vert_labels])
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
