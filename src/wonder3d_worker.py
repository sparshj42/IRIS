"""Persistent Wonder3D image->3D worker — runs in the pinned `wonder3d` conda env.

Wonder3D is two stages: (1) cross-domain multi-view DIFFUSION that turns one
masked object image into 6 orthographic views (color + normals), and (2) a heavy
NeuS/instant-nsr reconstruction. We run stage 1 (the actual Wonder3D model) and
replace stage 2 with a lightweight VISUAL-HULL carve from the 6 silhouettes — the
6 views are at *known fixed orthographic poses*, so silhouette intersection gives
a correctly-shaped object point cloud with no tinycudann/NeuS build. IRIS only
needs the right shape/scale (register_object rescales + orients it), so the hull
is sufficient. (NeuS is the higher-fidelity, much slower alternative.)

Runs with cwd=models/Wonder3D. Protocol (line-based, "@@" prefix):
    parent -> worker : {"image": <crop_png>, "mask": <mask_png>, "out": <npy>, "n": 10000}
    worker  -> parent: @@OK <npy>   (or)   @@ERR <message>
"@@READY" once the pipeline is loaded.
"""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# parent IRIS sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, but this env's
# torch 2.0.1 doesn't recognise that option and aborts at cuda init — drop it.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, ".")            # cwd is models/Wonder3D

VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]
POSE_DIR = "./mvdiffusion/data/fixed_poses/nine_views"


def load_w2c(view):
    """3x4 world->cam RT for a fixed ortho view."""
    rt = np.loadtxt(os.path.join(POSE_DIR, f"000_{view}_RT.txt")).astype(np.float64)
    return rt[:, :3], rt[:, 3]          # R (3,3), t (3,)


def _R_align(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b); s = float(np.linalg.norm(v)); c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def hull_up_to_z():
    """The 6 ortho cameras share an up-direction (their world2cam y-row) — that is
    the object's vertical axis in the hull's world frame, ~[0,0.49,0.87], NOT +Z.
    IRIS's register_object assumes the canonical object is +Z-up (like TRELLIS),
    so rotate the hull so this axis -> +Z. Cameras look down slightly, so cam-y
    points toward the object's TOP; map it to +Z to keep the object upright."""
    ys = np.array([load_w2c(v)[0][1] for v in VIEWS])    # cam-y per view
    up = ys.mean(0)
    return _R_align(up, np.array([0.0, 0.0, 1.0]))


def visual_hull(masks, grid=128, bound=1.0):
    """Silhouette intersection over [-bound,bound]^3 using the 6 fixed ortho
    cameras. ortho image plane spans [-1,1] (get_ortho_ray_directions_origins:
    origin = (i/W-0.5)*2). A voxel is kept only if it projects into foreground in
    EVERY view. Returns surface-voxel world points."""
    lin = np.linspace(-bound, bound, grid).astype(np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)        # (G^3, 3)
    keep = np.ones(len(P), bool)
    for view, m in zip(VIEWS, masks):
        H, W = m.shape
        R, t = load_w2c(view)
        Pc = P @ R.T + t                                        # world->cam
        u = (Pc[:, 0] * 0.5 + 0.5) * W                          # ortho x -> col
        v = (Pc[:, 1] * 0.5 + 0.5) * H                          # ortho y -> row
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        fg = np.zeros(len(P), bool)
        fg[inb] = m[vi[inb], ui[inb]] > 0.5
        keep &= fg
        if not keep.any():
            break
    occ = keep.reshape(grid, grid, grid)
    if not occ.any():
        return np.zeros((0, 3), np.float32)
    # surface voxels = occupied with an empty 6-neighbour (thin shell, fewer pts)
    from scipy import ndimage
    eroded = ndimage.binary_erosion(occ)
    surf = occ & ~eroded
    pts = P[surf.ravel()]
    return pts.astype(np.float32)


def main():
    real = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from einops import rearrange
    from rembg import remove, new_session
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
    from mvdiffusion.data.single_image_dataset import SingleImageDataset

    dtype = torch.float16
    pipe = MVDiffusionImagePipeline.from_pretrained(
        "flamehaze1115/wonder3d-v1.0", torch_dtype=dtype, trust_remote_code=True)
    # Wonder3D's multi-view attention only has the sparse_mv_attention kwarg on its
    # XFormers processor; enabling xformers selects it (the plain processor errors).
    pipe.unet.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rembg_session = new_session("u2net")
    _R_up = hull_up_to_z()               # object up-axis -> +Z (computed from fixed poses)
    sys.stdout = real

    print("@@READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            n = int(req.get("n", 10000))
            r = sys.stdout
            sys.stdout = sys.stderr

            # build RGBA (alpha = object mask) — SingleImageDataset centers by alpha
            rgb = Image.open(req["image"]).convert("RGB")
            if req.get("mask"):
                a = Image.open(req["mask"]).convert("L").resize(rgb.size)
            else:
                a = Image.new("L", rgb.size, 255)
            rgba = Image.merge("RGBA", (*rgb.split(), a))

            ds = SingleImageDataset(root_dir="", num_views=6, img_wh=[256, 256],
                                    bg_color="white", crop_size=192, single_image=rgba)
            batch = ds[0]
            imgs_in = torch.cat([batch["imgs_in"]] * 2, 0).to(dtype).cuda()
            cam = torch.cat([batch["camera_embeddings"]] * 2, 0).to(dtype).cuda()
            task = torch.cat([batch["normal_task_embeddings"],
                              batch["color_task_embeddings"]], 0).to(dtype).cuda()
            cam = torch.cat([cam, task], -1)
            imgs_in = rearrange(imgs_in, "n c h w -> (n) c h w")

            g = torch.Generator(device="cuda").manual_seed(42)
            with torch.no_grad():
                out = pipe(imgs_in, cam, generator=g, guidance_scale=3.0,
                           output_type="pt", num_images_per_prompt=1, eta=1.0).images
            colors = out[out.shape[0] // 2:]                 # 6 color views (3,256,256)

            masks = []
            for i in range(6):
                c = (colors[i].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                rgba_out = remove(c, session=rembg_session)   # u2net alpha
                masks.append((np.asarray(rgba_out)[:, :, 3] > 127).astype(np.float32))

            pts = visual_hull(masks, grid=128, bound=1.0)
            if len(pts) == 0:
                raise RuntimeError("empty visual hull (silhouettes did not intersect)")
            pts = pts @ _R_up.T           # reorient object up-axis -> +Z (register_object convention)
            # center + scale into a unit-ish box (register_object rescales anyway)
            pts = pts - pts.mean(0)
            pts = pts / (np.abs(pts).max() + 1e-9) * 0.5
            if len(pts) > n:
                pts = pts[np.random.choice(len(pts), n, replace=False)]

            sys.stdout = r
            np.save(req["out"], pts.astype(np.float32))
            print(f"@@OK {req['out']}", flush=True)
        except Exception as e:
            sys.stdout = real
            print(f"@@ERR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
