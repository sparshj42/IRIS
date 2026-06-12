"""Run DA3 multi-view on the IRIS synthetic views -> fused world point cloud.
Runs in the `da3` env. Saves output_trellis/da3_scene_points.npy for comparison
against VGGT's scene_pointcloud.npy.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "models/depth-anything-3/src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth

VIEWS_DIR = "output_trellis/synthetic_views"
IMAGE = "data/test3.png"
CONF_THR = float(os.environ.get("DA3_CONF", "0.5"))

view_paths = [IMAGE] + sorted(
    os.path.join(VIEWS_DIR, f) for f in os.listdir(VIEWS_DIR) if f.endswith(".png"))
print(f"DA3 on {len(view_paths)} views")

model = DepthAnything3.from_pretrained("depth-anything/da3-large").to("cuda").eval()
with torch.no_grad():
    pred = model.inference(view_paths, export_format="mini_npz")

depth = torch.as_tensor(np.asarray(pred.depth), dtype=torch.float32, device="cuda")  # (N,H,W)
K = torch.as_tensor(np.asarray(pred.intrinsics), dtype=torch.float32, device="cuda")  # (N,3,3)
c2w = torch.as_tensor(np.asarray(pred.extrinsics), dtype=torch.float32, device="cuda")  # (N,4,4)
N, H, W = depth.shape
print(f"depth {tuple(depth.shape)}, metric={pred.is_metric}")

wp = unproject_depth(depth[None, ..., None], K[None], c2w[None])  # (1,N,H,W,3)
wp = wp[0].reshape(-1, 3).cpu().numpy()

if pred.conf is not None:
    conf = np.asarray(pred.conf).reshape(-1)
    mask = conf > (CONF_THR * conf.max() if conf.max() > 1 else CONF_THR)
    wp = wp[mask]
# drop any non-finite
wp = wp[np.isfinite(wp).all(1)]

np.save("output_trellis/da3_scene_points.npy", wp)
print(f"DA3 scene point cloud: {wp.shape} -> output_trellis/da3_scene_points.npy")
