"""CPU-only validation of sparse-depth metric scaling on already-saved VGGT data."""
import numpy as np

dm = np.load("depth_map.npy")
H0, W0 = dm.shape
ys, xs = np.where(np.isfinite(dm) & (dm > 0))
sel = np.random.RandomState(0).choice(len(ys), 500, replace=False)
sparse = np.stack([ys[sel], xs[sel], dm[ys[sel], xs[sel]]], 1).astype(np.float32)

d = np.load("output_trellis/vggt_pointmaps.npz", allow_pickle=True)
wp0, extr0 = d["world_points"][0], d["extrinsics"][0]
Hv, Wv = wp0.shape[:2]
rows = np.clip((sparse[:, 0] * Hv / H0).astype(int), 0, Hv - 1)
cols = np.clip((sparse[:, 1] * Wv / W0).astype(int), 0, Wv - 1)
R, t = extr0[:, :3], extr0[:, 3]
P = wp0[rows, cols]
z_vggt = (P @ R.T + t)[:, 2]
zm = sparse[:, 2]
valid = (z_vggt > 1e-6) & np.isfinite(z_vggt) & (zm > 0)
s = float(np.median(zm[valid] / z_vggt[valid]))
print(f"orig {H0}x{W0}, vggt {Hv}x{Wv}, valid {int(valid.sum())}/500")
print(f"VGGT cam-depth median: {np.median(z_vggt[valid]):.4f} (arbitrary scale)")
print(f"sparse depth median:   {np.median(zm[valid]):.4f}")
print(f"=> metric scale s = {s:.4f} (finite&positive: {np.isfinite(s) and s > 0})")
