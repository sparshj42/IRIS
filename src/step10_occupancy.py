"""
Step 10 — Free / Occupied / Occluded occupancy classification.

Builds a voxel occupancy grid over the reconstructed scene by ray-casting from the
camera through VGGT's per-pixel world points:

  • FREE     — a camera ray passed through the voxel before hitting a surface
  • OCCUPIED — a reconstructed surface (revealed scene OR fused object) is there
  • OCCLUDED — never observed by any ray and not reconstructed (the default)

Because the synthetic views are SAME-POSE, each peeled view re-casts the same rays
but reaches deeper, so previously-occluded space behind a removed object gets
re-marked free/occupied automatically — peeling resolves occlusion. Object backs
are marked occupied via the fused TRELLIS reconstruction (learned-prior completion).

Inputs : output/vggt_pointmaps.npz (world_points, conf, extrinsics), fused cloud
Outputs: output/occupancy_grid.npy (int8: 0=occluded,1=free,2=occupied),
         output/occupancy_meta.npz (grid_min, voxel, dims), occupancy_render.png
"""
import os
import numpy as np

OCCLUDED, FREE, OCCUPIED = 0, 1, 2
LABEL_NAME = {OCCLUDED: "occluded", FREE: "free", OCCUPIED: "occupied"}


def camera_center(extr):
    """world-to-camera [R|t] (3,4) -> camera centre in world = -R^T t."""
    R, t = extr[:, :3], extr[:, 3]
    return -R.T @ t


def _voxelize(pts, grid_min, voxel, dims):
    idx = np.floor((pts - grid_min) / voxel).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < dims), axis=1)
    return idx[ok]


def _solidify(occ, obj_pts, grid_min, voxel, dims):
    """Mark every grid voxel inside an object's convex hull as OCCUPIED.
    Conservative (slight overfill) so objects are solid volumes, not hollow shells,
    and so peeled-view free-marking can't carve out their interior. Safe for a
    navigation/occupancy map (over-estimating occupied is the safe error)."""
    from scipy.spatial import Delaunay
    if len(obj_pts) < 4:
        return
    lo = np.floor((obj_pts.min(0) - grid_min) / voxel).astype(int)
    hi = np.ceil((obj_pts.max(0) - grid_min) / voxel).astype(int)
    lo = np.clip(lo, 0, dims - 1)
    hi = np.clip(hi, 0, dims - 1)
    gx, gy, gz = (np.arange(lo[i], hi[i] + 1) for i in range(3))
    if min(len(gx), len(gy), len(gz)) == 0:
        return
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing="ij")
    cells = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    centers = grid_min + (cells + 0.5) * voxel
    try:
        inside = Delaunay(obj_pts).find_simplex(centers) >= 0
    except Exception:
        return
    c = cells[inside]
    occ[c[:, 0], c[:, 1], c[:, 2]] = OCCUPIED


def compute_occupancy(world_points, conf, extrinsics, fused_pc, object_clouds=None,
                      grid=160, conf_thr=0.5, pixel_stride=2, surface_eps=0.97):
    V, H, W, _ = world_points.shape

    # grid over the fused reconstruction extent (+5% margin), capped to the scene
    lo = fused_pc.min(0)
    hi = fused_pc.max(0)
    margin = 0.05 * (hi - lo)
    grid_min, grid_max = lo - margin, hi + margin
    voxel = float((grid_max - grid_min).max() / grid)
    dims = np.ceil((grid_max - grid_min) / voxel).astype(int) + 1
    occ = np.full(tuple(dims), OCCLUDED, dtype=np.int8)

    # ---- FREE: march camera -> surface for every (sub-sampled) valid pixel ----
    for v in range(V):
        C = camera_center(extrinsics[v])
        m = conf[v] > conf_thr
        P = world_points[v][m][::pixel_stride]          # (N,3) surface hits
        if len(P) == 0:
            continue
        d = P - C[None, :]
        L = np.linalg.norm(d, axis=1)
        steps = int(np.ceil(L.max() / voxel)) + 1
        frac = (np.arange(steps)[:, None] * voxel) / np.maximum(L[None, :], 1e-6)  # (S,N)
        keep = frac <= surface_eps                       # stop just before the surface
        pts = C[None, None, :] + frac[..., None] * d[None]   # (S,N,3)
        pts = pts[keep]
        idx = _voxelize(pts, grid_min, voxel, dims)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = FREE

    # ---- OCCUPIED: revealed surfaces (all views) + fused reconstruction ----
    surf = world_points[conf > conf_thr]
    for pts in (surf, fused_pc):
        idx = _voxelize(pts, grid_min, voxel, dims)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = OCCUPIED

    # ---- SOLIDIFY objects: fill each reconstructed object as a solid volume ----
    for obj_pts in (object_clouds or []):
        _solidify(occ, np.asarray(obj_pts), grid_min, voxel, dims)

    return occ, grid_min, voxel, dims


def occupancy_points(occ, grid_min, voxel):
    """voxel grid -> (points, labels) at voxel centres for in-frustum voxels."""
    out_pts, out_lab = [], []
    for lab in (OCCLUDED, FREE, OCCUPIED):
        ijk = np.argwhere(occ == lab)
        if len(ijk):
            out_pts.append(grid_min + (ijk + 0.5) * voxel)
            out_lab.append(np.full(len(ijk), lab))
    return np.concatenate(out_pts), np.concatenate(out_lab)


def render(occ, grid_min, voxel, path="occupancy_render.png"):
    """Orthogonal cross-section slices — clearest view of free/occupied/occluded
    (free is a frustum cone from the camera; occupied at surfaces; occluded behind)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([(0.5, 0.5, 0.9), (0.6, 0.95, 0.6), (0.9, 0.2, 0.2)])
    dims = occ.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis, name in zip(axes, [0, 1, 2], ["X", "Y", "Z"]):
        counts = [(np.take(occ, i, axis=axis) == OCCUPIED).sum() for i in range(dims[axis])]
        i = int(np.argmax(counts))
        ax.imshow(np.take(occ, i, axis=axis).T, origin="lower", cmap=cmap,
                  vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"slice ⟂ {name} @ {i}"); ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=12, label=l)
               for c, l in [((0.6, 0.95, 0.6), "free"), ((0.9, 0.2, 0.2), "occupied"),
                            ((0.5, 0.5, 0.9), "occluded")]]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    plt.tight_layout(); plt.savefig(path, dpi=95, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--grid", type=int, default=160)
    args = ap.parse_args()

    import pickle
    d = np.load(os.path.join(args.output_dir, "vggt_pointmaps.npz"), allow_pickle=True)
    fused = np.load(os.path.join(args.output_dir, "fused_pointcloud.npy"))
    objs_path = os.path.join(args.output_dir, "registered_objects.pkl")
    objs = pickle.load(open(objs_path, "rb")) if os.path.exists(objs_path) else []
    occ, gmin, voxel, dims = compute_occupancy(
        d["world_points"], d["conf"], d["extrinsics"], fused,
        object_clouds=objs, grid=args.grid)

    total = int(np.prod(dims))
    print(f"Occupancy grid {tuple(dims)} (voxel={voxel:.4f}):")
    for lab in (FREE, OCCUPIED, OCCLUDED):
        n = int((occ == lab).sum())
        print(f"  {LABEL_NAME[lab]:9s}: {n:>9d}  ({100*n/total:5.1f}%)")

    np.save(os.path.join(args.output_dir, "occupancy_grid.npy"), occ)
    np.savez(os.path.join(args.output_dir, "occupancy_meta.npz"),
             grid_min=gmin, voxel=voxel, dims=dims)
    render(occ, gmin, voxel, os.path.join(args.output_dir, "occupancy_render.png"))
