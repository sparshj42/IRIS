"""Benchmark an IRIS reconstruction against ScanNet GT — honest split:
  - VISIBLE F1 / accuracy : observed surface vs GT-visible       (reconstruction quality)
  - OCCLUDED recall       : completions vs GT occluded-in-view    (occlusion recovery)

Single-view recon: alignment uses the clean OBSERVED cloud (best overlap); the same
rigid transform is applied to the occluded completions. With RGB-D the recon is
metric, so the cm numbers are meaningful.

Usage: python scripts/benchmark_scannet.py <iris_out> <scannet_scene_dir> <gt_geom_dir> [tau_m]
"""
import sys, glob, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

iris_dir, scene_dir, gt_dir = sys.argv[1], sys.argv[2], sys.argv[3]
TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05


def pcd(p):
    q = o3d.geometry.PointCloud(); q.points = o3d.utility.Vector3dVector(p); return q


# ── GT: visible (frame depth), full mesh, and occluded-in-view ──────────────
depth = np.load(f"{gt_dir}/depth.npy"); intr = np.load(f"{gt_dir}/intr_depth.npy")
c2w = np.load(f"{gt_dir}/c2w.npy"); w2c = np.linalg.inv(c2w)
H, W = depth.shape; fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
ys, xs = np.where(depth > 0); dd = depth[ys, xs]
gt_vis = (c2w[:3, :3] @ np.stack([(xs-cx)/fx*dd, (ys-cy)/fy*dd, dd], 1).T).T + c2w[:3, 3]
gt_full = np.asarray(o3d.io.read_triangle_mesh(
    glob.glob(f"{scene_dir}/*_vh_clean_2.ply")[0]).sample_points_uniformly(400000).points)

# occluded-in-view = GT points that fall inside the frame but lie BEHIND the
# visible depth at their pixel (i.e. hidden by a foreground surface).
gcam = (w2c[:3, :3] @ gt_full.T).T + w2c[:3, 3]; z = gcam[:, 2]
u = (gcam[:, 0] / np.clip(z, 1e-6, None) * fx + cx)
v = (gcam[:, 1] / np.clip(z, 1e-6, None) * fy + cy)
inv = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
ui = np.clip(u, 0, W-1).astype(int); vi = np.clip(v, 0, H-1).astype(int)
vis_d = depth[vi, ui]
occ = inv & (vis_d > 0) & (z > vis_d + 0.05)        # >5cm behind the visible surface
gt_occ = gt_full[occ]
print(f"GT: visible {len(gt_vis)} | full {len(gt_full)} | occluded-in-view {len(gt_occ)}")

obs = np.load(f"{iris_dir}/observed_pointcloud.npy")
comp = np.load(f"{iris_dir}/completion_pointcloud.npy") if glob.glob(f"{iris_dir}/completion_pointcloud.npy") else None
print(f"IRIS: observed {len(obs)}" + (f" | completions {len(comp)}" if comp is not None else " | (no completions / skip_3d)"))


def align(src_np, tgt_np):  # rigid: FPFH global -> point-to-plane ICP
    s, t = pcd(src_np), pcd(tgt_np); vv = 0.04
    sd, td = s.voxel_down_sample(vv), t.voxel_down_sample(vv)
    for p in (sd, td, s, t):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vv*2, max_nn=30))
    fs, ft = (o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=vv*5, max_nn=100)) for p in (sd, td))
    best = None
    for _ in range(3):
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sd, td, fs, ft, True, vv*1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vv*1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 2000))
        ic = o3d.pipelines.registration.registration_icp(
            s, t, 0.10, r.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        if best is None or ic.fitness > best.fitness:
            best = ic
    return best


# align on the OBSERVED cloud (clean), apply to completions too
a = align(obs, gt_vis); T = a.transformation
def xf(P): return (T[:3, :3] @ P.T).T + T[:3, 3]
obs_w = xf(obs)
print(f"alignment (observed->GT): fitness {a.fitness:.3f}  RMSE {a.inlier_rmse*100:.1f} cm")

# ── VISIBLE region (reconstruction quality) ─────────────────────────────────
d_acc = cKDTree(gt_full).query(obs_w)[0]; d_comp = cKDTree(obs_w).query(gt_vis)[0]
P = (d_acc < TAU).mean(); R = (d_comp < TAU).mean(); F1 = 2*P*R/(P+R+1e-9)
print("\n=== VISIBLE region (observed surface vs GT) ===")
print(f"  Reconstruction accuracy : {d_acc.mean()*100:.1f} cm mean ({np.median(d_acc)*100:.1f} cm median)")
print(f"  Precision@{int(TAU*100)}cm {P:.3f} | Recall@{int(TAU*100)}cm {R:.3f} | F1 {F1:.3f}")

# ── OCCLUDED recall (occlusion recovery) ────────────────────────────────────
print("\n=== OCCLUDED recovery (completions vs GT occluded-in-view) ===")
if comp is not None and len(gt_occ):
    comp_w = xf(comp)
    rec_occ = (cKDTree(comp_w).query(gt_occ)[0] < TAU).mean()
    # how much occluded surface does the OBSERVED-only recon get (baseline ~0)?
    rec_occ_obs = (cKDTree(obs_w).query(gt_occ)[0] < TAU).mean()
    print(f"  Occluded recall @{int(TAU*100)}cm : {rec_occ:.3f}   (observed-only baseline {rec_occ_obs:.3f})")
    print(f"  -> image-to-3D recovers {100*(rec_occ-rec_occ_obs):.1f}% more occluded surface")
else:
    print("  no completions (skip_3d) -> occluded recall not applicable")

print("\n=== KPI SCORECARD ===")
print(f"  Visible F1            {F1:.2f}      (target >0.95, benchmark 0.85)")
print(f"  Reconstruction acc   {d_acc.mean()*100:.1f} cm   (target <2cm, benchmark 5cm)")
