"""
Step 7: Register TRELLIS object point clouds into the VGGT scene and fuse.

TRELLIS objects live in their own local frame, so plain ICP from identity
never converges (fitness 0). Instead we use the VGGT point maps: the pixels of an
object's mask in the view where it was extracted give us exactly where that object
sits in scene coordinates. We scale + translate the object cloud onto those points,
then refine with ICP.

Inputs:  output/vggt_pointmaps.npz (step 6), object_records.pkl (pipeline)
         or falls back to masks.pkl + object_pointclouds.pkl (steps 1/5, view 0)
Outputs: output/fused_pointcloud.{npy,ply}
"""

import numpy as np
import pickle
import os
import cv2
import open3d as o3d


def load_inputs():
    data = np.load("output/vggt_pointmaps.npz", allow_pickle=True)
    world_points, conf = data["world_points"], data["conf"]

    if os.path.exists("object_records.pkl"):
        with open("object_records.pkl", "rb") as f:
            records = pickle.load(f)  # (name, view_idx, mask, pc)
    else:
        # standalone fallback: masks (view 0) + per-object point clouds from prior phases
        with open("masks.pkl", "rb") as f:
            masks = pickle.load(f)
        with open("object_pointclouds.pkl", "rb") as f:
            pcs = pickle.load(f)
        records = [(n, 0, masks[n], pc) for n, pc in pcs.items() if n in masks]

    return world_points, conf, records


def register_object(obj_pc, target_pts, scene_diag):
    """Scale+centroid init from mask-region scene points, then ICP refine."""
    src_diag = np.linalg.norm(obj_pc.max(0) - obj_pc.min(0))
    tgt_diag = np.linalg.norm(target_pts.max(0) - target_pts.min(0))
    scale = tgt_diag / max(src_diag, 1e-8)
    init = obj_pc * scale + (target_pts.mean(0) - obj_pc.mean(0) * scale)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(init)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, 0.05 * scene_diag, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    src.transform(reg.transformation)
    print(f"  ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
    return np.asarray(src.points)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    world_points, conf, records = load_inputs()
    V, Hv, Wv = conf.shape
    scene_mask = conf > 0.5
    scene_pc = world_points[scene_mask]
    scene_diag = np.linalg.norm(scene_pc.max(0) - scene_pc.min(0))
    print(f"Scene point cloud: {scene_pc.shape}, {len(records)} objects")

    fused = [scene_pc]
    for name, view_idx, mask, obj_pc in records:
        print(f"\nRegistering: {name} (view {view_idx})")
        mask_v = cv2.resize(mask, (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0.5
        target_pts = world_points[view_idx][mask_v & scene_mask[view_idx]]
        if len(target_pts) < 50:
            print(f"  Only {len(target_pts)} mask points in scene, skipping")
            continue
        aligned = register_object(obj_pc, target_pts, scene_diag)
        fused.append(aligned)
        print(f"  Added {len(aligned)} points")

    fused_pc = np.concatenate(fused, axis=0)
    print(f"\nFused point cloud: {fused_pc.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    pcd = pcd.voxel_down_sample(voxel_size=0.005 * scene_diag)
    fused_pc = np.asarray(pcd.points)
    print(f"Downsampled: {fused_pc.shape}")

    np.save("output/fused_pointcloud.npy", fused_pc)
    o3d.io.write_point_cloud("output/fused_pointcloud.ply", pcd)
    print("Saved output/fused_pointcloud.npy and .ply")
