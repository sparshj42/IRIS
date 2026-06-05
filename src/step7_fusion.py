import numpy as np
import pickle
import os
import open3d as o3d

def load_pointclouds():
    # Load scene point cloud from VGGT
    scene_pc = np.load("output/scene_pointcloud.npy")
    print(f"Scene point cloud: {scene_pc.shape}")

    # Load object point clouds from TripoSR
    with open("object_pointclouds.pkl", "rb") as f:
        object_pointclouds = pickle.load(f)
    print(f"Object point clouds: {len(object_pointclouds)} objects")

    return scene_pc, object_pointclouds


def align_object_to_scene(obj_pc: np.ndarray, scene_pc: np.ndarray) -> np.ndarray:
    """
    Align object point cloud into scene coordinate space using ICP.
    TripoSR generates objects in their own local coordinate system,
    so we need to register them into the scene.
    """
    # Create open3d point clouds
    obj_o3d = o3d.geometry.PointCloud()
    obj_o3d.points = o3d.utility.Vector3dVector(obj_pc)

    scene_o3d = o3d.geometry.PointCloud()
    scene_o3d.points = o3d.utility.Vector3dVector(scene_pc)

    # Estimate normals (needed for ICP)
    obj_o3d.estimate_normals()
    scene_o3d.estimate_normals()

    # Run ICP registration
    threshold = 0.1  # max correspondence distance
    reg = o3d.pipelines.registration.registration_icp(
        obj_o3d, scene_o3d,
        threshold,
        np.eye(4),  # initial transformation
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print(f"  ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")

    # Apply transformation to object point cloud
    obj_o3d.transform(reg.transformation)
    aligned_pc = np.asarray(obj_o3d.points)

    return aligned_pc


def fuse_pointclouds(scene_pc: np.ndarray, object_pointclouds: dict) -> np.ndarray:
    """
    Fuse all point clouds into one unified point cloud.
    """
    all_points = [scene_pc]

    for obj_name, obj_pc in object_pointclouds.items():
        print(f"\nAligning: {obj_name}")
        try:
            aligned_pc = align_object_to_scene(obj_pc, scene_pc)
            all_points.append(aligned_pc)
            print(f"  Added {len(aligned_pc)} points")
        except Exception as e:
            print(f"  Failed to align {obj_name}: {e}, skipping")

    # Concatenate all point clouds
    fused = np.concatenate(all_points, axis=0)
    print(f"\nFused point cloud: {fused.shape}")

    return fused


def downsample_pointcloud(pc: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """Downsample point cloud using voxel grid to remove duplicates"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    result = np.asarray(pcd_down.points)
    print(f"Downsampled: {pc.shape[0]} → {result.shape[0]} points")
    return result


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Load point clouds
    scene_pc, object_pointclouds = load_pointclouds()

    # Fuse everything together
    print("\nFusing point clouds...")
    fused_pc = fuse_pointclouds(scene_pc, object_pointclouds)

    # Downsample to remove redundant points
    print("\nDownsampling...")
    fused_pc = downsample_pointcloud(fused_pc, voxel_size=0.02)

    # Save
    np.save("output/fused_pointcloud.npy", fused_pc)
    print(f"Saved output/fused_pointcloud.npy")

    # Also save as .ply for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    o3d.io.write_point_cloud("output/fused_pointcloud.ply", pcd)
    print(f"Saved output/fused_pointcloud.ply")