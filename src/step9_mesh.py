"""
Step 9: Final semantic mesh extraction via Marching Cubes.

Voxelizes the labeled point cloud into an occupancy grid, runs Marching Cubes,
and transfers per-point semantic labels onto mesh vertices.

Inputs:  output/labeled_pointcloud_{points,labels}.npy (step 8)
Outputs: output/final_semantic_mesh.ply, output/final_mesh_vertex_labels.npy
"""

import numpy as np
import open3d as o3d
from skimage import measure
from scipy import ndimage
from sklearn.neighbors import KDTree

LABEL_COLORS = {
    0: [0.6, 0.4, 0.2],   # floor - brown
    1: [0.8, 0.8, 0.8],   # wall - gray
    2: [0.9, 0.9, 0.7],   # ceiling - light yellow
    3: [0.2, 0.6, 0.9],   # platform - blue
    4: [0.2, 0.8, 0.2],   # other - green
}

GRID = 128

if __name__ == "__main__":
    points = np.load("output/labeled_pointcloud_points.npy")
    labels = np.load("output/labeled_pointcloud_labels.npy")
    print(f"Labeled point cloud: {points.shape}")

    mins = points.min(0)
    span = (points.max(0) - mins).max()
    voxel = span / (GRID - 1)

    idx = np.clip(((points - mins) / voxel).astype(int), 0, GRID - 1)
    occ = np.zeros((GRID, GRID, GRID), dtype=np.float32)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    occ = ndimage.binary_dilation(occ, iterations=1).astype(np.float32)
    occ = ndimage.gaussian_filter(occ, sigma=1.0)

    verts, faces, _, _ = measure.marching_cubes(occ, level=0.5)
    verts_world = verts * voxel + mins
    print(f"Mesh: {len(verts_world)} vertices, {len(faces)} faces")

    # Transfer labels from nearest labeled point
    tree = KDTree(points)
    _, vi = tree.query(verts_world, k=1)
    vert_labels = labels[vi[:, 0]]
    vert_colors = np.array([LABEL_COLORS[l] for l in vert_labels])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("output/final_semantic_mesh.ply", mesh)
    np.save("output/final_mesh_vertex_labels.npy", vert_labels)
    print("Saved output/final_semantic_mesh.ply")
