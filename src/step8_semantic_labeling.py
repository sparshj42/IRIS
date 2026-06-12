"""
Step 8: Semantic labeling of the fused point cloud via multi-view voting.

Instead of back-projecting with guessed camera intrinsics (which never lined up
with VGGT's coordinate frame), we use VGGT's own per-pixel world points: each
pixel of each view already has a 3D location in scene coordinates, so we label
those points from Mask2Former's 2D semantic map and vote onto the fused cloud
with a KD-tree.

Inputs:  output/fused_pointcloud.npy (step 7), output/vggt_pointmaps.npz (step 6)
Outputs: output/labeled_pointcloud_{points,labels}.npy, output/labeled_pointcloud.ply
"""

import torch
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from sklearn.neighbors import KDTree
import open3d as o3d

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


def load_mask2former():
    model_id = "facebook/mask2former-swin-large-ade-semantic"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id, torch_dtype=torch.float16).to("cuda").eval()
    print("Mask2Former loaded.")
    return processor, model


def get_iris_map(processor, model, image: Image.Image, target_size) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device="cuda", dtype=torch.float16) if torch.is_floating_point(v) else v.to("cuda")
              for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[target_size])[0].cpu().numpy()
    iris_map = np.full(seg.shape, IRIS_LABEL_TO_ID["other"], dtype=np.int32)
    for ade_id, iris_label in ADE20K_TO_IRIS.items():
        iris_map[seg == ade_id] = IRIS_LABEL_TO_ID[iris_label]
    return iris_map


if __name__ == "__main__":
    fused_pc = np.load("output/fused_pointcloud.npy")
    data = np.load("output/vggt_pointmaps.npz", allow_pickle=True)
    world_points, conf, view_paths = data["world_points"], data["conf"], data["view_paths"]
    V, Hv, Wv = conf.shape
    scene_mask = conf > 0.5
    scene_pc = world_points[scene_mask]
    scene_diag = np.linalg.norm(scene_pc.max(0) - scene_pc.min(0))
    print(f"Point cloud: {fused_pc.shape}, {V} views")

    processor, model = load_mask2former()

    # Build a labeled reference cloud from VGGT's per-pixel world points
    ref_pts, ref_labels = [], []
    for v, path in enumerate(view_paths):
        print(f"  Processing view: {os.path.basename(str(path))}")
        img = Image.open(str(path)).convert("RGB")
        iris_map = get_iris_map(processor, model, img, (Hv, Wv))
        valid = scene_mask[v]
        ref_pts.append(world_points[v][valid])
        ref_labels.append(iris_map[valid])

    ref_pts = np.concatenate(ref_pts, axis=0)
    ref_labels = np.concatenate(ref_labels, axis=0)

    # Vote: each fused point takes the majority label of its 5 nearest reference points
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

    np.save("output/labeled_pointcloud_labels.npy", labels)
    np.save("output/labeled_pointcloud_points.npy", fused_pc)

    colors = np.array([LABEL_COLORS[l] for l in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("output/labeled_pointcloud.ply", pcd)
    print("\nSaved output/labeled_pointcloud.ply")
