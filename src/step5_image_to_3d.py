import sys
sys.path.insert(0, 'models/TripoSR')

import torch
import numpy as np
from PIL import Image
import pickle
import os
import trimesh
from tsr.system import TSR

def load_triposr():
    model = TSR.from_pretrained(
        'stabilityai/TripoSR',
        config_name='config.yaml',
        weight_name='model.ckpt'
    )
    model = model.to('cuda')
    model.eval()
    print("TripoSR loaded.")
    return model

def mesh_to_pointcloud(mesh, num_points: int = 10000) -> np.ndarray:
    """Sample points uniformly from mesh surface → point cloud"""
    tri_mesh = trimesh.Trimesh(
        vertices=np.array(mesh.vertices),
        faces=np.array(mesh.faces)
    )
    points, _ = trimesh.sample.sample_surface(tri_mesh, num_points)
    return points  # (N, 3)

def image_to_3d(model, image: Image.Image, obj_name: str, output_dir: str) -> np.ndarray:
    os.makedirs(output_dir, exist_ok=True)

    image_resized = image.resize((512, 512))

    with torch.no_grad():
        scene_codes = model([image_resized], device='cuda')
        meshes = model.extract_mesh(scene_codes, resolution=64)

    mesh = meshes[0]

    mesh_path = os.path.join(output_dir, f"{obj_name.replace(' ', '_')}.obj")
    mesh.export(mesh_path)
    print(f"  Saved mesh: {mesh_path}")

    pointcloud = mesh_to_pointcloud(mesh)
    print(f"  Point cloud: {pointcloud.shape}")

    return pointcloud


if __name__ == "__main__":
    model = load_triposr()

    image = Image.open("test.png").convert("RGB")
    with open("masks.pkl", "rb") as f:
        all_masks = pickle.load(f)
    with open("sorted_objects.pkl", "rb") as f:
        sorted_objects = pickle.load(f)

    all_pointclouds = {}

    for obj_name, depth_val in sorted_objects:
        if obj_name not in all_masks:
            continue

        print(f"\nProcessing: {obj_name}")

        mask = all_masks[obj_name]
        mask_binary = (mask > 0.5).astype(np.uint8)

        rows = np.any(mask_binary, axis=1)
        cols = np.any(mask_binary, axis=0)

        if not rows.any():
            continue

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        pad = 20
        h, w = image.size[1], image.size[0]
        rmin = max(0, rmin - pad)
        rmax = min(h, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(w, cmax + pad)

        cropped = image.crop((cmin, rmin, cmax, rmax))

        pc = image_to_3d(model, cropped, obj_name, "output/meshes")
        all_pointclouds[obj_name] = pc

    with open("object_pointclouds.pkl", "wb") as f:
        pickle.dump(all_pointclouds, f)

    print(f"\nSaved {len(all_pointclouds)} point clouds to object_pointclouds.pkl")