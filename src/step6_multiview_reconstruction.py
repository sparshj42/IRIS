import torch
import numpy as np
import pickle
import os
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def load_vggt():
    device = "cuda"
    dtype = torch.float32

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device, dtype=dtype)
    print("VGGT loaded.")
    return model, device, dtype


def synthetic_views_to_pointcloud(model, device, dtype, image_paths: list) -> np.ndarray:
    """
    Takes synthetic views from peeling loop → scene point cloud
    This is the novel use: same camera position, progressively fewer objects
    """
    # Load and preprocess all views
    images = load_and_preprocess_images(image_paths).to(device, dtype=dtype)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Extract point map — shape (N_views, H, W, 3)
    point_map = predictions["world_points"]  # 3D points in world space
    conf = predictions["world_points_conf"]  # confidence scores

    # Flatten to point cloud, filter by confidence
    points = point_map.reshape(-1, 3).cpu().float().numpy()
    confidences = conf.reshape(-1).cpu().float().numpy()

    # Keep only high-confidence points
    mask = confidences > 0.5
    points = points[mask]

    print(f"  Scene point cloud: {points.shape}")
    return points


if __name__ == "__main__":
    # Load synthetic views saved by pipeline.py
    views_dir = "output/synthetic_views"
    image_paths = sorted([
        os.path.join(views_dir, f)
        for f in os.listdir(views_dir)
        if f.endswith(".png")
    ])

    # Also include original image as first view
    image_paths = ["test.png"] + image_paths

    print(f"Using {len(image_paths)} synthetic views")
    for p in image_paths:
        print(f"  {p}")

    model, device, dtype = load_vggt()

    scene_pointcloud = synthetic_views_to_pointcloud(model, device, dtype, image_paths)

    # Save
    os.makedirs("output", exist_ok=True)
    np.save("output/scene_pointcloud.npy", scene_pointcloud)
    print(f"Saved scene point cloud: {scene_pointcloud.shape}")