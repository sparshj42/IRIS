import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

parser = argparse.ArgumentParser()
parser.add_argument("--sparse_depth", default=None,
                    help="optional .npy of (row, col, metric_depth) rows for metric scale recovery")
args = parser.parse_args()

model_id = "depth-anything/Depth-Anything-V2-Large-hf"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=torch.float16)
model = model.to("cuda")   
model.eval()              

print("Model loaded.")

image_path = "test.png"          
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()} 

with torch.no_grad(): 
    outputs = model(**inputs)

depth_map = outputs.predicted_depth
depth_map = depth_map.squeeze()
depth_map = depth_map.cpu().numpy()

# Optional sparse-depth-guided metric scale recovery (blueprint Step 2):
# least-squares fit of scale/shift aligning relative depth to sparse metric points
if args.sparse_depth:
    sparse = np.load(args.sparse_depth)  # (N, 3): row, col, metric_depth
    h, w = depth_map.shape
    rows = np.clip((sparse[:, 0] * h / Image.open(image_path).height).astype(int), 0, h - 1)
    cols = np.clip((sparse[:, 1] * w / Image.open(image_path).width).astype(int), 0, w - 1)
    d = depth_map[rows, cols]
    A = np.stack([d, np.ones_like(d)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, sparse[:, 2], rcond=None)
    depth_map = s * depth_map + t
    print(f"Metric scale recovered: scale={s:.4f}, shift={t:.4f} from {len(sparse)} sparse points")

depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_normalized = (depth_normalized * 255).astype(np.uint8)

np.save("depth_map.npy", depth_map)
print("Saved depth_map.npy")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(image);                    plt.title("Input"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(depth_normalized, cmap="inferno"); plt.title("Depth"); plt.axis("off")
plt.tight_layout()
plt.savefig("depth_output.png")
print("Saved depth_output.png")

print(f"Depth map shape: {depth_map.shape}, min: {depth_map.min():.2f}, max: {depth_map.max():.2f}")