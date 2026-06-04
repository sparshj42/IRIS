import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pickle
import cv2

# ── 1. Load outputs from previous steps ──────────────────────────────────────
image_path = "test.png"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Load masks from step1
with open("masks.pkl", "rb") as f:
    all_masks = pickle.load(f)

# Load depth map from step2
depth_map = np.load("depth_map.npy")

print(f"Loaded {len(all_masks)} masks")
print(f"Depth map shape: {depth_map.shape}")

depth_map = depth_map.astype(np.float32)
depth_map = cv2.resize(depth_map, (image_np.shape[1], image_np.shape[0]), 
                        interpolation=cv2.INTER_LINEAR)
print(f"Depth map resized to: {depth_map.shape}")
# ── 2. For each mask, find minimum depth (nearest point) ─────────────────────
object_depths = {}

for obj_name, mask in all_masks.items():
    # Get depth values only inside this mask
    mask_binary = mask > 0.5
    
    if mask_binary.sum() == 0:
        print(f"  {obj_name}: empty mask, skipping")
        continue
    
    depth_values = depth_map[mask_binary]
    

    min_depth = depth_values.min()  
    object_depths[obj_name] = min_depth
    
    print(f"  {obj_name}: nearest depth = {min_depth:.3f}")

# ── 3. Sort objects nearest to farthest ──────────────────────────────────────
sorted_objects = sorted(object_depths.items(), key=lambda x: x[1], reverse=False)  

print("\n" + "="*50)
print("Peeling order (nearest → farthest):")
print("="*50)
for idx, (obj_name, depth) in enumerate(sorted_objects, 1):
    print(f"  {idx}. {obj_name} (depth: {depth:.3f})")

# ── 4. Visualize depth ordering ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: image with masks colored by depth order
axes[0].imshow(image)
cmap = plt.cm.RdYlGn_r  # Red = nearest, Green = farthest
n = len(sorted_objects)
for idx, (obj_name, depth) in enumerate(sorted_objects):
    mask = all_masks[obj_name]
    color = cmap(idx / max(n - 1, 1))[:3]
    colored_mask = np.zeros((*mask.shape, 4))
    colored_mask[mask > 0.5] = [*color, 0.5]
    axes[0].imshow(colored_mask)
    # Label each mask
    mask_coords = np.where(mask > 0.5)
    if len(mask_coords[0]) > 0:
        cy = int(mask_coords[0].mean())
        cx = int(mask_coords[1].mean())
        axes[0].text(cx, cy, str(idx), color='white', fontsize=8,
                    ha='center', va='center', fontweight='bold')

axes[0].set_title("Depth Order (Red=Nearest, Green=Farthest)")
axes[0].axis("off")

# Right: depth map
axes[1].imshow(depth_map, cmap='inferno')
axes[1].set_title("Depth Map")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("depth_ordering.png", dpi=100)
print("\nSaved depth_ordering.png")

# Save sorted order for next step
with open("sorted_objects.pkl", "wb") as f:
    pickle.dump(sorted_objects, f)
print("Saved sorted_objects.pkl")