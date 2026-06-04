import torch
import numpy as np
import cv2
import pickle
import os
from PIL import Image

# ── Imports from our steps ───────────────────────────────────────────────────
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    Sam2Model, Sam2Processor,
    AutoModelForDepthEstimation, AutoImageProcessor,
    Qwen3VLForConditionalGeneration
)
from simple_lama_inpainting import SimpleLama
from qwen_vl_utils import process_vision_info

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
IMAGE_PATH = "test.png"
OUTPUT_DIR = "output/synthetic_views"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: LOAD ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Loading models...")
print("="*60)

# VLM
vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
vlm_model.eval()
print("✓ VLM loaded")

# Grounding DINO
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = dino_model.to("cuda")
dino_model.eval()
print("✓ Grounding DINO loaded")

# SAM2
sam_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
sam_model = Sam2Model.from_pretrained("facebook/sam2-hiera-large", torch_dtype=torch.bfloat16)
sam_model = sam_model.to("cuda")
sam_model.eval()
print("✓ SAM2 loaded")

# Depth
depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Large-hf",
    torch_dtype=torch.float16
)
depth_model = depth_model.to("cuda")
depth_model.eval()
print("✓ DepthAnything loaded")

# LaMa
lama = SimpleLama()
print("✓ LaMa loaded")

print("\nAll models loaded!")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_object_names(image: Image.Image) -> list:
    """VLM: image → list of object names"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": """Look at this image carefully. List only the objects you can CLEARLY see.
Rules:
- Maximum 10 objects
- Only real, distinct objects visible in the image
- No sub-parts of objects
- No guesses
- Output ONLY a comma-separated list, nothing else"""}
        ]
    }]
    
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages, image_patch_size=vlm_processor.image_processor.patch_size)
    inputs = vlm_processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    inputs = inputs.to(vlm_model.device)
    
    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    object_names = [obj.strip() for obj in response.split(",") if obj.strip()]
    return object_names


def get_masks(image: Image.Image, object_names: list) -> dict:
    """Grounding DINO + SAM2: image + names → masks dict"""
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    combined_prompt = " . ".join(object_names) + " ."
    
    inputs = dino_processor(images=image, text=combined_prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=0.3,
        target_sizes=[(height, width)]
    )
    
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]
    
    all_masks = {}
    for box, score, label in zip(boxes, scores, labels):
        box_np = box.cpu().numpy()
        
        sam_inputs = sam_processor(images=image, input_boxes=[[box_np.tolist()]], return_tensors="pt")
        sam_inputs = {
            k: v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda")
            for k, v in sam_inputs.items()
        }
        
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)
        
        masks = sam_outputs.pred_masks
        if masks is not None and len(masks) > 0:
            mask = masks[0, 0, 0].cpu().float().numpy()
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            all_masks[label] = mask
    
    return all_masks


def get_depth_map(image: Image.Image) -> np.ndarray:
    """DepthAnything: image → depth map"""
    inputs = depth_processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = depth_model(**inputs)
    
    depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    
    # Resize to image size
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    depth_map = depth_map.astype(np.float32)
    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return depth_map


def order_masks_by_depth(all_masks: dict, depth_map: np.ndarray) -> list:
    """Sort masks nearest to farthest using min depth of each object"""
    object_depths = {}
    for obj_name, mask in all_masks.items():
        mask_binary = mask > 0.5
        if mask_binary.sum() == 0:
            continue
        depth_values = depth_map[mask_binary]
        object_depths[obj_name] = depth_values.min()  # min = farthest point
    
    sorted_objects = sorted(object_depths.items(), key=lambda x: x[1], reverse=False)
    return sorted_objects


def remove_object(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """LaMa: image + mask → image with object removed"""
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    mask_pil = Image.fromarray(mask_uint8)
    return lama(image, mask_pil)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("Running IRIS pipeline...")
print("="*60)

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {image.size}")

# Step 0: VLM → object names
print("\n[Step 0] Detecting objects with VLM...")
object_names = get_object_names(image)
print(f"Found {len(object_names)} objects: {object_names}")

# Step 1: Grounding DINO + SAM2 → masks
print("\n[Step 1] Segmenting objects...")
# TODO: re-evaluate masks on each peeled image (currently computed once)
all_masks = get_masks(image, object_names)
print(f"Created {len(all_masks)} masks")

# Step 2: DepthAnything → depth map
print("\n[Step 2] Computing depth map...")
depth_map = get_depth_map(image)
print(f"Depth map shape: {depth_map.shape}")

# Step 3: Order masks by depth
print("\n[Step 3] Ordering objects by depth...")
sorted_objects = order_masks_by_depth(all_masks, depth_map)
print("Peeling order:")
for idx, (name, depth) in enumerate(sorted_objects, 1):
    print(f"  {idx}. {name} (depth: {depth:.3f})")

# Step 4: Iterative occlusion peeling loop
print("\n[Step 4] Iterative occlusion peeling...")
current_image = image.copy()
synthetic_views = [image]  # First view is the original

for idx, (obj_name, depth) in enumerate(sorted_objects):
    print(f"\n  Peeling [{idx+1}/{len(sorted_objects)}]: {obj_name}")
    
    if obj_name not in all_masks:
        print(f"  No mask found, skipping")
        continue
    
    mask = all_masks[obj_name]
    
    # Remove object from current image
    current_image = remove_object(current_image, mask)
    
    # Store as synthetic view
    view_path = os.path.join(OUTPUT_DIR, f"view_{idx+1:02d}_{obj_name.replace(' ', '_')}.png")
    current_image.save(view_path)
    synthetic_views.append(current_image.copy())
    
    print(f"  Saved synthetic view: {view_path}")

print(f"\n{'='*60}")
print(f"Peeling complete!")
print(f"Generated {len(synthetic_views)} synthetic views")
print(f"Saved to: {OUTPUT_DIR}/")

# Save synthetic views list for next step (multi-view reconstruction)
with open("synthetic_views.pkl", "wb") as f:
    pickle.dump([np.array(v) for v in synthetic_views], f)
print("Saved synthetic_views.pkl")