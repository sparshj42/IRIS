import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor

# ── 1. Load Grounding DINO ───────────────────────────────────────────────────
dino_model_id = "IDEA-Research/grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id) 
dino_model = dino_model.to("cuda")
dino_model.eval()

print("Grounding DINO model loaded.")

# ── 2. Load SAM2 ─────────────────────────────────────────────────────────────
sam_model_id = "facebook/sam2-hiera-large"
sam_processor = Sam2Processor.from_pretrained(sam_model_id)
sam_model = Sam2Model.from_pretrained(sam_model_id, torch_dtype=torch.bfloat16)
sam_model = sam_model.to("cuda")
sam_model.eval()

print("SAM2 model loaded.")

# ── 3. Load image ────────────────────────────────────────────────────────────
image_path = "test.png"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
height, width = image_np.shape[:2]

print(f"Image size: {width}x{height}")

# ── 4. Read detected objects from VLM ─────────────────────────────────────────
with open("detected_objects.txt", "r") as f:
    object_names = [line.strip() for line in f if line.strip()]

print(f"\nDetected {len(object_names)} objects:")
for obj in object_names:
    print(f"  - {obj}")

# ── 5. Grounding DINO: detect ALL objects in one pass ────────────────────────
# Grounding DINO expects all objects as one prompt separated by " . "
combined_prompt = " . ".join(object_names) + " ."
print(f"\nPrompt: {combined_prompt}")

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

boxes = results[0]["boxes"]      # [N, 4]
scores = results[0]["scores"]    # [N]
labels = results[0]["labels"]    # [N] - detected object names

print(f"Found {len(boxes)} detections:")
for box, score, label in zip(boxes, scores, labels):
    print(f"  {label}: {score:.2f}")

# ── 6. SAM2: segment each detected box ───────────────────────────────────────
all_masks = {}
all_boxes = {}

for box, score, label in zip(boxes, scores, labels):
    box_np = box.cpu().numpy()
    all_boxes[label] = box_np

    sam_inputs = sam_processor(images=image, input_boxes=[[box_np.tolist()]], return_tensors="pt")
    sam_inputs = {
        k: v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda")
        for k, v in sam_inputs.items()
    }

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_outputs.pred_masks
    if masks is not None and len(masks) > 0:
        import cv2
        mask = masks[0, 0, 0].cpu().float().numpy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        all_masks[label] = mask
        print(f"  Mask created for: {label}")

# ── 6. Visualize ─────────────────────────────────────────────────────────────
num_objects = len(all_masks)
if num_objects == 0:
    print("\nNo masks created!")
    exit()

fig, axes = plt.subplots(1, num_objects + 1, figsize=(4 * (num_objects + 1), 4))

axes[0].imshow(image)
for obj_name, box in all_boxes.items():
    x0, y0, x1, y1 = box
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', linewidth=2)
    axes[0].add_patch(rect)
axes[0].set_title("Detections")
axes[0].axis("off")

colors = plt.cm.tab10(np.linspace(0, 1, num_objects))
for idx, (obj_name, mask) in enumerate(all_masks.items()):
    axes[idx + 1].imshow(image)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    colored_mask[mask > 0.5] = colors[idx][:3]
    axes[idx + 1].imshow(colored_mask, alpha=0.5)
    axes[idx + 1].set_title(obj_name[:15])
    axes[idx + 1].axis("off")

plt.tight_layout()
plt.savefig("grounding_output.png", dpi=100)
print(f"\nSaved grounding_output.png with {num_objects} masks")

import pickle
with open("masks.pkl", "wb") as f:
    pickle.dump(all_masks, f)
print("Saved masks.pkl")