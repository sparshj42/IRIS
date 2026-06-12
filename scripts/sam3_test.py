"""Test SAM3 text-prompted segmentation mask quality vs the DINO+SAM2 misses."""
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

device = "cuda"
model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

image = Image.open("data/test3.png").convert("RGB")
W, H = image.size

prompts = ["purple water bottle", "black computer mouse", "black tool case", "white marker"]
overlay = np.array(image).astype(np.float32)
colors = [(255,0,255),(0,255,255),(255,128,0),(0,255,0)]

for prompt, col in zip(prompts, colors):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[(H, W)])[0]
    masks = res.get("masks")
    scores = res.get("scores")
    n = 0 if masks is None else len(masks)
    print(f"{prompt}: {n} instances, scores={[round(float(s),2) for s in scores] if n else []}")
    if n:
        for m in masks:
            mb = m.cpu().numpy() > 0.5 if hasattr(m, "cpu") else np.asarray(m) > 0.5
            overlay[mb] = 0.5*overlay[mb] + 0.5*np.array(col)

Image.fromarray(overlay.clip(0,255).astype(np.uint8)).resize((800,600)).save("sam3_masks.png")
print("saved sam3_masks.png")
