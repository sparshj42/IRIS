import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
import pickle
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
ROREM_CKPT = config.ROREM_CKPT
SDXL_BASE = config.SDXL_BASE

# Load RORem — use CPU offload to avoid exhausting system RAM and VRAM
pipe = AutoPipelineForInpainting.from_pretrained(SDXL_BASE, torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(ROREM_CKPT, torch_dtype=torch.float16)
pipe.unet = unet
pipe.enable_model_cpu_offload()  # streams components to GPU only during inference
print("RORem loaded.")

PROMPT = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
NEG_PROMPT = "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background"

def remove_object(image: Image.Image, mask: np.ndarray) -> Image.Image:
    orig_size = image.size
    # Generous dilation to catch soft shadows around the object
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((25, 25), np.uint8)
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

    image_512 = image.resize((512, 512)).convert("RGB")
    mask_512 = Image.fromarray(mask_uint8).resize((512, 512)).convert("L")

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        height=512,
        width=512,
        image=image_512,
        mask_image=mask_512,
        guidance_scale=1.0,
        num_inference_steps=25,
        strength=0.99,
    ).images[0]
    result = result.resize(orig_size, Image.LANCZOS)

    # Feathered composite: only the masked region is replaced, so the rest of
    # the image never round-trips through the VAE (no cumulative blur)
    alpha = cv2.GaussianBlur(mask_uint8.astype(np.float32) / 255.0, (31, 31), 0)[..., None]
    out = np.array(image).astype(np.float32) * (1 - alpha) + np.array(result).astype(np.float32) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

# ── Test on nearest object ────────────────────────────────────────────────────
image = Image.open("test.png").convert("RGB")

with open("masks.pkl", "rb") as f:
    all_masks = pickle.load(f)

with open("sorted_objects.pkl", "rb") as f:
    sorted_objects = pickle.load(f)

nearest_obj = sorted_objects[0][0]
print(f"Removing: {nearest_obj}")

result = remove_object(image, all_masks[nearest_obj])
result.save(f"removal_test.png")
print("Saved removal_test.png")