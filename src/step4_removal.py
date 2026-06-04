import numpy as np
import cv2
from PIL import Image
import pickle
from simple_lama_inpainting import SimpleLama

# Load LaMa
lama = SimpleLama()
print("LaMa loaded.")

def remove_object(image: Image.Image, mask: np.ndarray) -> Image.Image:
    # Dilate mask slightly to avoid edge artifacts
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    mask_pil = Image.fromarray(mask_uint8)
    
    result = lama(image, mask_pil)
    return result

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