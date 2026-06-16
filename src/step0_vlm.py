import argparse
import os
import sys
import torch
import re
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="test.png")
parser.add_argument("--out", default="detected_objects.txt")
args = parser.parse_args()

model_id = config.VLM_ID

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print("Qwen3-VL model loaded.")
image_path = args.image

user_prompt = """List every distinct movable object you can CLEARLY see in this image.
Rules:
- Use a DESCRIPTIVE name: colour + object type, 2-3 words, always SINGULAR.
  Good: "orange bucket", "white chair", "wooden shelf". Bad: "bucket", "chairs", "toy".
- Name specific object types, never the generic word "toy": say "toy block", "toy dinosaur", "stuffed animal".
- If several similar objects are visible, repeat the name once per instance: ["white chair", "white chair"]
- No brand names or text written on objects
- No parts of objects (no "table leg"), no fixed room structure (no wall, floor, ceiling, window, door, mounted shelf)
- Only objects you can clearly and fully identify. Never guess.
- Maximum 12 entries
- Output ONLY a JSON array of strings, nothing else. Example: ["white chair", "white chair", "white table"]"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

from qwen_vl_utils import process_vision_info
image_inputs, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)

inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt",
    padding=True,
)
inputs = inputs.to(model.device)

print("Running VLM to detect objects...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,  
        do_sample=False,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\n" + "="*60)
print("VLM Output:")
print("="*60)
print(response)
print("="*60)

import json
try:
    m = re.search(r"\[.*\]", response, re.DOTALL)
    parsed = json.loads(m.group(0)) if m else []
    object_names = [str(o).strip().lower() for o in parsed if str(o).strip()]
except (ValueError, AttributeError):
    object_names = [obj.strip().lower() for obj in response.split(",") if obj.strip()]

object_names = [re.sub(r'\s*\([^)]*\)', '', obj).strip() for obj in object_names]
object_names = list(dict.fromkeys(object_names))  # de-dup; instances recovered by detector

print(f"\nExtracted {len(object_names)} object types:")
for i, obj in enumerate(object_names, 1):
    print(f"  {i}. {obj}")

with open(args.out, "w") as f:
    for obj in object_names:
        f.write(obj + "\n")

print(f"\nSaved to {args.out}")