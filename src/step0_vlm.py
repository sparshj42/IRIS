import torch
import re
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "Qwen/Qwen3-VL-8B-Instruct"  

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print("Qwen3-VL model loaded.")
image_path = "test.png"

user_prompt = user_prompt = """Look at this image carefully. List only the objects you can CLEARLY see.
Rules:
- Maximum 10 objects
- Only real, distinct objects visible in the image
- No sub-parts of objects (e.g. just "train" not "train wheel", "train door")
- No groups of objects, give me just individual objects (e.g. just "dog plushie" not "stuffed animals")
- No guesses — only what you can clearly see
- Output ONLY a comma-separated list, nothing else"""

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

numbered_pattern = r'\d+\.\s*([^,\n]+)'
matches = re.findall(numbered_pattern, response)

if matches:
    object_names = [m.strip() for m in matches]
else:
    object_names = [obj.strip() for obj in response.split(",") if obj.strip()]

object_names = [re.sub(r'\s*\([^)]*\)', '', obj).strip() for obj in object_names]

print(f"\nExtracted {len(object_names)} objects:")
for i, obj in enumerate(object_names, 1):
    print(f"  {i}. {obj}")

with open("detected_objects.txt", "w") as f:
    for obj in object_names:
        f.write(obj + "\n")

print("\nSaved to detected_objects.txt")