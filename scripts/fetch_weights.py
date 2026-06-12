"""Download / prepare model weights that need explicit handling.

Run (in the `iris` env) after scripts/setup_envs.sh:
    conda run -n iris python scripts/fetch_weights.py

Most models are open-weight and auto-download from Hugging Face on first pipeline
run (Qwen3-VL, SAM3, Depth-Anything-V2, SDXL-inpainting, TRELLIS-image-large,
VGGT, Mask2Former). This script only handles RORem, whose weights live on a
public Google Drive folder (linked from the RORem GitHub README).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── RORem (the remover) — public Google Drive folder (RORem GitHub README) ──
rorem = os.environ.get("IRIS_ROREM_CKPT", os.path.join(REPO, "checkpoints", "RORem"))
ROREM_GDRIVE = "https://drive.google.com/drive/folders/1-ZOLMkifypR2SW0n4pOw6_0iIuHu2Ovy"
if not os.path.exists(os.path.join(rorem, "config.json")):
    os.makedirs(rorem, exist_ok=True)
    try:
        import gdown
        print("[RORem] downloading public checkpoint from Google Drive ...")
        gdown.download_folder(ROREM_GDRIVE, output=rorem, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"[RORem] auto-download failed ({e}).")
        print(f"        Manually download {ROREM_GDRIVE} into {rorem} "
              f"(or `pip install gdown` and rerun).")
else:
    print(f"[RORem] found at {rorem}")

print("\nWeight setup complete.")
