"""Persistent Amodal3R (occlusion-aware image->3D) worker.

Amodal3R is a TRELLIS fork that reconstructs the *complete* object from a possibly
occluded view, guided by a mask that marks the visible target (and, optionally, the
occluder). It runs in the `tigon` conda env, which already provides the TRELLIS
stack (spconv, xformers, utils3d). Output is an object point cloud (gaussian xyz) —
the same contract as the trellis/tigon workers, so it drops into IRIS Phase B2.

IRIS already segments every object with SAM3, so the worker is handed that mask
(letterboxed to the crop frame) and encodes it the way Amodal3R expects:
    gray  (128) = visible target            (SAM3 mask region)
    white (255) = background
    black (0)   = occluder                  (optional; not used on de-occluded crops)
Using the SAM3 mask — rather than re-deriving it from the white background — keeps
white object parts (e.g. a white bottle cap) inside the target instead of the bg.

Protocol (line-based, "@@" prefix so library log noise on stdout is ignored):
    parent -> worker : {"image": <crop_png>, "mask": <mask_png>, "out": <npy>, "n": 10000}
    worker  -> parent: @@OK <npy>   (or)   @@ERR <message>
"@@READY" once the pipeline is loaded.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import json
import sys

import numpy as np
from PIL import Image

# Some xformers releases moved BlockDiagonalMask under fmha.attn_bias; the TRELLIS
# code (which Amodal3R reuses) expects it on fmha directly. Alias it back if needed.
try:
    import xformers.ops.fmha as _fmha
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask as _BDM
    if not hasattr(_fmha, "BlockDiagonalMask"):
        _fmha.BlockDiagonalMask = _BDM
except Exception:
    pass

sys.path.insert(0, "models/Amodal3R")   # cwd is the repo root → `import amodal3r`

SS_STEPS = int(os.environ.get("AMODAL3R_SS_STEPS", "12"))
SS_CFG = float(os.environ.get("AMODAL3R_SS_CFG", "7.5"))
SLAT_STEPS = int(os.environ.get("AMODAL3R_SLAT_STEPS", "12"))
SLAT_CFG = float(os.environ.get("AMODAL3R_SLAT_CFG", "3"))


def encode_mask(mask_path: str) -> Image.Image:
    """SAM3 binary mask (0/255) -> Amodal3R 3-value mask (gray target, white bg)."""
    binmask = np.array(Image.open(mask_path).convert("L"))
    amask = np.full(binmask.shape, 255, np.uint8)   # background
    amask[binmask > 127] = 128                       # visible target
    return Image.fromarray(amask)


def main():
    real = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from amodal3r.pipelines import Amodal3RImageTo3DPipeline
    pipe = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipe.cuda()
    sys.stdout = real

    print("@@READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            n = int(req.get("n", 10000))
            img = Image.open(req["image"]).convert("RGB")
            mask = encode_mask(req["mask"])
            r = sys.stdout
            sys.stdout = sys.stderr
            with torch.no_grad():
                outputs = pipe.run_multi_image(
                    [img], [mask], seed=1,
                    sparse_structure_sampler_params={"steps": SS_STEPS, "cfg_strength": SS_CFG},
                    slat_sampler_params={"steps": SLAT_STEPS, "cfg_strength": SLAT_CFG},
                    formats=["gaussian"],
                )
            xyz = outputs["gaussian"][0].get_xyz.detach().cpu().numpy()
            sys.stdout = r
            if len(xyz) > n:                       # uniform subsample to n points
                xyz = xyz[np.random.choice(len(xyz), n, replace=False)]
            np.save(req["out"], xyz.astype(np.float32))
            print(f"@@OK {req['out']}", flush=True)
        except Exception as e:
            sys.stdout = real
            print(f"@@ERR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
