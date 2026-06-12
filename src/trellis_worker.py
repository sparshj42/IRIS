"""Persistent TRELLIS image->3D worker — runs in the pinned `trellis` conda env.

We only decode the GAUSSIAN format and read its point positions (get_xyz), which
needs just the sparse-structure + slat decoders (spconv + xformers) — no mesh
extraction CUDA builds. Output is an object point cloud.

Protocol (line-based, "@@" prefix so library log noise on stdout is ignored):
    parent -> worker : {"image": <crop_png>, "out": <points_npy>, "n": 10000}\n
    worker  -> parent: @@OK <points_npy>   (or)   @@ERR <message>
"@@READY" once the pipeline is loaded.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import json
import sys

import numpy as np
from PIL import Image

# xformers 0.0.28 moved BlockDiagonalMask under fmha.attn_bias; TRELLIS (written
# for an older xformers) expects it on fmha directly. Alias it back.
import xformers.ops.fmha as _fmha
from xformers.ops.fmha.attn_bias import BlockDiagonalMask as _BDM
if not hasattr(_fmha, "BlockDiagonalMask"):
    _fmha.BlockDiagonalMask = _BDM

sys.path.insert(0, "models/TRELLIS")


def main():
    real = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
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
            r = sys.stdout
            sys.stdout = sys.stderr
            with torch.no_grad():
                outputs = pipe.run(img, formats=["gaussian"], preprocess_image=True)
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
