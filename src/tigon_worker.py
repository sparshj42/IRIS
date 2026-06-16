"""Persistent TIGON (text+image -> 3D) worker — runs in the pinned `tigon` conda env.

TIGON is built on TRELLIS infrastructure; we decode the GAUSSIAN format and read
its point positions (get_xyz), giving an object point cloud — the same contract as
trellis_worker.py, so it is a drop-in image-to-3D backend for the IRIS pipeline.

Unlike TRELLIS, TIGON is *text-image* conditioned, so each request may carry the
object's semantic label (from IRIS's Qwen3-VL discovery) as an extra text prompt.

This worker is launched with cwd=models/TIGON so TIGON's relative paths resolve:
  ./mix_e2e_pipe                       (checkpoint)
  ./external/dinov3                    (DINOv3 hubconf repo)
  ./external/dinov3_vith16plus_*.pth   (DINOv3 weights)

Protocol (line-based, "@@" prefix so library log noise on stdout is ignored):
    parent -> worker : {"image": <crop_png>, "out": <points_npy>, "n": 10000,
                        "text": <label>, "seed": 42}\n
    worker  -> parent: @@OK <points_npy>   (or)   @@ERR <message>
"@@READY" once the pipeline is loaded.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")   # avoid the flash-attn build
os.environ.setdefault("SPCONV_ALGO", "native")

import json
import sys

import numpy as np
from PIL import Image

# Some xformers releases moved BlockDiagonalMask under fmha.attn_bias; the TRELLIS
# code (which TIGON reuses) expects it on fmha directly. Alias it back if needed.
try:
    import xformers.ops.fmha as _fmha
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask as _BDM
    if not hasattr(_fmha, "BlockDiagonalMask"):
        _fmha.BlockDiagonalMask = _BDM
except Exception:
    pass

sys.path.insert(0, ".")   # cwd is models/TIGON → `import trellis`

# Generation knobs (match TIGON's demo.py inference settings)
SS_STEPS = int(os.environ.get("TIGON_SS_STEPS", "35"))
SS_CFG = float(os.environ.get("TIGON_SS_CFG", "3"))


def main():
    real = sys.stdout
    sys.stdout = sys.stderr          # mute library prints on the protocol channel
    import torch
    from trellis.pipelines import TrellisE2EInterleaveResCondPipeline

    pipe = TrellisE2EInterleaveResCondPipeline.from_pretrained("mix_e2e_pipe")
    # H100 80GB has room for the full pipeline resident; fall back to sequential
    # offload only if a smaller GPU OOMs on .cuda().
    if os.environ.get("TIGON_ENABLE_OFFLOAD", "0").lower() in {"1", "true", "yes"}:
        pipe.enable_sequential_offload()
    else:
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
            text = req.get("text", "") or ""
            seed = int(req.get("seed", 42))
            img = Image.open(req["image"]).convert("RGB")
            r = sys.stdout
            sys.stdout = sys.stderr
            with torch.no_grad():
                outputs = pipe.run(
                    text,
                    [img],
                    seed=seed,
                    sparse_structure_sampler_params={"steps": SS_STEPS, "cfg_strength": SS_CFG},
                    formats=["gaussian"],
                    preprocess_image=True,
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
