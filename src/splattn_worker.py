"""Persistent SplAttN point-cloud-completion worker (runs in the `splattn` env).

Unlike the image-to-3D backends (trellis/tigon/amodal3r), SplAttN does not take an
object crop and produce a canonical object that must then be registered. It takes
the object's VGGT *partial point cloud* — already in scene coordinates — and
completes it in place, so no registration step is needed.

The partial is canonicalized to the model's expected frame (gravity→+Y, unit
sphere) before inference and un-canonicalized back to scene coordinates after, so
the completed object lands exactly where the observed segment is.

Runs with cwd=models/SplAttN. Protocol (line-based, "@@" prefix):
    parent -> worker : {"partial": <npy>, "up": [x,y,z], "out": <npy>}
    worker  -> parent: @@OK <npy>   (or)   @@ERR <message>
"@@READY" once the model is loaded.
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

import sys
import json
sys.path.insert(0, ".")            # cwd is models/SplAttN
sys.path.insert(0, "KNN_CUDA")

import numpy as np


def _R_align(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def main():
    real = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from config_55 import cfg
    # config_55.py is the PCN config copy; fix it to the released ShapeNet-55 model.
    cfg.NETWORK.step1 = 2
    cfg.NETWORK.step2 = 4
    cfg.DATASET.TEST_DATASET = "ShapeNet55"   # → self_attention decoder (matches ckpt)
    from models.SplAttN import Model
    from models.model_utils import SoftSplatCCM
    from pointnet2_ops import pointnet2_utils

    model = Model(cfg).cuda().eval()
    ck = torch.load("ckpts/shapenet55/splattn-55.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    render = SoftSplatCCM(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224,
                          kernel_size=cfg.NETWORK.splat_kernel, sigma=cfg.NETWORK.splat_sigma)
    sys.stdout = real

    print("@@READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            P = np.load(req["partial"]).astype(np.float32)
            up = np.asarray(req["up"], np.float32)
            r = sys.stdout
            sys.stdout = sys.stderr
            # canonicalize: gravity → +Y, center, unit sphere
            Rg = _R_align(up, np.array([0, 1.0, 0]))
            c0 = P.mean(0)
            Pc = (P - c0) @ Rg.T
            scale = float(np.linalg.norm(Pc, axis=1).max()) + 1e-9
            Pn = Pc / scale
            if len(Pn) < 2048:                      # pad short partials
                Pn = np.concatenate([Pn, Pn[np.random.choice(len(Pn), 2048 - len(Pn))]], 0)
            part = torch.tensor(Pn, dtype=torch.float32).cuda().unsqueeze(0)
            idx = pointnet2_utils.furthest_point_sample(part.contiguous(), 2048)
            part = pointnet2_utils.gather_operation(
                part.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
            depth = render.get_CCM(part)
            with torch.no_grad():
                pred = model(part.contiguous(), depth)[-1][0].cpu().numpy()
            completed = (pred * scale) @ Rg + c0      # back to scene coordinates
            sys.stdout = r
            np.save(req["out"], completed.astype(np.float32))
            print(f"@@OK {req['out']}", flush=True)
        except Exception as e:
            sys.stdout = real
            print(f"@@ERR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
