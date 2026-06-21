"""Prepare IRIS GT for one ScanNet scene from its .sens, Python-3, memory-light.

Scans frame byte-offsets without decoding (like extract_sens.py), auto-selects a
content-rich frame (good depth coverage, object-range median depth), then writes:
    <frames_dir>/frame_<idx>.jpg            (color, for IRIS --image)
    <gt_dir>/depth.npy   (480x640 float32 metres)
    <gt_dir>/intr_depth.npy (4x4)
    <gt_dir>/c2w.npy        (4x4 frame pose)

Usage: python prep_gt.py <scene.sens> <frames_dir> <gt_dir>
"""
import os, sys, struct, zlib, numpy as np, cv2

sens, frames_dir, gt_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(frames_dir, exist_ok=True); os.makedirs(gt_dir, exist_ok=True)

COMP_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMP_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def read_header(f):
    assert struct.unpack("I", f.read(4))[0] == 4
    strlen = struct.unpack("Q", f.read(8))[0]; f.read(strlen)
    f.read(64)  # intr_color
    f.read(64)  # extr_color
    intr_depth = np.frombuffer(f.read(64), np.float32).reshape(4, 4).copy()
    f.read(64)  # extr_depth
    color_comp = COMP_COLOR[struct.unpack("i", f.read(4))[0]]
    depth_comp = COMP_DEPTH[struct.unpack("i", f.read(4))[0]]
    cw, ch = struct.unpack("I", f.read(4))[0], struct.unpack("I", f.read(4))[0]
    dw, dh = struct.unpack("I", f.read(4))[0], struct.unpack("I", f.read(4))[0]
    depth_shift = struct.unpack("f", f.read(4))[0]
    nframes = struct.unpack("Q", f.read(8))[0]
    return dict(intr_depth=intr_depth, color_comp=color_comp, depth_comp=depth_comp,
                dw=dw, dh=dh, depth_shift=depth_shift, nframes=nframes)


def scan_offsets(f, n):
    offs = []
    for _ in range(n):
        pose = np.frombuffer(f.read(64), np.float32).reshape(4, 4).copy()
        f.read(16)  # 2 timestamps
        cb = struct.unpack("Q", f.read(8))[0]; db = struct.unpack("Q", f.read(8))[0]
        coff = f.tell(); f.seek(cb + db, os.SEEK_CUR)
        offs.append((coff, cb, db, pose))
    return offs


def decode_depth(f, coff, cb, db, h):
    f.seek(coff + cb); data = zlib.decompress(f.read(db))
    return np.frombuffer(data, np.uint16).reshape(h["dh"], h["dw"]).astype(np.float32) / h["depth_shift"]


def decode_color(f, coff, cb, comp):
    f.seek(coff); d = f.read(cb)
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)


with open(sens, "rb") as f:
    h = read_header(f); offs = scan_offsets(f, h["nframes"])
    print(f"frames={h['nframes']} depth={h['dw']}x{h['dh']} shift={h['depth_shift']}")

    # auto-select: among evenly-spaced candidates, prefer good coverage and a
    # median depth in the object range (~1.5-4 m), avoiding wall/hallway frames.
    cand = np.linspace(0.15, 0.85, 15) * (h["nframes"] - 1)
    best, best_score = None, -1
    for ci in cand.astype(int):
        coff, cb, db, pose = offs[ci]
        if not np.isfinite(pose).all():
            continue
        d = decode_depth(f, coff, cb, db, h)
        valid = d > 0; frac = valid.mean()
        if frac < 0.5:
            continue
        med = np.median(d[valid])
        score = frac - abs(med - 2.5) * 0.15   # coverage, penalise far/near median
        if score > best_score:
            best, best_score = ci, score
    if best is None:
        best = int(0.5 * (h["nframes"] - 1))
    print(f"selected frame {best} (score {best_score:.3f})")

    coff, cb, db, pose = offs[best]
    depth = decode_depth(f, coff, cb, db, h)
    color = decode_color(f, coff, cb, h["color_comp"])

np.save(f"{gt_dir}/depth.npy", depth.astype(np.float32))
np.save(f"{gt_dir}/intr_depth.npy", h["intr_depth"].astype(np.float32))
np.save(f"{gt_dir}/c2w.npy", pose.astype(np.float32))
cv2.imwrite(f"{frames_dir}/frame_{best:06d}.jpg", color)
print(f"wrote {gt_dir}/{{depth,intr_depth,c2w}}.npy and {frames_dir}/frame_{best:06d}.jpg "
      f"({color.shape[1]}x{color.shape[0]})")
