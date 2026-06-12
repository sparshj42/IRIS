import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

da3 = np.load("output_trellis/da3_scene_points.npy")
vggt = np.load("output_trellis/scene_pointcloud.npy")


def norm(p):
    p = p - np.median(p, 0)
    s = np.percentile(np.linalg.norm(p, axis=1), 95)
    return p / s


fig = plt.figure(figsize=(16, 8))
for col, (pc, name) in enumerate([(vggt, f"VGGT ({len(vggt)//1000}K)"),
                                  (da3, f"DA3 ({len(da3)//1000}K)")]):
    p = norm(pc)
    # subsample for plotting
    if len(p) > 120000:
        p = p[np.random.choice(len(p), 120000, replace=False)]
    for row, (elev, azim, t) in enumerate([(-75, -90, "front"), (-55, -35, "angled")]):
        ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.6, marker=".", c=p[:, 2], cmap="viridis")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} — {t}", fontsize=10)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
        ax.set_box_aspect((1, 1, 1)); ax.axis("off")
plt.tight_layout()
plt.savefig("compare_da3_vggt.png", dpi=85, bbox_inches="tight")
print("saved compare_da3_vggt.png")
