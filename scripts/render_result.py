"""Render the final labeled point cloud (semantic colors) from two viewpoints."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pts = np.load("output_full/labeled_pointcloud_points.npy")
lab = np.load("output_full/labeled_pointcloud_labels.npy")
COL = {0:[0.6,0.4,0.2],1:[0.8,0.8,0.8],2:[0.9,0.9,0.7],3:[0.2,0.6,0.9],4:[0.2,0.8,0.2]}
NAME = {0:"floor",1:"wall",2:"ceiling",3:"platform/table",4:"object"}
colors = np.array([COL[l] for l in lab])

# center + scale
c = pts.mean(0); p = pts - c
s = np.percentile(np.linalg.norm(p, axis=1), 99)
p = p / s

fig = plt.figure(figsize=(16, 7))
for i, (elev, azim, title) in enumerate([(-70, -90, "front"), (-60, -30, "angled")]):
    ax = fig.add_subplot(1, 2, i+1, projection="3d")
    ax.scatter(p[:,0], p[:,1], p[:,2], c=colors, s=1.5, marker=".")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"IRIS labeled cloud ({title})")
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2); ax.set_zlim(-1.2,1.2)
    ax.set_box_aspect((1,1,1)); ax.axis("off")
handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=COL[k],markersize=9,
           label=NAME[k]) for k in [1,3,4,0]]
fig.legend(handles=handles, loc="lower center", ncol=4)
plt.tight_layout(); plt.savefig("result_labeled_cloud.png", dpi=90, bbox_inches="tight")
print("saved result_labeled_cloud.png")
