"""Cross-sectional slices of the occupancy grid — clearest check of free/occupied/
occluded geometry (free should be a cone in front of surfaces; occluded behind)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

occ = np.load("output_trellis/occupancy_grid.npy")        # 0 occluded,1 free,2 occupied
dims = occ.shape
cmap = ListedColormap([(0.5, 0.5, 0.9), (0.6, 0.95, 0.6), (0.9, 0.2, 0.2)])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# middle slice along each axis, plus a couple of offset slices to show the cone
for ax, axis, name in zip(axes, [0, 1, 2], ["X", "Y", "Z"]):
    # pick the slice with the most OCCUPIED voxels (cuts through the scene surfaces)
    occ_counts = [(np.take(occ, i, axis=axis) == 2).sum() for i in range(dims[axis])]
    i = int(np.argmax(occ_counts))
    sl = np.take(occ, i, axis=axis)
    ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(f"slice ⟂ {name} @ {i}")
    ax.axis("off")
handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=12, label=l)
           for c, l in [((0.6, 0.95, 0.6), "free"), ((0.9, 0.2, 0.2), "occupied"),
                        ((0.5, 0.5, 0.9), "occluded")]]
fig.legend(handles=handles, loc="lower center", ncol=3)
plt.tight_layout()
plt.savefig("occ_slices.png", dpi=95, bbox_inches="tight")
print("saved occ_slices.png")
