"""Turn a colored point cloud (.ply) into a colored surface mesh.

Usage:
    python scripts/mesh_pointcloud.py <in.ply> <out.ply> [--method poisson|bpa] [--per_object]

  --method poisson  (default) watertight/smooth, fills gaps   (best for solid objects)
  --method bpa      ball-pivoting: hugs the points, keeps detail, no gap-filling
  --per_object      cluster into separate objects (DBSCAN) and mesh each on its own,
                    then merge — avoids webbing nearby objects together (cleanest)
"""
import sys, argparse, numpy as np, open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--method", choices=["poisson", "bpa"], default="poisson")
ap.add_argument("--per_object", action="store_true")
a = ap.parse_args()

pcd = o3d.io.read_point_cloud(a.inp)
diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())


def mesh_one(p):
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * diag, max_nn=40))
    p.orient_normals_consistent_tangent_plane(20)
    if a.method == "poisson":
        m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p, depth=9)
        m.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), 0.05))
        m = m.crop(p.get_axis_aligned_bounding_box())     # trim Poisson balloon
    else:  # ball pivoting — radii from point spacing
        d = np.mean(p.compute_nearest_neighbor_distance())
        radii = o3d.utility.DoubleVector([d * r for r in (1.5, 2.0, 3.0, 4.0)])
        m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(p, radii)
    return m


if a.per_object:
    labels = np.array(pcd.cluster_dbscan(eps=0.03 * diag, min_points=30))
    mesh = o3d.geometry.TriangleMesh()
    for k in range(labels.max() + 1):
        sub = pcd.select_by_index(np.where(labels == k)[0])
        if len(sub.points) >= 50:
            mesh += mesh_one(sub)
    print(f"meshed {labels.max() + 1} objects separately")
else:
    mesh = mesh_one(pcd)

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, mesh)
print(f"saved {a.out}: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")
