#!/usr/bin/env python3
"""
align_tube_to_mano.py

Align a simple tube‐skeleton mesh to a flat MANO hand mesh via ICP, producing
a realistic hand surface mesh that follows the pose of the tube.

Usage:
    python align_tube_to_mano.py \
        --tube-obj   hand_tube.obj \
        --mano-dir   ./mano \
        --out-obj    hand_aligned.obj \
        [--n-points 20000] \
        [--max-corr 0.05]

Dependencies:
    pip install open3d torch smplx trimesh numpy
"""

import argparse
import numpy as np
import open3d as o3d
import torch
from smplx import MANO
import trimesh

def load_tube_pcd(tube_path: str, n_points: int) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(tube_path)
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    return pcd

def build_flat_mano_pcd(mano_dir: str, n_points: int, device: torch.device):
    # Instantiate MANO with flat mean hand
    mano = MANO(
        model_path=mano_dir,
        use_pca=False,
        is_rhand=True,
        flat_hand_mean=True
    ).to(device)
    # Forward with default (zeros) params to get mean shape
    out = mano()
    verts = out.vertices.detach().cpu().numpy().squeeze()  # (778,3)
    faces = mano.faces                                     # (1538,3)

    # Create trimesh and convert to Open3D
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tri.vertices),
        o3d.utility.Vector3iVector(tri.faces)
    )
    o3d_mesh.compute_vertex_normals()

    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=n_points)
    return o3d_mesh, pcd

def align_meshes(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    max_corr_dist: float
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=max_corr_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-obj",  required=True, help="Input tube skeleton OBJ")
    parser.add_argument("--mano-dir",  required=True, help="Path to downloaded MANO folder")
    parser.add_argument("--out-obj",   required=True, help="Output aligned hand OBJ")
    parser.add_argument("--n-points",  type=int, default=20000,
                        help="Number of points to sample on each mesh")
    parser.add_argument("--max-corr",  type=float, default=0.05,
                        help="Max correspondence distance for ICP")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1) Sampling tube point cloud...")
    tube_pcd = load_tube_pcd(args.tube_obj, args.n_points)
    print(f"   Tube has {len(tube_pcd.points)} points")

    print("2) Sampling flat MANO mean hand point cloud...")
    mano_mesh, mano_pcd = build_flat_mano_pcd(args.mano_dir, args.n_points, device)
    print(f"   MANO mean has {len(mano_pcd.points)} points")

    print("3) Running ICP alignment...")
    reg = align_meshes(source_pcd=mano_pcd,
                       target_pcd=tube_pcd,
                       max_corr_dist=args.max_corr)
    print(f"   ICP fitness: {reg.fitness:.4f}, RMSE: {reg.inlier_rmse:.4f}")

    print("4) Transforming MANO mesh...")
    mano_mesh.transform(reg.transformation)

    print(f"5) Saving aligned mesh to {args.out_obj}...")
    o3d.io.write_triangle_mesh(args.out_obj, mano_mesh)
    print("Done.")

if __name__ == "__main__":
    main()
