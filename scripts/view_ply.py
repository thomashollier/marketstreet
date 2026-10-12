"""View a LoGeR point cloud (and optionally the camera path) with Open3D.

    pip install open3d numpy
    python scripts/view_ply.py outputs/SF_1906_marketstreet_16/LoGeR_star

Pass the folder that `modal_app.py::pull` created (points.ply [+ poses.npz]),
or a path to a .ply file. Mouse: left-drag orbit, wheel zoom, shift-drag pan.
Keys: +/- point size, h help.
"""
import argparse
import pathlib

import numpy as np
import open3d as o3d


def camera_geometry(poses, every, size):
    """Camera path polyline + a small axes triad at every Nth camera."""
    pos = poses[:, :3, 3]
    path = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pos),
        lines=o3d.utility.Vector2iVector(np.c_[np.arange(len(pos) - 1), np.arange(1, len(pos))]),
    )
    path.paint_uniform_color([1.0, 0.2, 0.2])
    geoms = [path]
    for T in poses[::every]:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        axes.transform(T)
        geoms.append(axes)
    return geoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="folder containing points.ply (+ poses.npz), or a .ply file")
    ap.add_argument("--flip", action="store_true",
                    help="rotate 180 deg about X (use if the scene is upside-down; OpenCV y-down -> y-up)")
    ap.add_argument("--no-cameras", action="store_true", help="hide the camera path")
    ap.add_argument("--camera-every", type=int, default=10, help="draw axes every Nth camera")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel size to downsample (0 = off)")
    ap.add_argument("--point-size", type=float, default=2.0)
    a = ap.parse_args()

    p = pathlib.Path(a.path)
    ply = p if p.suffix == ".ply" else p / "points.ply"
    poses_file = ply.parent / "poses.npz"

    pcd = o3d.io.read_point_cloud(str(ply))
    if pcd.is_empty():
        raise SystemExit(f"no points read from {ply}")
    print(f"{ply}: {len(pcd.points):,} points")
    if a.voxel > 0:
        pcd = pcd.voxel_down_sample(a.voxel)
        print(f"downsampled to {len(pcd.points):,} points")

    geoms = [pcd]
    if not a.no_cameras and poses_file.exists():
        poses = np.load(poses_file)["poses"]
        extent = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        print(f"{len(poses)} cameras; scene extent {extent:.2f}")
        geoms += camera_geometry(poses, max(a.camera_every, 1), size=extent * 0.02)

    if a.flip:
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        for g in geoms:
            g.rotate(R, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(ply), width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().point_size = a.point_size
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
