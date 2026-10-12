"""Side-by-side Open3D viewer for two reconstructions: two windows, two processes, one camera.

    pip install open3d numpy
    python scripts/compare_view.py outputs/<movie>/LoGeR_star/aligned_tumbled \\
                                   outputs/<movie>/lingbot-map/aligned_tumbled

Each reconstruction opens in its own window in its own process, using Open3D's
classic Visualizer (the same one scripts/view_ply.py uses, so the mouse feels the
same). The two keep their cameras identical over a pipe: each process watches its
own camera, and when it changes (you dragged, scrolled ...) sends the new camera
matrix to the other, which applies it. Only the 4x4 matrix is exchanged, so window
sizes may differ, and changes below a tiny tolerance are ignored so the windows
cannot echo each other. Closing either window closes both.

Each folder needs points.ply (+ poses.npz, recommended). Each cloud is shifted so
its first camera is at the origin and scaled so its camera path has length 1
(the models' scales differ), so differences show up as shape, not size.
The red line is the camera path (--no-path to hide it).

Open3D's usual mouse/keys apply: left-drag orbit, wheel zoom, ctrl+left-drag or
shift+left-drag pan, +/- point size (synced). R resets both to a view from the
first camera looking down the camera path.
"""
import argparse
import multiprocessing
import pathlib
import queue
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_to_path import read_ply   # noqa: E402

TOL = 1e-4          # max abs change in the 4x4 camera matrix that counts as "the camera moved"
START_DIST = 0.3    # initial look-at point, in path lengths ahead of the first camera


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def load_pane(d, max_points):
    import open3d as o3d
    d = pathlib.Path(d)
    xyz, rgb = read_ply(d / "points.ply")
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    pane = {"name": "/".join(d.resolve().parts[-2:]), "count": len(xyz), "path": None}
    poses_file = d / "poses.npz"
    if poses_file.exists():
        C = np.load(poses_file)["poses"].astype(np.float64)[:, :3, 3]
        origin = C[0]
        scale = 1.0 / np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
        pane["path"] = (C[:: max(1, len(C) // 300)] - origin) * scale
    else:
        print(f"warning: {poses_file} missing; scaling by bounding box, no camera path", file=sys.stderr)
        origin = xyz.mean(0)
        scale = 1.0 / np.linalg.norm(xyz.max(0) - xyz.min(0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz - origin) * scale)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    pane["pcd"] = pcd
    return pane


def start_view(d):
    """(up, forward) of the camera path, from the average camera pose (OpenCV axes: y down, z forward)."""
    f = pathlib.Path(d) / "poses.npz"
    if not f.exists():
        return [0, -1, 0], [0, 0, 1]
    P = np.load(f)["poses"].astype(np.float64)
    return (-_unit(P[:, :3, 1].mean(0))).tolist(), _unit(P[:, :3, 2].mean(0)).tolist()


def viewer(idx, d, conn, up, forward, rect, max_points, show_path):
    """One window in one process. `conn` is the pipe to the other viewer."""
    import faulthandler
    import threading
    faulthandler.enable()               # a native crash prints which Python line it happened in
    import open3d as o3d
    pane = load_pane(d, max_points)
    up, forward = _unit(up), _unit(forward)
    x, y, width, height = rect

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(f"{pane['name']} - {pane['count']:,} pts", width, height, x, y):
        raise RuntimeError("Open3D could not create a window")
    vis.add_geometry(pane["pcd"])
    if show_path and pane["path"] is not None:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pane["path"])
        ls.lines = o3d.utility.Vector2iVector(np.c_[np.arange(len(pane["path"]) - 1), np.arange(1, len(pane["path"]))])
        ls.paint_uniform_color([1.0, 0.2, 0.2])
        vis.add_geometry(ls, reset_bounding_box=False)
    ro, vc = vis.get_render_option(), vis.get_view_control()
    ro.background_color = np.array([0.05, 0.05, 0.06])
    ro.point_size = 2.0

    inbox = queue.SimpleQueue()
    st = {"init": False, "last": None, "size": None}

    def ext():
        return np.asarray(vc.convert_to_pinhole_camera_parameters().extrinsic, dtype=float).copy()

    def send(msg):
        try:
            conn.send(msg)
        except (OSError, ValueError):
            pass                        # the other window is already gone

    def reset_view():
        """From the first camera, looking down the path. Distance is fixed up from the camera matrix itself."""
        look = forward * START_DIST
        zoom = 0.2
        vc.set_lookat(look)
        vc.set_front(-forward)
        vc.set_up(up)
        vc.set_zoom(zoom)
        E = ext()
        eye = -E[:3, :3].T @ E[:3, 3]
        dist = np.linalg.norm(eye - look)
        if dist > 1e-9:
            vc.set_zoom(zoom * START_DIST / dist)

    def tick(_vis):                     # called every frame, on the GUI thread
        if not st["init"]:
            reset_view()
            st.update(init=True, last=None if idx == 0 else ext(), size=ro.point_size)
        cam = size = None
        while True:                     # newest message of each kind wins
            try:
                msg = inbox.get_nowait()
            except queue.Empty:
                break
            if "quit" in msg:
                vis.close()
                return False
            cam = msg.get("cam", cam)
            size = msg.get("size", size)
        if cam is not None:
            p = vc.convert_to_pinhole_camera_parameters()
            cam = np.asarray(cam, dtype=float)
            if np.abs(cam - np.asarray(p.extrinsic)).max() > TOL:
                p.extrinsic = cam
                vc.convert_from_pinhole_camera_parameters(p, True)
            st["last"] = ext()          # remember what it reads back as, so it is not echoed
        if size is not None:
            ro.point_size = st["size"] = size
        e = ext()                       # anything else that moved the camera is the user: tell the other window
        if st["last"] is None or np.abs(e - st["last"]).max() > TOL:
            send({"cam": e.tolist()})
            st["last"] = e
        if ro.point_size != st["size"]:
            st["size"] = ro.point_size
            send({"size": st["size"]})
        return False

    def listen():                       # blocks on the pipe; the GUI thread picks messages up from `inbox`
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                msg = {"quit": True}
            inbox.put(msg)
            if "quit" in msg:
                return

    vis.register_key_callback(ord("R"), lambda _v: reset_view() or False)
    vis.register_animation_callback(tick)
    threading.Thread(target=listen, daemon=True).start()
    vis.run()
    send({"quit": True})
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--max-points", type=int, default=1_000_000, help="per cloud, random subsample")
    ap.add_argument("--width", type=int, default=780, help="width of each window")
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--no-path", action="store_true", help="don't draw the camera path")
    a = ap.parse_args()

    up, forward = start_view(a.left)        # shared start view, from the left cloud's cameras
    ctx = multiprocessing.get_context("spawn")
    c0, c1 = ctx.Pipe()
    procs = [ctx.Process(target=viewer,
                         args=(i, d, c, up, forward, (20 + i * (a.width + 20), 40, a.width, a.height),
                               a.max_points, not a.no_path))
             for i, (d, c) in enumerate(((a.left, c0), (a.right, c1)))]
    for p in procs:
        p.start()
    try:
        while all(p.is_alive() for p in procs):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    for p in procs:
        if p.is_alive():
            p.terminate()
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode not in (0, None, -15)]
    if bad:
        sys.exit(f"a viewer process exited with code {bad[0]}")


if __name__ == "__main__":
    main()
