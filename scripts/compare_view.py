"""Side-by-side Open3D viewer for two reconstructions: two windows, two processes, one camera.

    pip install open3d numpy
    python scripts/compare_view.py outputs/<movie>/LoGeR_star/aligned_tumbled \\
                                   outputs/<movie>/lingbot-map/aligned_tumbled

Each reconstruction opens in its own window, in its own process (own GL context,
own event loop), and the two keep their cameras identical over a pipe: every
camera change is sent as the complete camera state, so there is no leader and
nothing to drift. Open3D's own mouse controller is switched off (events are
consumed) and replaced by the small `Orbit` controller below, so both processes
move exactly the same way. Closing either window closes both.

Each folder needs points.ply (+ poses.npz, recommended). Each cloud is shifted so
its first camera is at the origin and scaled so its camera path has length 1
(the models' scales differ), so differences show up as shape, not size.
The red line is the camera path.

  left-drag orbit | right-drag or ctrl+left-drag pan | wheel zoom
  R reset view | P toggle camera path | + / - point size   (all synced)
Starts at the first camera, looking down the camera path.
"""
import argparse
import multiprocessing
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_to_path import read_ply   # noqa: E402

FOV = 60.0


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot(v, axis, ang):
    """Rodrigues rotation of v about a unit axis."""
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


class Orbit:
    """Turntable-style camera: yaw about the scene's up axis, pitch about the camera's right axis."""
    ROTATE = 0.005      # radians per pixel
    ZOOM = 0.9          # distance factor per wheel notch

    def __init__(self, up, forward, dist=0.3):
        self.world_up = _unit(up)
        self._init = (_unit(forward) * dist, -_unit(forward), self.world_up.copy(), dist)
        self.reset()

    def reset(self):
        self.target, self.dir, self.up = (np.array(x, float) for x in self._init[:3])
        self.dist = self._init[3]

    def state(self):
        """Complete camera state as plain lists (picklable, sent over the pipe)."""
        return {"target": self.target.tolist(), "dir": self.dir.tolist(), "up": self.up.tolist(),
                "dist": float(self.dist)}

    def set_state(self, s):
        self.target, self.dir, self.up = (np.array(s[k], float) for k in ("target", "dir", "up"))
        self.dist = float(s["dist"])

    @property
    def eye(self):
        return self.target + self.dist * self.dir

    def _basis(self):
        f = -self.dir
        r = _unit(np.cross(f, self.up))
        return f, r, np.cross(r, f)

    def rotate(self, dx, dy):
        a = -dx * self.ROTATE
        self.dir, self.up = _rot(self.dir, self.world_up, a), _rot(self.up, self.world_up, a)
        _, r, _ = self._basis()
        b = -dy * self.ROTATE
        self.dir, self.up = _rot(self.dir, r, b), _rot(self.up, r, b)
        self.dir, self.up = _unit(self.dir), _unit(self.up - (self.up @ self.dir) * self.dir)

    def pan(self, dx, dy, height_px):
        per_px = 2 * self.dist * np.tan(np.radians(FOV) / 2) / height_px
        _, r, u = self._basis()
        self.target = self.target - r * dx * per_px + u * dy * per_px

    def zoom(self, notches):
        self.dist *= self.ZOOM ** notches


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
    """(up, forward) of the first camera path, from the average camera pose."""
    f = pathlib.Path(d) / "poses.npz"
    if not f.exists():
        return [0, -1, 0], [0, 0, 1]
    P = np.load(f)["poses"].astype(np.float64)
    return (-_unit(P[:, :3, 1].mean(0))).tolist(), _unit(P[:, :3, 2].mean(0)).tolist()


def viewer(idx, d, conn, up, forward, rect, max_points, no_downsample):
    """One window in one process. `conn` is the pipe to the other viewer."""
    import threading
    import open3d as o3d
    gui, rendering = o3d.visualization.gui, o3d.visualization.rendering
    pane = load_pane(d, max_points)
    cam = Orbit(up, forward)
    app = gui.Application.instance
    app.initialize()
    x, y, width, height = rect
    win = app.create_window(pane["name"], width, height, x, y)
    CONSUMED, HANDLED = gui.Widget.EventCallbackResult.CONSUMED, gui.Widget.EventCallbackResult.HANDLED
    state = {"size": 2.0, "path": True, "last": (0, 0)}

    w = gui.SceneWidget()
    w.scene = rendering.Open3DScene(win.renderer)
    w.enable_scene_caching(False)       # we move the camera ourselves; cached frames would look stale
    w.scene.set_background([0.05, 0.05, 0.06, 1.0])
    m = rendering.MaterialRecord()
    m.shader, m.point_size = "defaultUnlit", state["size"] * win.scaling
    w.scene.add_geometry("pts", pane["pcd"], m, not no_downsample)
    if pane["path"] is not None:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pane["path"])
        ls.lines = o3d.utility.Vector2iVector(np.c_[np.arange(len(pane["path"]) - 1), np.arange(1, len(pane["path"]))])
        lm = rendering.MaterialRecord()
        lm.shader, lm.line_width, lm.base_color = "unlitLine", 2.0 * win.scaling, [1.0, 0.2, 0.2, 1.0]
        w.scene.add_geometry("path", ls, lm)
    win.add_child(w)
    lab = gui.Label(f"{pane['name']} - {pane['count']:,} pts")
    lab.text_color = gui.Color(1, 1, 1)
    win.add_child(lab)

    def send(msg):
        try:
            conn.send(msg)
        except (OSError, ValueError):
            pass            # the other window is already gone

    def apply():
        w.look_at(cam.target.astype(np.float32), cam.eye.astype(np.float32), cam.up.astype(np.float32))
        win.post_redraw()

    def set_size(v):
        state["size"] = v
        m.point_size = v * win.scaling
        w.scene.modify_geometry_material("pts", m)

    def set_path(v):
        state["path"] = v
        if pane["path"] is not None:
            w.scene.show_geometry("path", v)

    def on_layout(ctx):
        r = win.content_rect
        w.frame = gui.Rect(r.x, r.y, r.width, r.height)
        lab.frame = gui.Rect(r.x + 10, r.y + 8, r.width - 20, 24)
        w.scene.camera.set_projection(FOV, r.width / max(r.height, 1), 0.005, 500.0,
                                      rendering.Camera.FovType.Vertical)
        apply()

    def on_mouse(e):
        T = gui.MouseEvent.Type
        if e.type == T.BUTTON_DOWN:
            state["last"] = (e.x, e.y)
        elif e.type == T.DRAG:
            dx, dy = e.x - state["last"][0], e.y - state["last"][1]
            state["last"] = (e.x, e.y)
            if e.is_button_down(gui.MouseButton.RIGHT) or e.is_modifier_down(gui.KeyModifier.CTRL):
                cam.pan(dx, dy, win.content_rect.height)
            else:
                cam.rotate(dx, dy)
            apply()
            send({"cam": cam.state()})
        elif e.type == T.WHEEL:
            cam.zoom(e.wheel_dy)
            apply()
            send({"cam": cam.state()})
        return CONSUMED     # keep Open3D's own (unsynchronised) controller out of it

    def on_key(e):
        if e.type != gui.KeyEvent.Type.DOWN:
            return HANDLED
        if e.key == gui.KeyName.R:
            cam.reset()
            apply()
            send({"cam": cam.state()})
        elif e.key == gui.KeyName.P:
            set_path(not state["path"])
            send({"path": state["path"]})
        elif e.key in (gui.KeyName.EQUALS, gui.KeyName.MINUS):
            set_size(max(1.0, state["size"] + (0.5 if e.key == gui.KeyName.EQUALS else -0.5)))
            send({"size": state["size"]})
        return HANDLED

    def handle(msg):                        # runs on the GUI thread
        if "quit" in msg:
            app.quit()
            return
        if "cam" in msg:
            cam.set_state(msg["cam"])
            apply()
        if "size" in msg:
            set_size(msg["size"])
        if "path" in msg:
            set_path(msg["path"])

    def listen():                           # blocks on the pipe, hands each message to the GUI thread
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                msg = {"quit": True}
            app.post_to_main_thread(win, lambda m=msg: handle(m))
            if "quit" in msg:
                return

    def on_close():
        send({"quit": True})
        return True

    w.set_on_mouse(on_mouse)
    w.set_on_key(on_key)
    win.set_on_layout(on_layout)
    win.set_on_close(on_close)
    threading.Thread(target=listen, daemon=True).start()
    app.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--max-points", type=int, default=1_000_000, help="per cloud, random subsample")
    ap.add_argument("--width", type=int, default=780, help="width of each window")
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--no-downsample", action="store_true",
                    help="always render every point (Open3D otherwise draws a thinned copy while you interact)")
    a = ap.parse_args()

    up, forward = start_view(a.left)        # shared start view, from the left cloud's cameras
    ctx = multiprocessing.get_context("spawn")
    c0, c1 = ctx.Pipe()
    procs = [ctx.Process(target=viewer,
                         args=(i, d, c, up, forward, (20 + i * (a.width + 20), 40, a.width, a.height),
                               a.max_points, a.no_downsample))
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
