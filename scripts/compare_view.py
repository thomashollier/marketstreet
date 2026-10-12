"""Side-by-side Open3D viewer for two reconstructions, one shared camera.

    pip install open3d numpy
    python scripts/compare_view.py outputs/<movie>/LoGeR_star/aligned_tumbled \\
                                   outputs/<movie>/lingbot-map/aligned_tumbled

One window, two scene viewports. A single orbit / pan / zoom controller (below)
takes the mouse from BOTH halves and applies the identical camera to each, so
they always transform the same way. Open3D's own per-widget controllers are
switched off by consuming the events.

Each folder needs points.ply (+ poses.npz, recommended). Each cloud is shifted so
its first camera is at the origin and scaled so its camera path has length 1
(the models' scales differ), so differences show up as shape, not size.
The red line is the camera path.

  left-drag orbit | right-drag or ctrl+left-drag pan | wheel zoom
  R reset view | P toggle camera path | + / - point size
Starts at the first camera, looking down the camera path.
"""
import argparse
import pathlib
import sys

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
        self._init = (np.zeros(3) + _unit(forward) * dist, -_unit(forward), self.world_up.copy(), dist)
        self.reset()

    def reset(self):
        self.target, self.dir, self.up, self.dist = (np.array(x, float) if i < 3 else x
                                                    for i, x in enumerate(self._init))

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
    pane = {"name": "/".join(d.resolve().parts[-2:]), "count": len(xyz), "fwd": [0, 0, 1], "down": [0, 1, 0],
            "path": None}
    poses_file = d / "poses.npz"
    if poses_file.exists():
        P = np.load(poses_file)["poses"].astype(np.float64)
        C = P[:, :3, 3]
        origin = C[0]
        scale = 1.0 / np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
        pane.update(fwd=P[:, :3, 2].mean(0), down=P[:, :3, 1].mean(0),
                    path=(C[:: max(1, len(C) // 300)] - origin) * scale)
    else:
        print(f"warning: {poses_file} missing; scaling by bounding box, no path / start view", file=sys.stderr)
        origin = xyz.mean(0)
        scale = 1.0 / np.linalg.norm(xyz.max(0) - xyz.min(0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((xyz - origin) * scale)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    pane["pcd"] = pcd
    return pane


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--max-points", type=int, default=1_000_000, help="per cloud, random subsample")
    a = ap.parse_args()

    import open3d as o3d
    gui, rendering = o3d.visualization.gui, o3d.visualization.rendering
    panes = [load_pane(d, a.max_points) for d in (a.left, a.right)]

    # shared start view, from the left cloud's cameras (both are expected to be aligned the same way)
    cam = Orbit(up=-_unit(panes[0]["down"]), forward=_unit(panes[0]["fwd"]))

    app = gui.Application.instance
    app.initialize()
    win = app.create_window("compare", 1600, 800)
    CONSUMED, HANDLED = gui.Widget.EventCallbackResult.CONSUMED, gui.Widget.EventCallbackResult.HANDLED
    state = {"size": 2.0, "path": True, "last": (0, 0)}

    widgets, labels, mats = [], [], []
    for p in panes:
        w = gui.SceneWidget()
        w.scene = rendering.Open3DScene(win.renderer)
        w.scene.set_background([0.05, 0.05, 0.06, 1.0])
        m = rendering.MaterialRecord()
        m.shader, m.point_size = "defaultUnlit", state["size"] * win.scaling
        w.scene.add_geometry("pts", p["pcd"], m)
        if p["path"] is not None:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(p["path"])
            ls.lines = o3d.utility.Vector2iVector(np.c_[np.arange(len(p["path"]) - 1), np.arange(1, len(p["path"]))])
            lm = rendering.MaterialRecord()
            lm.shader, lm.line_width, lm.base_color = "unlitLine", 2.0 * win.scaling, [1.0, 0.2, 0.2, 1.0]
            w.scene.add_geometry("path", ls, lm)
        widgets.append(w)
        mats.append(m)
        win.add_child(w)
    for p in panes:
        lab = gui.Label(f"{p['name']} - {p['count']:,} pts")
        lab.text_color = gui.Color(1, 1, 1)
        labels.append(lab)
        win.add_child(lab)

    def apply():
        for w in widgets:
            w.look_at(cam.target.astype(np.float32), cam.eye.astype(np.float32), cam.up.astype(np.float32))
        win.post_redraw()

    def on_layout(ctx):
        r = win.content_rect
        half = r.width // 2
        for i, (w, lab) in enumerate(zip(widgets, labels)):
            x, wd = (r.x, half - 1) if i == 0 else (r.x + half + 1, r.width - half - 1)
            w.frame = gui.Rect(x, r.y, wd, r.height)
            lab.frame = gui.Rect(x + 10, r.y + 8, wd - 20, 24)
            w.scene.camera.set_projection(FOV, wd / max(r.height, 1), 0.005, 500.0,
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
        elif e.type == T.WHEEL:
            cam.zoom(e.wheel_dy)
            apply()
        return CONSUMED     # keep Open3D's own (unsynchronised) controller out of it

    def on_key(e):
        if e.type != gui.KeyEvent.Type.DOWN:
            return HANDLED
        if e.key == gui.KeyName.R:
            cam.reset()
        elif e.key == gui.KeyName.P:
            state["path"] = not state["path"]
            for w, p in zip(widgets, panes):
                if p["path"] is not None:
                    w.scene.show_geometry("path", state["path"])
        elif e.key in (gui.KeyName.EQUALS, gui.KeyName.MINUS):
            state["size"] = max(1.0, state["size"] + (0.5 if e.key == gui.KeyName.EQUALS else -0.5))
            for w, m in zip(widgets, mats):
                m.point_size = state["size"] * win.scaling
                w.scene.modify_geometry_material("pts", m)
        apply()
        return HANDLED

    for w in widgets:
        w.set_on_mouse(on_mouse)
        w.set_on_key(on_key)
    win.set_on_layout(on_layout)
    app.run()


if __name__ == "__main__":
    main()
