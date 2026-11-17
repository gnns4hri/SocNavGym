"""Renders a recorded trajectory from above with the 3D view and saves it as an MP4 video.

    python -m socnavgym.vision.trajectory_video trajectory.json                       -> trajectory.mp4, next to the JSON
    python -m socnavgym.vision.trajectory_video trajectory.json out.mp4 --elevation 55 --width 1280 --height 720

The trajectory is a JSON file in the format of the SocNav datasets, which is also what SocNavGym's sn26 rewards record:

    {"sequence": [{"timestamp": s, "robot": {"x", "y", "angle"}, "people": [{"id", "x", "y", "angle"}, ...],
                   "objects": [{"id", "type": table | chair | plant | laptop, "x", "y", "angle", "shape": {"type", "width", "length"}}, ...]}, ...],
     "walls": [[x1, y1, x2, y2], ...]}

The simulator is not run: the scene is built from the file alone (so the JSON can come from anywhere) and every instant of the
sequence becomes one picture. The view is the 3D view's default - from straight above, +X to the right and +Y up - framed ONCE
on everything the trajectory ever touches, so the picture does not zoom during the video. The video plays in real time (the
frame rate is taken from the timestamps) unless --fps says otherwise.
Needs the renderer (README.md in this folder). The video is H.264 when `ffmpeg` is installed, otherwise OpenCV's MPEG-4.
"""
import os, sys, json, math, time, shutil, argparse, subprocess
from types import SimpleNamespace as NS
import numpy as np

KINDS = ("table", "chair", "plant", "laptop")
AWAY = 1.0e4                                   # metres: where a person who is not in an instant of the sequence is put (out of every picture)


class Scene:
    """What the vision system reads of an environment, taken from a trajectory file instead."""
    def __init__(self, data, wall_thickness=0.2):
        self.seq = seq = data["sequence"]; assert len(seq) > 0, "the trajectory's sequence is empty"
        self.walls = []; self.skipped = {}; objs = {}; xs = []; ys = []
        for i, (x1, y1, x2, y2) in enumerate(w[:4] for w in data.get("walls") or []):
            L = math.hypot(x2 - x1, y2 - y1)
            if L > 1e-6: self.walls.append(NS(id=100000 + i, x=(x1 + x2) / 2, y=(y1 + y2) / 2, orientation=math.atan2(y2 - y1, x2 - x1), length=L, thickness=wall_thickness))
        for fr in seq:                                           # furniture: every object that appears, where it first appears
            for j, o in enumerate(fr.get("objects") or []):
                k = o.get("id", j)
                if k in objs: continue
                kind = str(o.get("type", "")).lower(); sh = o.get("shape") or {}; w = float(sh.get("width", 0.5)); l = float(sh.get("length", w))
                if kind not in KINDS: self.skipped[kind] = self.skipped.get(kind, 0) + 1; objs[k] = None; continue
                objs[k] = NS(kind=kind, id=int(k) if isinstance(k, int) else 200000 + len(objs), x=float(o["x"]), y=float(o["y"]), orientation=float(o.get("angle", 0.0)), length=l, width=w, radius=max(w, l) / 2)
        objs = [o for o in objs.values() if o is not None]
        self.tables = [o for o in objs if o.kind == "table"]; self.chairs = [o for o in objs if o.kind == "chair"]; self.laptops = [o for o in objs if o.kind == "laptop"]
        self.plants = [NS(kind="plant", id=o.id, x=o.x, y=o.y, orientation=o.orientation, radius=o.radius) for o in objs if o.kind == "plant"]
        ids = sorted({p["id"] for fr in seq for p in fr.get("people") or []}); self.humans = {k: NS(id=int(k), x=AWAY + 2.0 * i, y=AWAY, orientation=0.0, width=0.5, here=False) for i, k in enumerate(ids)}
        self.static_humans = list(self.humans.values()); self.dynamic_humans = []; self.static_interactions = []; self.moving_interactions = []; self.h_l_interactions = []
        self.robot = NS(id=0, x=0.0, y=0.0, orientation=0.0); self.ticks = 0.0; self.TIMESTEP = 1.0
        for fr in seq:                                           # the box around everything, at every instant: the view is framed on it once
            for q in [fr["robot"]] + list(fr.get("people") or []): xs += [q["x"] - 0.6, q["x"] + 0.6]; ys += [q["y"] - 0.6, q["y"] + 0.6]
        for o in objs: xs += [o.x - o.radius, o.x + o.radius]; ys += [o.y - o.radius, o.y + o.radius]
        self.box = [min(xs), max(xs), min(ys), max(ys)]; t = [float(fr.get("timestamp", 0.1 * i)) for i, fr in enumerate(seq)]; self.times = t
        self.dt = float(np.median(np.diff(t))) if len(t) > 1 and np.median(np.diff(t)) > 0 else 0.1
        self.set(0)

    def set(self, i):
        """Puts the robot and the people where they are at instant i of the sequence."""
        fr = self.seq[i]; r = fr["robot"]; self.robot.x, self.robot.y, self.robot.orientation = float(r["x"]), float(r["y"]), float(r.get("angle", 0.0)); self.ticks = self.times[i]
        for j, h in enumerate(self.static_humans): h.x, h.y, h.here = AWAY + 2.0 * j, AWAY, False
        for p in fr.get("people") or []:
            h = self.humans[p["id"]]; h.x, h.y, h.orientation, h.here = float(p["x"]), float(p["y"]), float(p.get("angle", 0.0)), True


def vision_for(scene, cfg):
    from .system import VisionSystem

    class TrajectoryVision(VisionSystem):
        def map_box(self): return list(self.env.box)             # the floor and the framing cover the whole trajectory from the start
        def moving(self): e = self.env; return np.array([[e.robot.x, e.robot.y]] + [[h.x, h.y] for h in e.static_humans if h.here], float)
    return TrajectoryVision(scene, cfg)


class Video:
    def __init__(self, path, width, height, fps):
        self.proc = self.cv = None; ff = shutil.which("ffmpeg")
        if ff and "libx264" in subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout:
            self.proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:.6g}", "-i", "-",
                                          "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            import cv2
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)); assert self.cv.isOpened(), f"cannot write {path}"
        self.codec = "H.264" if self.proc else "MPEG-4 (OpenCV; install ffmpeg for H.264)"

    def add(self, rgb):
        if self.proc: self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        else: self.cv.write(np.ascontiguousarray(rgb[..., ::-1]))

    def close(self):
        if self.proc:
            self.proc.stdin.close(); code = self.proc.wait(); assert code == 0, f"ffmpeg ended with code {code}"
        else: self.cv.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", help="the JSON file"); ap.add_argument("video", nargs="?", default=None, help="the MP4 file to write (default: the JSON's name with .mp4)")
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=None, help="frames per second of the video (default: real time, from the timestamps)")
    ap.add_argument("--samples", type=int, default=12, help="light samples per pixel: more = cleaner and slower")
    ap.add_argument("--elevation", type=float, default=90.0, help="degrees above the floor the scene is seen from (90: straight above)")
    ap.add_argument("--azimuth", type=float, default=-90.0, help="degrees: the side the camera is on (-90: +X to the right, +Y up)")
    ap.add_argument("--wall-thickness", type=float, default=0.2, help="metres (the file's walls are lines)"); ap.add_argument("--wall-height", type=float, default=2.5)
    ap.add_argument("--robot", default="round_pan_tilt_robot", help="the robot's URDF: a name in socnavgym/vision/robots or a path")
    a = ap.parse_args(); out = a.video or os.path.splitext(a.trajectory)[0] + ".mp4"; W, H = a.width - a.width % 2, a.height - a.height % 2      # H.264 wants even sizes
    with open(a.trajectory) as f: scene = Scene(json.load(f), a.wall_thickness)
    cfg = dict(robot=dict(urdf=a.robot), viewer=dict(enabled=True, show=False, samples=a.samples, elevation=a.elevation, azimuth=a.azimuth),
               renderer=dict(required=True), scene=dict(wall_height=a.wall_height))
    scene.RESOLUTION_X, scene.RESOLUTION_Y = W, H                    # the picture's size: what `rendering: resolution_x, resolution_y` is in an environment
    try: vis = vision_for(scene, cfg)
    except RuntimeError as e: sys.exit(str(e))
    n = len(scene.seq); fps = a.fps or 1.0 / scene.dt
    print(f"{a.trajectory}: {n} instants over {scene.times[-1] - scene.times[0]:.1f} s; {len(scene.humans)} people, {len(scene.tables)} tables, {len(scene.chairs)} chairs, "
          f"{len(scene.plants)} plants, {len(scene.laptops)} laptops, {len(scene.walls)} walls" + (f"; left out (no 3D model): {scene.skipped}" if scene.skipped else ""))
    vis.reset({}, seed=0); device = vis.client.info["device"]; video = Video(out, W, H, fps); t0 = time.time()
    try:
        for i in range(n):
            scene.set(i); vis.observe({}, advance=False); video.add(vis.view())
            if i == 0 or (i + 1) % 20 == 0 or i == n - 1: el = time.time() - t0; print(f"  {i + 1} / {n}   {el:.0f} s, about {el / (i + 1) * (n - i - 1):.0f} s left", flush=True)
    finally: video.close(); vis.close()
    print(f"{out}: {W} x {H}, {fps:.3g} frames per second, {n / fps:.1f} s, {video.codec}, rendered on {device}")


if __name__ == "__main__": main()
