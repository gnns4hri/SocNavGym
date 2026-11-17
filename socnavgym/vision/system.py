"""SocNavGym's vision system: sensors (RGB-D cameras, lidars) mounted on the robot through a URDF, their readings in the
observation, the robot's joints in the action, and a 3D view of the scene. The 2D simulation is not touched: this module
only LOOKS at it. It exists only when the environment's configuration has a `vision:` section that declares a sensor or the
3D view (see README.md in this folder); otherwise the environment never imports it.

    vision:
      robot:    {urdf: round_pan_tilt_robot, joints: {head_tilt: 0.2}}
      sensors:
        camera: {type: rgbd,  link: camera_link}                 # RealSense D435-like unless said otherwise
        scan:   {type: lidar, link: lidar_link, mode: simple}    # simple: against the simulator's 2D shapes; mesh: the 3D models
      viewer:   {enabled: true}

Without the renderer installed (Blender's `bpy`), what needs it - cameras, `mesh` lidars, the 3D view - is left out with a
warning and the simulation goes on (`renderer: {required: true}`: an error instead).

Observation (added keys):  obs["sensors"][name] = {"rgb", "depth", ...} or {"ranges", ...};  obs["joints"] = positions.
Action: the simulator's 3 numbers, followed by one number in [-1, 1] per movable joint (its position target between the
joint's limits; the joint moves towards it at most at the URDF's velocity limit).
"""
import os, math, copy, warnings
import numpy as np
from gymnasium import spaces
from .ipc import CLASSES
from .urdf import URDF

HERE = os.path.dirname(os.path.abspath(__file__))

# defaults of a sensor of each type; `model` picks a set, any field can be overridden in the configuration
MODELS = {
    "rgbd": {
        # Intel RealSense D435, colour stream with the depth aligned to it
        "realsense_d435": dict(width=640, height=480, hfov=69.0, depth_min=0.3, depth_max=10.0, depth_noise=True, baseline=0.05, subpixel=0.08),
        # Intel RealSense D455: longer baseline (less depth noise), wider colour image
        "realsense_d455": dict(width=640, height=480, hfov=87.0, depth_min=0.6, depth_max=20.0, depth_noise=True, baseline=0.095, subpixel=0.08),
        "ideal": dict(width=640, height=480, hfov=69.0, depth_min=0.0, depth_max=100.0, depth_noise=False, baseline=0.05, subpixel=0.0),
    },
    "lidar": {
        # a common 2D scanner (RPLIDAR A2 / S1 class): one ring, a reading per degree
        "generic_2d": dict(beams=360, fov=360.0, rings=1, vfov=0.0, range_min=0.15, range_max=12.0, noise=0.01),
        # 16 rings over +-15 degrees (Velodyne VLP-16-like); needs mode: mesh
        "generic_3d_16": dict(beams=360, fov=360.0, rings=16, vfov=30.0, range_min=0.4, range_max=30.0, noise=0.02),
        "ideal": dict(beams=360, fov=360.0, rings=1, vfov=0.0, range_min=0.0, range_max=100.0, noise=0.0),
    },
}
DEFAULT_MODEL = {"rgbd": "realsense_d435", "lidar": "generic_2d"}
VIEW = "3D view"                               # the 3D view's name among the windows
SCENE = dict(wall_height=2.5, table_height=0.75, chair_height=0.0, laptop_height=0.0, human_height=0.0, human_height_spread=0.04,
             humans=["human_male_01", "human_female_01"], lighting="auto", floor_margin=3.0)      # lighting auto: indoor (ceiling with lamps) when the scene has walls, outdoor (open sky) when it has none


def base_pose(x, y, yaw, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw); M = np.eye(4); M[:2, :2] = [[c, -s], [s, c]]; M[:3, 3] = (x, y, z); return M


class Sensor:
    def __init__(self, name, cfg, urdf, ground_truth):
        cfg = dict(cfg); self.name = name; self.type = str(cfg.pop("type", "rgbd")).lower()
        assert self.type in MODELS, f"vision sensor '{name}': type must be one of {list(MODELS)}, not '{self.type}'"
        model = cfg.pop("model", DEFAULT_MODEL[self.type])
        assert model in MODELS[self.type], f"vision sensor '{name}': unknown {self.type} model '{model}' (known: {list(MODELS[self.type])})"
        self.link = cfg.pop("link", None)
        assert self.link in urdf.links, f"vision sensor '{name}': link '{self.link}' is not in the robot's URDF (links: {list(urdf.links)})"
        self.ground_truth = bool(cfg.pop("ground_truth", ground_truth)); self.model = model
        p = dict(MODELS[self.type][model]); extra = {"mode": "simple", "no_return": "inf", "ignore_robot": True} if self.type == "lidar" else {"samples": None, "exposure": 1.0}
        p.update(extra); unknown = [k for k in cfg if k not in p]
        assert not unknown, f"vision sensor '{name}': unknown field(s) {unknown} (known: {sorted(p)})"
        p.update(cfg); self.p = p
        if self.type == "lidar":
            n = int(p["beams"]); f = math.radians(float(p["fov"])); r = int(p["rings"])
            self.azimuths = -f / 2 + f * np.arange(n) / n if abs(f - 2 * math.pi) < 1e-6 else np.linspace(-f / 2, f / 2, n)      # full turn: no doubled beam
            self.elevations = np.linspace(-math.radians(p["vfov"]) / 2, math.radians(p["vfov"]) / 2, r) if r > 1 else np.zeros(1)
            assert p["mode"] in ("simple", "mesh"), f"vision sensor '{name}': mode must be simple or mesh"
            assert r == 1 or p["mode"] == "mesh", f"vision sensor '{name}': a lidar with several rings needs mode: mesh"
            assert p["no_return"] in ("inf", "max", "zero")
            self.shape = (n,) if r == 1 else (r, n); self.needs_renderer = p["mode"] == "mesh"
        else:
            self.shape = (int(p["height"]), int(p["width"])); self.needs_renderer = True
            self.focal = (p["width"] / 2) / math.tan(math.radians(p["hfov"]) / 2)

    def space(self):
        d = {}
        if self.type == "rgbd":
            d["rgb"] = spaces.Box(0, 255, self.shape + (3,), np.uint8); d["depth"] = spaces.Box(0.0, float(self.p["depth_max"]), self.shape, np.float32)
        else:
            d["ranges"] = spaces.Box(0.0, np.inf if self.p["no_return"] == "inf" else float(self.p["range_max"]), self.shape, np.float32)
        if self.ground_truth:
            d["segmentation"] = spaces.Box(0, 255, self.shape, np.uint8); d["instance"] = spaces.Box(-1, np.iinfo(np.int32).max, self.shape, np.int32)
            d["pose"] = spaces.Box(-np.inf, np.inf, (4, 4), np.float32)
        return spaces.Dict(d)

    def info(self):
        """What a user of the readings needs to know about the sensor (constant during the run)."""
        if self.type == "rgbd":
            W, H = self.p["width"], self.p["height"]; f = self.focal
            return dict(type="rgbd", model=self.model, link=self.link, width=W, height=H, hfov=math.radians(self.p["hfov"]),
                        K=np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]), depth_min=self.p["depth_min"], depth_max=self.p["depth_max"],
                        frame="pose: X forward, Y left, Z up; image: u right, v down; depth along the optical axis in metres, 0 = no reading")
        return dict(type="lidar", model=self.model, link=self.link, azimuths=self.azimuths.copy(), elevations=self.elevations.copy(),
                    range_min=self.p["range_min"], range_max=self.p["range_max"], mode=self.p["mode"], no_return=self.p["no_return"])


class VisionSystem:
    @staticmethod
    def wanted(cfg):
        """Does this `vision:` section ask for anything? (no sensors and no 3D view: the environment stays as it was)"""
        return bool(cfg) and (bool(cfg.get("sensors")) or bool((cfg.get("viewer") or {}).get("enabled")))

    def __init__(self, env, cfg, config_path=None):
        self.env = env; self.cfg = cfg = copy.deepcopy(cfg); rc = cfg.get("robot") or {}
        known = {"robot", "sensors", "viewer", "renderer", "scene", "ground_truth", "display"}; unknown = [k for k in cfg if k not in known]
        assert not unknown, f"vision: unknown field(s) {unknown} (known: {sorted(known)})"
        # the robot's 3D model: the one named, else the one of the simulator's body (robot: robot_shape): round or box
        f = str(rc.get("urdf") or ("box_pan_tilt_robot" if getattr(env, "ROBOT_SHAPE", "circle") == "rectangle" else "round_pan_tilt_robot")); cands = [f, os.path.join(os.path.dirname(os.path.abspath(config_path)) if config_path else "", f), os.path.join(HERE, "robots", f), os.path.join(HERE, "robots", f + ".urdf")]
        path = next((c for c in cands if os.path.isfile(os.path.expanduser(c))), None)
        assert path is not None, f"vision: robot URDF '{f}' not found (tried {cands})"
        self.urdf = URDF(path, packages=rc.get("packages") or []); self.joints = self.urdf.movable; self.nj = len(self.joints)
        init = rc.get("joints") or {}; names = [j.name for j in self.joints]
        assert all(k in names for k in init), f"vision: robot.joints names {list(init)} must be movable joints of the URDF ({names})"
        self.lower = np.array([j.lower for j in self.joints], np.float32); self.upper = np.array([j.upper for j in self.joints], np.float32)
        self.vmax = np.array([j.velocity for j in self.joints], np.float32); self.q0 = np.clip(np.array([float(init.get(n, 0.0)) for n in names], np.float32), self.lower, self.upper)
        self.q = self.q0.copy(); self.target = self.q0.copy()
        gt = bool(cfg.get("ground_truth", True)); self.sensors = {n: Sensor(n, c, self.urdf, gt) for n, c in (cfg.get("sensors") or {}).items()}
        self.scene_cfg = dict(SCENE); unknown = [k for k in (cfg.get("scene") or {}) if k not in SCENE and k not in ("floor_color", "wall_color", "ceiling_color", "lamp_power", "lamp_spacing", "lamp_color", "sky_strength", "shadows")]
        assert not unknown, f"vision: unknown scene field(s) {unknown}"
        self.scene_cfg.update(cfg.get("scene") or {})
        gone = [k for k in ("width", "height") if k in (cfg.get("viewer") or {})]
        assert not gone, f"vision: remove viewer.{' and viewer.'.join(gone)}: the 3D view has the size given by `rendering: resolution_x, resolution_y`"
        vc = dict(enabled=False, samples=12, fov=60.0, show=True); vc.update(cfg.get("viewer") or {}); self.viewer_cfg = vc
        vc["width"], vc["height"] = int(getattr(env, "CANVAS_X", env.RESOLUTION_X)), int(getattr(env, "CANVAS_Y", env.RESOLUTION_Y))        # the configuration's canvas (rendering: resolution_x, resolution_y): one size for the 2D window and the 3D view
        self.display = bool(cfg.get("display", False)); self.client = None; self.viewer = None; self.windows = None; self.index = {}
        self.needs_renderer = vc["enabled"] or any(s.needs_renderer for s in self.sensors.values()); self.missing = None
        if self.needs_renderer:                                 # without the renderer: what needs it is left out, the rest goes on
            from .client import renderer_missing
            rc = cfg.get("renderer") or {}; self.missing = renderer_missing(rc.get("python"))
            if self.missing is not None:
                gone = [n for n, s in self.sensors.items() if s.needs_renderer] + (["the 3D view"] if vc["enabled"] else [])
                msg = f"SocNavGym vision: {self.missing}; left out without it: {', '.join(gone)} (set-up: socnavgym/vision/README.md)."
                if rc.get("required", False): raise RuntimeError(msg)
                self.sensors = {n: s for n, s in self.sensors.items() if not s.needs_renderer}; vc["enabled"] = False; self.needs_renderer = False
                warnings.warn(msg + (" The simulation goes on with " + ", ".join(self.sensors) + "." if self.sensors else " The simulation goes on in 2D, as without a `vision:` section."), stacklevel=2)
        self.rng = np.random.default_rng(0); self.last = {}; self.links = {}; self.class_names = {v: k for k, v in CLASSES.items()}
        self.sensor_info = {n: s.info() for n, s in self.sensors.items()}; self.timing = {}
        if vc["enabled"]:
            from .viewer import OrbitCamera
            self.viewer = OrbitCamera(vc)
        self.shown = {VIEW: bool(vc["show"])} if self.viewer is not None else {}          # the rendered windows: name -> shown now? (switched with the menu on the 2D window)
        for n, s in self.sensors.items():
            for k in ([n + " rgb", n + " depth"] + ([n + " segmentation"] if s.ground_truth else []) if s.type == "rgbd" else [n]): self.shown[k] = self.display

    # ------------------------------------------------------------------ spaces and action
    def observation_spaces(self):
        d = {}
        if self.sensors: d["sensors"] = spaces.Dict({n: s.space() for n, s in self.sensors.items()})
        if self.nj: d["joints"] = spaces.Box(self.lower, self.upper, (self.nj,), np.float32)
        return d

    def action_bounds(self, low, high):
        return np.concatenate([low, -np.ones(self.nj, np.float32)]), np.concatenate([high, np.ones(self.nj, np.float32)])

    def split_action(self, action):
        """Takes the joints' part of the action (if given: an action of the old length leaves the targets as they were)."""
        a = np.asarray(action, np.float32).ravel()
        if self.nj and len(a) >= 3 + self.nj:
            u = np.clip(a[3:3 + self.nj], -1.0, 1.0); self.target = self.lower + (u + 1.0) / 2.0 * (self.upper - self.lower)
        return a[:3]

    def set_joint_targets(self, **targets):
        """Position targets in the joints' own units (radians / metres), by name."""
        for k, v in targets.items():
            i = [j.name for j in self.joints].index(k); self.target[i] = float(np.clip(v, self.lower[i], self.upper[i]))

    # ------------------------------------------------------------------ the renderer
    def start(self):
        if self.client is None and self.needs_renderer:
            from .client import RenderClient
            rc = dict(self.cfg.get("renderer") or {}); python = rc.pop("python", None); log = rc.pop("log", None); rc.pop("required", None)
            self.client = RenderClient(python=python, log=log, **rc)

    def map_box(self):
        """[x0, x1, y0, y1] of the map (SocNavGym's is centred on the origin)."""
        e = self.env; return [-e.MAP_X / 2, e.MAP_X / 2, -e.MAP_Y / 2, e.MAP_Y / 2]

    def scene_message(self):
        e = self.env; s = self.scene_cfg; b = self.map_box(); xs = list(b[:2]); ys = list(b[2:])
        for w in e.walls:
            c, sn = abs(math.cos(w.orientation)), abs(math.sin(w.orientation)); ex = (w.length * c + w.thickness * sn) / 2; ey = (w.length * sn + w.thickness * c) / 2
            xs += [w.x - ex, w.x + ex]; ys += [w.y - ey, w.y + ey]
        m = dict(bounds=[min(xs), max(xs), min(ys), max(ys)], ceiling_height=s["wall_height"], robot_id=int(e.robot.id) if e.robot.id is not None else 0)
        assert s["lighting"] in ("auto", "indoor", "outdoor"), f"vision: scene.lighting must be auto, indoor or outdoor, not '{s['lighting']}'"
        m["lighting"] = ("indoor" if len(e.walls) else "outdoor") if s["lighting"] == "auto" else s["lighting"]
        for k in ("floor_margin", "floor_color", "wall_color", "ceiling_color", "lamp_power", "lamp_spacing", "lamp_color", "sky_strength", "shadows"):
            if k in s: m[k] = s[k]
        m["walls"] = [[w.x, w.y, w.orientation, w.length, w.thickness, s["wall_height"], w.id] for w in e.walls]
        ents = [["table", t.id, t.x, t.y, t.orientation, t.length, t.width, 0.0, s["table_height"]] for t in e.tables]
        ents += [["chair", c.id, c.x, c.y, c.orientation, c.length, c.width, 0.0, s["chair_height"]] for c in e.chairs]
        ents += [["plant", p.id, p.x, p.y, p.orientation, 2 * p.radius, 2 * p.radius, 0.0, 0.0] for p in e.plants]
        ents += [["laptop", l.id, l.x, l.y, l.orientation, l.length, l.width, s["table_height"], s["laptop_height"]] for l in e.laptops]
        m["entities"] = ents; hs = []
        for i, h in enumerate(sorted(self.people(), key=lambda h: h.id)):          # the models take turns, so that every one is used
            r = np.random.default_rng(1000 + int(h.id)); asset = s["humans"][i % len(s["humans"])]
            hs.append([h.id, asset, h.x, h.y, h.orientation, s["human_height"], 1 + s["human_height_spread"] * float(r.uniform(-1, 1))])
        m["humans"] = hs
        m["robot_links"] = {n: [dict(geometry=v["geometry"], origin=v["origin"], color=v["color"]) for v in vis] for n, vis in self.urdf.links.items()}
        return m

    def people(self):
        """Every person of the scene (the environment keeps them in several lists; its own `all_humans` name is also used for a list)."""
        e = self.env; hs = list(e.static_humans) + list(e.dynamic_humans)
        for i in list(e.static_interactions) + list(e.moving_interactions): hs += list(i.humans)
        return hs + [i.human for i in e.h_l_interactions]

    # ------------------------------------------------------------------ per episode / per step
    def set_extent(self):
        """What stands still in this episode, for the framing of the 3D view and for the floor: the map, the walls, the furniture."""
        e = self.env; s = self.scene_cfg; b = self.map_box(); xs = list(b[:2]); ys = list(b[2:])
        for w in e.walls:
            c, sn = abs(math.cos(w.orientation)), abs(math.sin(w.orientation)); ex = (w.length * c + w.thickness * sn) / 2; ey = (w.length * sn + w.thickness * c) / 2
            xs += [w.x - ex, w.x + ex]; ys += [w.y - ey, w.y + ey]
        self.bounds = [min(xs), max(xs), min(ys), max(ys)]                       # the map and the walls: what the floor is made for
        for o in list(e.tables) + list(e.chairs) + list(e.laptops) + list(e.plants):
            r = o.radius if hasattr(o, "radius") else math.hypot(o.length, o.width) / 2; xs += [o.x - r, o.x + r]; ys += [o.y - r, o.y + r]
        self.still = [min(xs), max(xs), min(ys), max(ys)]; self.top = max(float(s["wall_height"]) if len(e.walls) else 0.0, 2.0)
        mg = float(s["floor_margin"]); self.floor = [self.bounds[0] - mg, self.bounds[1] + mg, self.bounds[2] - mg, self.bounds[3] + mg]

    def moving(self):
        """(n, 2) where the robot and the people are now."""
        e = self.env; return np.array([[e.robot.x, e.robot.y]] + [[h.x, h.y] for h in self.people()], float)

    def extent(self):
        """The box (lo, hi) around everything in the scene now: what stands still, the robot and the people (0.6 m around each)."""
        P = self.moving(); x0, x1, y0, y1 = self.still
        return np.array([min(x0, P[:, 0].min() - 0.6), min(y0, P[:, 1].min() - 0.6), 0.0]), np.array([max(x1, P[:, 0].max() + 0.6), max(y1, P[:, 1].max() + 0.6), self.top])

    def grow_floor(self):
        """Scenes without walls: when the robot or a person comes near the edge of the floor, the floor is made larger."""
        P = self.moving(); g = max(float(self.scene_cfg["floor_margin"]), 1.0); f = list(self.floor)
        if P[:, 0].min() - g < f[0]: f[0] = float(P[:, 0].min() - 2 * g)
        if P[:, 0].max() + g > f[1]: f[1] = float(P[:, 0].max() + 2 * g)
        if P[:, 1].min() - g < f[2]: f[2] = float(P[:, 1].min() - 2 * g)
        if P[:, 1].max() + g > f[3]: f[3] = float(P[:, 1].max() + 2 * g)
        if f != self.floor: self.floor = f; self.client.call("floor", bounds=f)

    def reset(self, obs, seed=None):
        self.rng = np.random.default_rng(seed if seed is not None else int(np.random.randint(2 ** 31)))
        self.q = self.q0.copy(); self.target = self.q0.copy(); self.start()
        if self.client is not None:
            r = self.client.call("scene", **self.scene_message()); self.index = {int(k): (v[0], int(v[1])) for k, v in r["index"].items()}
            n = max(self.index) + 1; self.lut_class = np.zeros(n, np.uint8); self.lut_inst = np.full(n, -1, np.int32)
            for k, (kind, eid) in self.index.items(): self.lut_class[k] = CLASSES.get(kind, 0); self.lut_inst[k] = eid
        self.set_extent()
        return self.observe(obs, advance=False)

    def observe(self, obs, advance=True):
        """Moves the joints one step, senses, and adds the readings to the observation `obs` (a dict)."""
        e = self.env
        if advance and self.nj:
            d = self.target - self.q; lim = np.where(self.vmax > 0, self.vmax * e.TIMESTEP, np.inf); self.q = (self.q + np.clip(d, -lim, lim)).astype(np.float32)
        self.links = self.urdf.fk({j.name: float(v) for j, v in zip(self.joints, self.q)}, base_pose(e.robot.x, e.robot.y, e.robot.orientation))
        out = {}; cams = []; lids = []
        for n, s in self.sensors.items():
            P = self.links[s.link]
            if s.type == "rgbd": cams.append(dict(name=n, pose=P, width=s.p["width"], height=s.p["height"], hfov=math.radians(s.p["hfov"]), exposure=s.p["exposure"], segmentation=s.ground_truth, **({"samples": s.p["samples"]} if s.p["samples"] else {})))
            elif s.p["mode"] == "mesh": lids.append(dict(name=n, pose=P, azimuths=s.azimuths, elevations=s.elevations, range_min=s.p["range_min"], range_max=s.p["range_max"], ignore_robot=s.p["ignore_robot"]))
            else: out[n] = self.simple_lidar(s, P)
        if self.client is not None:
            self.grow_floor(); hs = self.people()
            r = self.client.call("frame", t=float(e.ticks * e.TIMESTEP), human_ids=[int(h.id) for h in hs], humans=np.array([[h.x, h.y, h.orientation] for h in hs], np.float64).reshape(-1, 3),
                                 links=self.links, cameras=cams, lidars=lids)
            self.timing = dict(render=r["sec_render"], lidar=r["sec_lidar"])
            for n, g in r["cameras"].items(): out[n] = self.finish_rgbd(self.sensors[n], g, self.links[self.sensors[n].link])
            for n, g in r["lidars"].items(): out[n] = self.finish_lidar(self.sensors[n], g["ranges"], self.lut_class[g["index"]], self.lut_inst[g["index"]], self.links[self.sensors[n].link])
        self.last = out
        if self.sensors: obs["sensors"] = {n: out[n] for n in self.sensors}
        if self.nj: obs["joints"] = self.q.copy()
        return obs

    def finish_rgbd(self, s, g, P):
        z = g["depth"].copy(); p = s.p
        if p["depth_noise"]:                                    # a stereo depth camera's error grows with the square of the distance
            with np.errstate(invalid="ignore"):                 # pixels with nothing in front (the sky) are infinitely far: no reading, below
                z = z + self.rng.standard_normal(z.shape).astype(np.float32) * (z * z) * (p["subpixel"] / (s.focal * p["baseline"]))
        z[~np.isfinite(z) | (z < p["depth_min"]) | (z > p["depth_max"])] = 0.0      # no reading: 0, as the RealSense gives
        d = {"rgb": g["rgb"], "depth": z.astype(np.float32)}
        if s.ground_truth: d["segmentation"] = self.lut_class[g["index"]]; d["instance"] = self.lut_inst[g["index"]]; d["pose"] = P.astype(np.float32)
        return d

    def finish_lidar(self, s, ranges, cls, inst, P):
        p = s.p; r = np.asarray(ranges, np.float32).reshape(s.shape).copy(); hit = np.isfinite(r)
        if p["noise"] > 0: r = r + self.rng.standard_normal(r.shape).astype(np.float32) * float(p["noise"])
        hit &= (r >= p["range_min"]) & (r <= p["range_max"]); r[~hit] = {"inf": np.inf, "max": p["range_max"], "zero": 0.0}[p["no_return"]]
        d = {"ranges": r}
        if s.ground_truth:
            d["segmentation"] = np.where(hit, np.asarray(cls).reshape(s.shape), 0).astype(np.uint8); d["instance"] = np.where(hit, np.asarray(inst).reshape(s.shape), -1).astype(np.int32); d["pose"] = P.astype(np.float32)
        return d

    def simple_lidar(self, s, P):
        """A planar scan against the simulator's own 2D shapes (people and plants: circles; tables, chairs and walls: rectangles;
        laptops lie on the tables, above the scan). Needs no renderer."""
        e = self.env; o = P[:2, 3]; yaw = math.atan2(P[1, 0], P[0, 0]); a = yaw + s.azimuths; D = np.stack([np.cos(a), np.sin(a)], 1)
        best = np.full(len(a), np.inf); cls = np.zeros(len(a), np.uint8); inst = np.full(len(a), -1, np.int32)
        def keep(t, kind, eid):
            m = t < best; best[m] = t[m]; cls[m] = CLASSES[kind]; inst[m] = eid
        def circle(x, y, r, kind, eid):
            c = np.array([x, y]) - o; b = D @ c; disc = b * b - (c @ c - r * r); t = np.where(disc >= 0, b - np.sqrt(np.maximum(disc, 0)), np.inf); keep(np.where(t > 0, t, np.inf), kind, eid)
        def rect(x, y, th, L, W, kind, eid):
            c, sn = math.cos(th), math.sin(th); R = np.array([[c, sn], [-sn, c]]); p = R @ (o - np.array([x, y])); d = D @ R.T; h = np.array([L / 2, W / 2])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-h - p) / d; t2 = (h - p) / d
            lo = np.nanmax(np.minimum(t1, t2), 1); hi = np.nanmin(np.maximum(t1, t2), 1); inside = np.all(np.abs(p) <= h)
            t = np.where((lo <= hi) & (hi > 0), np.where(lo > 0, lo, np.inf if inside else lo), np.inf); keep(np.where(t > 0, t, np.inf), kind, eid)
        for h in self.people(): circle(h.x, h.y, h.width / 2, "human", h.id)
        for p in e.plants: circle(p.x, p.y, p.radius, "plant", p.id)
        for t in e.tables: rect(t.x, t.y, t.orientation, t.length, t.width, "table", t.id)
        for c in e.chairs: rect(c.x, c.y, c.orientation, c.length, c.width, "chair", c.id)
        for w in e.walls: rect(w.x, w.y, w.orientation, w.length, w.thickness, "wall", w.id)
        return self.finish_lidar(s, best, cls, inst, P)

    # ------------------------------------------------------------------ the 3D view and the sensors' windows
    def view(self, width=None, height=None, samples=None):
        """The image (H, W, 3 uint8 RGB) of the free 3D view at its present position; needs viewer.enabled."""
        assert self.viewer is not None and self.client is not None, "vision: the 3D view needs `viewer: {enabled: true}` and a reset"
        v = self.viewer_cfg
        if self.viewer.follow: self.viewer.target[:2] = (self.env.robot.x, self.env.robot.y)
        elif self.viewer.auto: self.viewer.frame(*self.extent(), math.radians(v["fov"]), float(width or v["width"]) / float(height or v["height"]))
        c = dict(name="view", pose=self.viewer.pose(), width=int(width or v["width"]), height=int(height or v["height"]), hfov=math.radians(v["fov"]), samples=int(samples or v["samples"]), overhead=False, segmentation=False, depth=False)
        return self.client.call("render", cameras=[c])["cameras"]["view"]["rgb"]

    def images(self, names=None):
        """The last readings as pictures (name -> H, W, 3 uint8 RGB), for display or for saving; `names`: only these."""
        from .viewer import depth_picture, scan_picture, label_picture
        out = {}; want = lambda k: names is None or k in names
        for n, s in self.sensors.items():
            g = self.last.get(n)
            if g is None: continue
            if s.type == "rgbd":
                if want(n + " rgb"): out[n + " rgb"] = g["rgb"]
                if want(n + " depth"): out[n + " depth"] = depth_picture(g["depth"], s.p["depth_max"])
                if s.ground_truth and want(n + " segmentation"): out[n + " segmentation"] = label_picture(g["segmentation"])
            elif want(n): out[n] = scan_picture(g["ranges"].reshape(len(s.elevations), -1)[len(s.elevations) // 2], s.azimuths, s.p["range_max"], g["segmentation"].reshape(len(s.elevations), -1)[len(s.elevations) // 2] if s.ground_truth else None)
        return out

    def menu_height(self):
        """Pixels that env.render() leaves free at the top of the 2D window for the menu (0: there is no window to show or hide)."""
        if not self.shown: return 0
        from .viewer import Windows
        return Windows.BAR

    def show(self, screen=None):
        """Called by env.render() with the 2D window: draws the menu that shows and hides the rendered windows (`self.shown`: name ->
        shown?), handles the events of all the windows (the menu, the 3D view's mouse and keys) and shows the windows that are on:
        the 3D view (not rendered while hidden) and the sensors' readings. False once the 2D window was closed or Q / Esc pressed."""
        if not self.shown: return True
        from .viewer import Windows
        if self.windows is None: self.windows = Windows()
        w = self.windows; alive = w.events(self.viewer, self.shown, VIEW)
        if screen is not None: w.menu(screen, self.shown); w.pg.display.flip()
        for n, on in self.shown.items():
            if not on: w.hide(n)
        if self.shown.get(VIEW) and self.client is not None:
            v = self.viewer_cfg; w.show(VIEW, self.view(*(w.size(VIEW) or (v["width"], v["height"]))), title="SocNavGym 3D view", resizable=True)
        names = {n for n, on in self.shown.items() if on and n != VIEW}
        for n, im in (self.images(names).items() if names else ()): w.show(n, im)
        return alive

    def close(self):
        if self.client is not None: self.client.close(); self.client = None
        if self.windows is not None: self.windows.close(); self.windows = None
