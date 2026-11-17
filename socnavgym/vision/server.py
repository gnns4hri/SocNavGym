"""The render server: a process of its own that keeps a 3D copy of the simulated scene in Blender and renders what the
sensors see. It runs with the RENDERER's python (Blender's `bpy` module; socnavgym itself is not imported here) and is started
by socnavgym.vision.client.RenderClient:

    python server.py <address of the client's listener>

Requests (dicts, see ipc.py), each answered with one dict:
  {"op": "init", ...settings}                 engine, samples, device
  {"op": "scene", ...}                        the scene of an episode: floor, walls, furniture, people, the robot's links
  {"op": "frame", ...}                        the moving things' poses for this instant + what to render -> the images
  {"op": "floor", "bounds": [x0, x1, y0, y1]} a larger floor for the present scene (the robot is leaving it)
  {"op": "close"}
Frames: metres, Z up; an entity's pose is (x, y, yaw) on the floor as in SocNavGym; a sensor's pose is a 4x4 matrix of its
frame in the world with X forward, Y left, Z up.
"""
import os, sys, math, time, traceback, tempfile
import numpy as np
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ipc  # noqa: E402

ASSETS = os.path.join(HERE, "assets")
CLASSES = ipc.CLASSES
# camera axes (X right, Y up, Z backwards) in the sensor frame (X forward, Y left, Z up)
CAM = Matrix(((0, 0, -1, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
YAW_OFFSET = {"laptop": -math.pi / 2}      # SocNavGym's laptop is used from its -Y side (human_laptop.py); the asset's front is +X


def pose2d(x, y, yaw, z=0.0):
    return Matrix.Translation((x, y, z)) @ Matrix.Rotation(yaw, 4, "Z")


# The sky used here (sun 50 degrees up), measured on a level floor: the light of the sun's disc / the light of the rest of the sky,
# and the colour of an even sky that lights level ground as the sun's disc does. SHADOWS: the default of scene.shadows.
SUN_SKY = 8.5; SUN_EVEN = (37.7, 31.0, 22.7); SHADOWS = 0.2


def tonemap(rgb, exposure):
    """Scene-linear float RGB -> 8-bit sRGB, with a filmic curve (ACES fit, K. Narkowicz) as a camera's processing would."""
    x = np.maximum(rgb * exposure, 0.0); x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14); x = np.clip(x, 0.0, 1.0)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (x * 255.0 + 0.5).astype(np.uint8)


class Server:
    def __init__(self):
        self.templates = {}; self.anims = {}; self.humans = {}; self.links = {}; self.entities = {}; self.index = {}; self.t_last = None
        self.exr = os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(), f"socnav_vision_{os.getpid()}.exr")
        self.exposure = 1.0

    # ------------------------------------------------------------------ set-up
    def init(self, m):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = self.sc = bpy.context.scene; sc.render.engine = "CYCLES"; dev = "CPU"; self.engine = str(m.get("engine", "cycles")).lower()
        assert self.engine in ("cycles", "eevee"), "renderer.engine must be cycles or eevee"
        # Draft mode, for debugging: the pictures are rendered smaller (half the width and height unless a number says otherwise),
        # with at most 4 samples and no denoiser, and enlarged to the size asked for: same sizes for whoever reads them, coarser.
        d = os.environ.get("SOCNAV_VISION_DRAFT", m.get("draft", False)); d = {"true": True, "false": False, "": False}.get(str(d).lower(), d)
        self.draft = 0.5 if d is True else float(d or 0.0); assert self.draft == 0.0 or 0.05 <= self.draft <= 1.0, "renderer.draft: true, false or a number from 0.05 to 1 (the pictures' size while rendering)"
        if self.draft: m = dict(m, denoise=False)
        if self.engine == "eevee": return self.init_rest(m, self.init_eevee(m))
        try:
            pr = bpy.context.preferences.addons["cycles"].preferences
            for kind in ([m["device"].upper()] if m.get("device", "auto") != "auto" else ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]):
                try: pr.compute_device_type = kind
                except TypeError: continue
                pr.get_devices(); gpus = [d for d in pr.devices if d.type == kind]
                if gpus:
                    for d in pr.devices: d.use = d.type == kind
                    dev = kind; break
        except Exception: traceback.print_exc()
        sc.cycles.device = "CPU" if dev == "CPU" else "GPU"; self.device = dev
        self.samples = int(m.get("samples", 24)); sc.cycles.samples = self.samples; sc.cycles.use_adaptive_sampling = False
        sc.cycles.use_denoising = bool(m.get("denoise", True))
        if sc.cycles.use_denoising:
            try: sc.cycles.denoiser = "OPTIX" if dev == "OPTIX" else "OPENIMAGEDENOISE"
            except TypeError: pass
        sc.cycles.max_bounces = int(m.get("bounces", 4)); sc.cycles.seed = int(m.get("seed", 0))
        return self.init_rest(m, dev)

    def init_eevee(self, m):
        """Blender's rasteriser: several times faster than Cycles on any graphics card, lighting less true (no path tracing)."""
        sc = self.sc
        for name in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try: sc.render.engine = name; break
            except TypeError: continue
        self.samples = int(m.get("samples", 16)); sc.eevee.taa_render_samples = self.samples
        for k, v in (("use_raytracing", True), ("use_shadows", True)):
            try: setattr(sc.eevee, k, v)
            except Exception: pass
        aov = bpy.context.view_layer.aovs.add(); aov.name = "idx"; aov.type = "VALUE"     # this engine has no object-index pass: the materials write it
        return "EEVEE"

    def tag_materials(self):
        """(eevee) every material also writes its object's index to the "idx" output."""
        for mt in bpy.data.materials:
            if not mt.use_nodes or mt.node_tree is None or any(n.type == "OUTPUT_AOV" for n in mt.node_tree.nodes): continue
            nt = mt.node_tree; oi = nt.nodes.new("ShaderNodeObjectInfo"); ao = nt.nodes.new("ShaderNodeOutputAOV"); ao.aov_name = "idx"; nt.links.new(oi.outputs["Object Index"], ao.inputs["Value"])

    def init_rest(self, m, dev):
        sc = self.sc; self.device = dev
        sc.render.use_persistent_data = True; sc.render.resolution_percentage = 100
        sc.render.threads_mode = "FIXED"; sc.render.threads = int(m.get("threads", 4))          # CPU side of the renderer (scene updates, denoising on CPU)
        ims = sc.render.image_settings
        try: ims.media_type = "MULTI_LAYER_IMAGE"                 # Blender 5: the multi-layer formats are a media type of their own
        except (AttributeError, TypeError): pass
        ims.file_format = "OPEN_EXR_MULTILAYER"; ims.color_depth = "32"
        try: sc.render.image_settings.exr_codec = "NONE"
        except TypeError: pass
        vl = bpy.context.view_layer; vl.use_pass_z = True
        if self.engine == "cycles": vl.use_pass_object_index = True
        self.exposure = float(m.get("exposure", 1.0))
        # world: a physical sky with the sun (the rooms have no ceiling unless one is asked for)
        w = bpy.data.worlds.new("world"); sc.world = w; w.use_nodes = True; nt = w.node_tree; bg = nt.nodes["Background"]
        try:
            sky = nt.nodes.new("ShaderNodeTexSky")
            for name in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA"):
                try: sky.sky_type = name; break
                except TypeError: continue
            sky.sun_elevation = math.radians(float(m.get("sun_elevation", 50.0))); sky.sun_rotation = math.radians(float(m.get("sun_azimuth", 130.0)))
            nt.links.new(sky.outputs[0], bg.inputs["Color"]); bg.inputs["Strength"].default_value = float(m.get("sky_strength", 0.05))
            # `bg` LIGHTS the scene: the sky with part of its sun (`sky`) + an even light from all of the sky (`even`) that stands for
            # the rest of the sun, which makes the sun's shadows lighter (build(), scene.shadows). What the cameras SEE of the sky
            # is another background, with the whole sun, so that the sky's picture does not change.
            self.sky = sky; self.even = nt.nodes.new("ShaderNodeVectorMath"); self.even.operation = "ADD"; self.even.inputs[1].default_value = (0.0, 0.0, 0.0)
            nt.links.new(sky.outputs[0], self.even.inputs[0]); nt.links.new(self.even.outputs[0], bg.inputs["Color"])
            seen = nt.nodes.new("ShaderNodeTexSky"); seen.sky_type = sky.sky_type; seen.sun_elevation = sky.sun_elevation; seen.sun_rotation = sky.sun_rotation
            self.bg_seen = nt.nodes.new("ShaderNodeBackground"); lp = nt.nodes.new("ShaderNodeLightPath"); mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(seen.outputs[0], self.bg_seen.inputs["Color"]); nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
            nt.links.new(bg.outputs[0], mix.inputs[1]); nt.links.new(self.bg_seen.outputs[0], mix.inputs[2])
            nt.links.new(mix.outputs[0], next(n for n in nt.nodes if n.type == "OUTPUT_WORLD").inputs["Surface"])
        except Exception:
            traceback.print_exc(); bg.inputs["Color"].default_value = (0.8, 0.86, 1.0, 1.0)
        cam = bpy.data.cameras.new("camera"); self.cam = bpy.data.objects.new("camera", cam); sc.collection.objects.link(self.cam); sc.camera = self.cam
        self.col = bpy.data.collections.new("episode"); sc.collection.children.link(self.col)
        self.mats = {}; self.tex = {}; f = os.path.join(ASSETS, "materials.blend")
        if os.path.exists(f):                                      # textured floor and walls (used unless the scene gives plain colours)
            with bpy.data.libraries.load(f) as (src, dst): dst.materials = [n for n in src.materials if n in ("floor", "wall")]
            self.tex = {mt.name.split(".")[0]: mt for mt in dst.materials if mt is not None}
        return {"device": dev, "engine": self.engine, "draft": self.draft, "blender": bpy.app.version_string, "classes": CLASSES}

    def material(self, name, color, rough=0.6, metal=0.0):
        key = (name, tuple(round(c, 3) for c in color), rough, metal)
        if key not in self.mats:
            mt = bpy.data.materials.new(name); mt.use_nodes = True; b = mt.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (*color[:3], 1.0); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
            self.mats[key] = mt
        return self.mats[key]

    # ------------------------------------------------------------------ assets
    def template(self, name):
        """The objects of an asset file, loaded once and kept out of the scene; instances are copies that share their data."""
        if name not in self.templates:
            with bpy.data.libraries.load(os.path.join(ASSETS, name + ".blend")) as (src, dst): dst.collections = ["asset"]
            col = dst.collections[0]; objs = list(col.objects); root = next(o for o in objs if o.type == "EMPTY" and o.parent is None)
            self.templates[name] = (objs, root, [float(v) for v in root["size"]])
            f = os.path.join(ASSETS, name + ".npz")
            if os.path.exists(f):
                d = np.load(f); arm = next(o for o in objs if o.type == "ARMATURE"); names = [b.name for b in arm.pose.bones]; a = {}
                for k in ("walk", "idle"):
                    Q = np.zeros((len(d[k + "_q"]), len(names), 4), np.float32); Q[..., 0] = 1.0
                    for j, b in enumerate(d["bones"]): Q[:, names.index(str(b))] = d[k + "_q"][:, j]
                    a[k] = (Q, d[k + "_loc"].astype(np.float32))
                a["root"] = names.index(str(d["bones"][0])); a["stride"] = float(d["stride"]); a["fps"] = float(d["fps"])
                self.anims[name] = a
        return self.templates[name]

    def instance(self, name, pass_index):
        objs, root, size = self.template(name); new = {}
        for o in objs:
            c = o.copy(); new[o] = c; self.col.objects.link(c); c.pass_index = pass_index
        for o, c in new.items():
            if o.parent in new: c.parent = new[o.parent]; c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
            for md in c.modifiers:
                if getattr(md, "object", None) in new: md.object = new[md.object]
        return new[root], list(new.values()), size

    # ------------------------------------------------------------------ the scene of an episode
    def scene(self, m):
        for o in list(self.col.objects): bpy.data.objects.remove(o, do_unlink=True)
        self.humans = {}; self.links = {}; self.entities = {}; self.index = {0: ("background", -1)}; self.t_last = None; nxt = [1]
        def idx(kind, eid):
            i = nxt[0]; nxt[0] += 1; self.index[i] = (kind, int(eid)); return i
        def add(obj, kind, eid):
            self.col.objects.link(obj); obj.pass_index = idx(kind, eid); return obj
        # floor
        x0, x1, y0, y1 = m["bounds"]; mg = float(m.get("floor_margin", 3.0)); me = bpy.data.meshes.new("floor")
        me.from_pydata([(x0 - mg, y0 - mg, 0), (x1 + mg, y0 - mg, 0), (x1 + mg, y1 + mg, 0), (x0 - mg, y1 + mg, 0)], [], [(0, 1, 2, 3)])
        fl = self.floor_obj = add(bpy.data.objects.new("floor", me), "floor", -1); me.materials.append(self.tex["floor"] if "floor" in self.tex and "floor_color" not in m else self.material("floor", m.get("floor_color", (0.45, 0.4, 0.34)), 0.45))
        self.overhead = []
        if m.get("lighting", "indoor") == "indoor":                # a ceiling with light panels; the free 3D view looks through it
            h = float(m.get("ceiling_height", 2.5)); me = bpy.data.meshes.new("ceiling")
            me.from_pydata([(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)], [], [(0, 3, 2, 1)])
            me.materials.append(self.material("ceiling", m.get("ceiling_color", (0.9, 0.9, 0.88)), 0.9)); self.overhead.append(add(bpy.data.objects.new("ceiling", me), "ceiling", -1))
            sp = float(m.get("lamp_spacing", 2.5)); nx = max(1, int(round((x1 - x0) / sp))); ny = max(1, int(round((y1 - y0) / sp)))
            for i in range(nx):
                for j in range(ny):
                    ld = bpy.data.lights.new("lamp", "AREA"); ld.shape = "RECTANGLE"; ld.size = 1.2; ld.size_y = 0.3
                    ld.energy = float(m.get("lamp_power", 20.0)) * ((x1 - x0) / nx) * ((y1 - y0) / ny) / (sp * sp); ld.color = m.get("lamp_color", (1.0, 0.96, 0.9))
                    lo = bpy.data.objects.new("lamp", ld); self.col.objects.link(lo); lo.location = (x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny, h - 0.01); self.overhead.append(lo)
        sky_s = float(m.get("sky_strength", (0.004 if self.engine == "eevee" else 0.05) if self.overhead else 0.04))   # open sky: 0.04 exposes a sunlit floor well (0.35 washed it out); eevee: the sky's light leaks through the ceiling
        if getattr(self, "sky", None) is not None:
            # the sun's shadows (scene.shadows, 0..1): sunlit ground is `(1 + SUN_SKY) ** shadows` times as bright as shaded ground
            # (1: the physical sky, the sun's shadows as dark as they come; 0: none). A share k of the sun's direct light stays; the
            # rest comes evenly from all of the sky, with the sun's colour and as much light on level ground, so what is in the sun
            # looks as before. Not with eevee under a ceiling, where the sky's light leaks into the room.
            sh = min(max(float(m.get("shadows", SHADOWS)), 0.0), 1.0); k = 1 - ((1 + SUN_SKY) ** (1 - sh) - 1) / SUN_SKY
            if self.engine == "eevee" and self.overhead: k = 1.0
            try: self.sky.sun_intensity = k; self.even.inputs[1].default_value = tuple((1 - k) * c for c in SUN_EVEN)
            except Exception: traceback.print_exc()
            self.bg_seen.inputs["Strength"].default_value = sky_s
        self.sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = sky_s
        # walls: boxes (x, y, yaw, length, thickness) of height h
        for wl in m.get("walls", []):
            x, y, yaw, L, th, h, eid = wl; me = bpy.data.meshes.new("wall"); a, b = L / 2, th / 2
            v = [(-a, -b, 0), (a, -b, 0), (a, b, 0), (-a, b, 0), (-a, -b, h), (a, -b, h), (a, b, h), (-a, b, h)]
            me.from_pydata(v, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            me.materials.append(self.tex["wall"] if "wall" in self.tex and "wall_color" not in m else self.material("wall", m.get("wall_color", (0.86, 0.85, 0.82)), 0.85))
            o = add(bpy.data.objects.new("wall", me), "wall", eid); o.matrix_world = pose2d(x, y, yaw)
        # furniture: (kind, id, x, y, yaw, size_x, size_y, z, height or 0)
        for e in m.get("entities", []):
            kind, eid, x, y, yaw, sx, sy, z, h = e; i = idx(kind, eid); root, objs, size = self.instance(kind, i)
            if kind == "laptop": fx, fy = sx / size[1], sy / size[0]        # turned by YAW_OFFSET: the asset's y lies along the entity's x
            else: fx, fy = sx / size[0], sy / size[1]
            if kind == "plant": fx = fy = fz = max(fx, fy)
            else: fz = h / size[2] if h > 0 else float(np.clip((fx + fy) / 2, 0.75, 1.25))
            sc = (fy, fx, fz) if kind == "laptop" else (fx, fy, fz)
            root.matrix_world = pose2d(x, y, yaw + YAW_OFFSET.get(kind, 0.0), z) @ Matrix.Diagonal((*sc, 1.0))
            self.entities[(kind, int(eid))] = (root, sc, size[2] * fz)
        # people: (id, asset, x, y, yaw, height or 0 for the asset's own, factor on it)
        for hmn in m.get("humans", []):
            eid, asset, x, y, yaw, h, k = hmn; i = idx("human", eid); root, objs, size = self.instance(asset, i); s = (h / size[2] if h > 0 else 1.0) * k
            arm = next(o for o in objs if o.type == "ARMATURE"); a = self.anims[asset]
            self.humans[int(eid)] = dict(root=root, arm=arm, anim=a, scale=s, phase=float(np.random.default_rng(int(eid)).uniform(0, 1)), w=0.0, t_idle=float(np.random.default_rng(int(eid) + 7).uniform(0, 5)), last=(x, y))
            root.matrix_world = pose2d(x, y, yaw) @ Matrix.Scale(s, 4)
        # the robot: per link an empty that carries the link's visuals
        ri = idx("robot", m.get("robot_id", 0))
        for name, visuals in m.get("robot_links", {}).items():
            e = bpy.data.objects.new("link_" + name, None); self.col.objects.link(e); self.links[name] = e
            for v in visuals:
                o = self.visual(v)
                if o is None: continue
                objs = o if isinstance(o, list) else [o]
                for ob in objs:
                    if ob.name not in self.col.objects: self.col.objects.link(ob)
                    ob.pass_index = ri
                    if ob.parent is None: ob.parent = e; ob.matrix_parent_inverse = Matrix.Identity(4); ob.matrix_basis = Matrix(np.array(v["origin"]).reshape(4, 4).tolist()) @ ob.matrix_basis
        if self.engine == "eevee": self.tag_materials()
        bpy.context.view_layer.update()
        return {"index": {str(k): list(v) for k, v in self.index.items()}, "heights": {f"{k[0]}:{k[1]}": v[2] for k, v in self.entities.items()}}

    def visual(self, v):
        g = v["geometry"]; col = v.get("color") or (0.7, 0.7, 0.72, 1.0); mat = self.material("robot", col, 0.5, 0.1)
        if g["type"] == "mesh":
            f = g["filename"]; ext = os.path.splitext(f)[1].lower(); before = set(bpy.data.objects)
            try:
                if ext == ".stl": bpy.ops.wm.stl_import(filepath=f)
                elif ext == ".obj": bpy.ops.wm.obj_import(filepath=f, forward_axis="Y", up_axis="Z")
                elif ext == ".dae": bpy.ops.wm.collada_import(filepath=f)
                elif ext in (".glb", ".gltf"): bpy.ops.import_scene.gltf(filepath=f)
                else: print("vision server: unsupported mesh format", f, flush=True); return None
            except Exception:
                traceback.print_exc(); return None
            new = [o for o in bpy.data.objects if o not in before]; s = g.get("scale") or (1, 1, 1)
            for o in new:
                for c in list(o.users_collection): c.objects.unlink(o)
                if o.parent is None: o.matrix_basis = Matrix.Diagonal((*s, 1.0)) @ o.matrix_basis
                if o.type == "MESH" and ext == ".stl": o.data.materials.append(mat)
            return new
        me = bpy.data.meshes.new("visual"); import bmesh; bm = bmesh.new()
        if g["type"] == "box": bmesh.ops.create_cube(bm, size=1.0); sc = g["size"]
        elif g["type"] == "cylinder": bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=g["radius"], radius2=g["radius"], depth=g["length"]); sc = (1, 1, 1)
        elif g["type"] == "sphere": bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=g["radius"]); sc = (1, 1, 1)
        else: bm.free(); return None
        bm.to_mesh(me); bm.free(); me.materials.append(mat)
        for p in me.polygons: p.use_smooth = g["type"] == "sphere" or (g["type"] == "cylinder" and abs(p.normal.z) < 0.5)   # a cylinder's caps stay flat
        o = bpy.data.objects.new("visual", me); o.matrix_basis = Matrix.Diagonal((*sc, 1.0)); return o

    # ------------------------------------------------------------------ one instant
    def floor(self, m):
        """A new extent (x0, x1, y0, y1) for the floor of the present scene (scenes without walls: the robot is leaving it)."""
        x0, x1, y0, y1 = m["bounds"]; me = self.floor_obj.data
        for v, co in zip(me.vertices, [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]): v.co = co
        me.update(); return {}

    def frame(self, m):
        t = float(m["t"]); dt = 0.0 if self.t_last is None else max(t - self.t_last, 0.0); self.t_last = t
        for eid, (x, y, yaw) in zip(m.get("human_ids", []), np.asarray(m.get("humans", np.zeros((0, 3)))).reshape(-1, 3)):
            h = self.humans.get(int(eid))
            if h is None: continue
            d = math.hypot(x - h["last"][0], y - h["last"][1]); h["last"] = (float(x), float(y)); a = h["anim"]
            speed = d / dt if dt > 0 else 0.0; target = 1.0 if speed > 0.08 else 0.0
            h["w"] += (target - h["w"]) * (1.0 - math.exp(-dt / 0.25)) if dt > 0 else 0.0
            h["phase"] = (h["phase"] + d / (a["stride"] * h["scale"])) % 1.0; h["t_idle"] += dt
            self.pose(h); h["root"].matrix_world = pose2d(float(x), float(y), float(yaw)) @ Matrix.Scale(h["scale"], 4)
        for name, M in (m.get("links") or {}).items():
            if name in self.links: self.links[name].matrix_world = Matrix(np.asarray(M).reshape(4, 4).tolist())
        for key, (x, y, yaw, z) in (m.get("entities") or {}).items():
            kind, eid = key.split(":"); e = self.entities.get((kind, int(eid)))
            if e is not None: e[0].matrix_world = pose2d(x, y, yaw + YAW_OFFSET.get(kind, 0.0), z) @ Matrix.Diagonal((*e[1], 1.0))
        out = {"cameras": {}, "lidars": {}}; t0 = time.time()
        for c in m.get("cameras", []): out["cameras"][c["name"]] = self.render(c)
        t1 = time.time()
        if m.get("lidars"):
            bpy.context.view_layer.update(); dg = bpy.context.evaluated_depsgraph_get()
            for l in m["lidars"]: out["lidars"][l["name"]] = self.lidar(l, dg)
        out["sec_render"] = t1 - t0; out["sec_lidar"] = time.time() - t1
        return out

    def render_only(self, m):
        return {"cameras": {c["name"]: self.render(c) for c in m.get("cameras", [])}}

    def pose(self, h):
        a = h["anim"]; arm = h["arm"]; Qw, Lw = a["walk"]; Qi, Li = a["idle"]
        def at(Q, L, f, loop):
            n = len(Q) - 1 if loop else len(Q); i0 = int(math.floor(f)) % n; i1 = (i0 + 1) % n if loop else min(i0 + 1, n - 1); u = f - math.floor(f)
            q1 = Q[i1] * np.where((Q[i0] * Q[i1]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
            return (1 - u) * Q[i0] + u * q1, (1 - u) * L[i0] + u * L[i1]
        qw, lw = at(Qw, Lw, h["phase"] * (len(Qw) - 1), True)
        fi = h["t_idle"] * a["fps"]; ni = len(Qi) - 1; fi = fi % (2 * ni); fi = fi if fi <= ni else 2 * ni - fi          # back and forth: no jump at the end
        qi, li = at(Qi, Li, min(fi, ni - 1e-3), False); w = h["w"]
        qw = qw * np.where((qi * qw).sum(-1, keepdims=True) < 0, -1.0, 1.0); q = (1 - w) * qi + w * qw; q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
        arm.pose.bones.foreach_set("rotation_quaternion", q.astype(np.float32).ravel())
        arm.pose.bones[a["root"]].location = ((1 - w) * li + w * lw).tolist(); arm.update_tag()

    def render(self, c):
        sc = self.sc; cam = self.cam; W, H = int(c["width"]), int(c["height"])
        cam.matrix_world = Matrix(np.asarray(c["pose"]).reshape(4, 4).tolist()) @ CAM
        cam.data.sensor_fit = "HORIZONTAL"; cam.data.angle_x = float(c["hfov"]); cam.data.clip_start = float(c.get("near", 0.05)); cam.data.clip_end = float(c.get("far", 100.0))
        FW, FH = W, H
        if self.draft: W, H = max(8, int(round(W * self.draft))), max(8, int(round(H * self.draft)))
        sc.render.resolution_x = W; sc.render.resolution_y = H
        see = bool(c.get("overhead", True))                        # False: the ceiling and the lamps are not seen by this camera (they still light the room)
        for ob in self.overhead:
            if ob.visible_camera != see: ob.visible_camera = see
        n = int(c.get("samples", self.samples)); sc.render.filepath = self.exr
        if self.draft: n = min(n, 4)
        if self.engine == "cycles": sc.cycles.samples = n
        else: sc.eevee.taa_render_samples = n
        t0 = time.time(); bpy.ops.render.render(write_still=True); t1 = time.time(); ch = self.read_exr()
        if os.environ.get("SOCNAV_VISION_PROFILE"): print(f"profile {c.get('name')} {W}x{H} {n} samples: render {t1 - t0:.3f} s, read {time.time() - t1:.3f} s", flush=True)
        if self.engine == "eevee" and c.get("segmentation", True):
            # the index written by the materials is blended at the objects' edges when several samples are taken: one more picture with a single sample
            sc.eevee.taa_render_samples = 1; bpy.ops.render.render(write_still=True); ch["Index"] = next((v for k, v in self.read_exr().items() if "idx" in k), None)
            if ch["Index"] is None: del ch["Index"]
        def get(*words):
            for k, v in ch.items():
                if all(w in k for w in words) and "Noisy" not in k and "idx" not in k: return v
            return None
        rgb = get("Combined")
        out = {"rgb": tonemap(rgb[..., :3].astype(np.float32), self.exposure * float(c.get("exposure", 1.0)))}
        z = get("Depth")
        if z is not None and c.get("depth", True):               # Cycles' Z pass (Blender 5.3): distance along the optical axis, as a depth camera gives;
            z = np.asarray(z, np.float32).reshape(H, W); out["depth"] = np.where(z > 0.99 * float(c.get("far", 100.0)), np.inf, z).astype(np.float32)   # nothing hit: inf
        ix = get("Index")
        if ix is not None and c.get("segmentation", True): out["index"] = np.rint(np.asarray(ix).reshape(H, W)).astype(np.int32)
        if (W, H) != (FW, FH):                                   # draft: enlarged to the size asked for (nearest pixel: depth and labels stay real values)
            r = np.minimum(np.arange(FH) * H // FH, H - 1)[:, None]; q = np.minimum(np.arange(FW) * W // FW, W - 1)[None, :]
            out = {k: np.ascontiguousarray(v[r, q]) for k, v in out.items()}
        return out

    def read_exr(self):
        import OpenEXR
        ch = {}
        with OpenEXR.File(self.exr) as f:                        # one part per pass: "<layer>.Combined" (RGBA), "<layer>.Depth.Z", "<layer>.Object Index.X"
            for p in f.parts:
                for k, v in p.channels.items(): ch[k] = np.asarray(v.pixels)
        return ch

    def lidar(self, l, dg):
        """Rays from the sensor's origin against the scene's real geometry (animated people included): ranges (rings, beams)."""
        M = Matrix(np.asarray(l["pose"]).reshape(4, 4).tolist()); o = M.to_translation(); R = M.to_3x3(); rmax = float(l["range_max"]); rmin = float(l.get("range_min", 0.0))
        skip = bool(l.get("ignore_robot", True)); az = np.asarray(l["azimuths"], float); el = np.asarray(l.get("elevations", [0.0]), float); out = np.full((len(el), len(az)), np.inf, np.float32); ids = np.zeros((len(el), len(az)), np.int32)
        for i, e in enumerate(el):
            ce, se = math.cos(e), math.sin(e)
            for j, a in enumerate(az):
                d = R @ Vector((ce * math.cos(a), ce * math.sin(a), se)); start = rmin
                for _ in range(8):                               # the robot's own body is transparent to its lidar unless asked otherwise
                    hit, loc, nrm, fi, ob, mw = self.sc.ray_cast(dg, o + d * start, d, distance=rmax - start)
                    if not hit: break
                    pi = ob.original.pass_index if ob is not None else 0
                    if skip and self.index.get(pi, ("", 0))[0] == "robot": start = (loc - o).length + 1e-3; continue
                    out[i, j] = (loc - o).length; ids[i, j] = pi; break
        return {"ranges": out, "index": ids}


def main():
    from multiprocessing.connection import Client
    conn = Client(sys.argv[-1], family="AF_UNIX"); s = Server()
    while True:
        try: m = ipc.unpack(conn.recv_bytes())
        except (EOFError, ConnectionResetError): break
        try:
            if m["op"] == "close": conn.send_bytes(ipc.pack({"ok": True})); break
            r = getattr(s, {"render": "render_only"}.get(m["op"], m["op"]))(m); r["ok"] = True
        except Exception:
            r = {"ok": False, "error": traceback.format_exc()}
        conn.send_bytes(ipc.pack(r))
    try: os.remove(s.exr)
    except OSError: pass


if __name__ == "__main__": main()
