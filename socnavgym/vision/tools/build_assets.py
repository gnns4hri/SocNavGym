"""Builds the 3D assets the simulator loads (socnavgym/vision/assets/) from the sources fetched by fetch_assets.py.
Run with the RENDERER's python (the one that has Blender's `bpy` module):

    .venv_render/bin/python socnavgym/vision/tools/build_assets.py [names ...]      (default: all)

Every asset is a .blend file with ONE collection called "asset":
  - an empty "ROOT" at the origin, everything else parented to it;
  - metres, Z up, the asset standing on z = 0, centred in x and y, its FRONT towards +X
    (the entity frame of SocNavGym: x along the entity's orientation / `length`, y along its `width`);
  - ROOT["size"] = (x, y, z) extent in metres, ROOT["kind"] = table | chair | plant | laptop | human.
materials.blend holds the textured materials "floor" and "wall".
People also get <name>.npz with the poses of the two animations, sampled per frame and already expressed for the avatar's own
skeleton (so that playing them needs no animation system: the renderer writes the bone rotations directly):
  bones (B,) names; walk_q / idle_q (F, B, 4) bone rotations (w, x, y, z; relative to the rest pose); walk_loc / idle_loc (F, 3)
  location of the root bone (pelvis; the walk's forward travel is removed: the simulator moves the person); fps; stride (metres
  travelled in one walking cycle, to advance the cycle with the distance walked so that the feet do not slide).
"""
import os, sys, glob, math
import bpy, numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CACHE = os.environ.get("SOCNAV_VISION_CACHE", os.path.join(REPO, "vision_cache")); OUT = os.path.abspath(os.path.join(HERE, "..", "assets"))

# name -> (kind, Poly Haven id, yaw in degrees that turns the model's front to +X)
FURNITURE = {"table": ("table", "metal_office_desk", 0.0), "chair": ("chair", "SchoolChair_01", 90.0),
             "plant": ("plant", "potted_plant_02", 0.0), "laptop": ("laptop", "classic_laptop", 90.0)}
# name -> (Rocketbox avatar, walking animation, standing animation)
HUMANS = {"human_male_01": ("Male_Adult_01", "xy/m_walk_neutral", "static/m_idle_neutral_01"),
          "human_female_01": ("Female_Adult_01", "xy/f_walk_neutral", "static/f_idle_neutral_01")}
# material name -> (Poly Haven texture, metres covered by one repetition of the picture)
# + a paint colour and how much of it covers the picture's own colour (0: the picture as it is)
SURFACES = {"floor": ("laminate_floor_02", 1.7, None, 0.0), "wall": ("white_plaster_02", 1.0, (0.87, 0.86, 0.82), 0.8)}
TEX_SIZE = 1024          # the avatars' textures are stored at this size (sources: 2048, 12 MB each)


def clean():
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.context.preferences.filepaths.save_version = 0      # no .blend1 backup copies


def bbox(objs):
    pts = np.array([(o.matrix_world @ Vector(c))[:] for o in objs if o.type == "MESH" for c in o.bound_box])
    return pts.min(0), pts.max(0)


def finish(name, kind, objs, yaw_deg, extra=None):
    """Parents the top-level objects to ROOT so that the asset stands on z = 0, centred, front to +X; saves <name>.blend."""
    bpy.context.view_layer.update()
    root = bpy.data.objects.new("ROOT", None); col = bpy.data.collections.new("asset"); bpy.context.scene.collection.children.link(col)
    tops = [o for o in objs if o.parent is None]
    R = Matrix.Rotation(math.radians(yaw_deg), 4, "Z")
    for o in tops: o.matrix_world = R @ o.matrix_world
    bpy.context.view_layer.update(); lo, hi = bbox(objs); c = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    if extra and "centre" in extra: c = Vector(extra["centre"])
    for o in tops: o.matrix_world = Matrix.Translation(-c) @ o.matrix_world
    bpy.context.view_layer.update()
    for o in objs:
        for cc in list(o.users_collection): cc.objects.unlink(o)
        col.objects.link(o)
    col.objects.link(root)
    for o in tops: o.parent = root; o.matrix_parent_inverse = Matrix.Identity(4)
    root["size"] = [float(v) for v in (hi - lo)]; root["kind"] = kind
    for im in bpy.data.images:
        if im.source == "FILE" and im.filepath and not im.packed_file:
            try: im.pack()
            except RuntimeError as e: print("   could not pack", im.name, e)
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, name + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, copy=True)
    print(f"{name}: {kind}, size {np.round(hi - lo, 3)} m, {os.path.getsize(path) / 1e6:.1f} MB")


def build_furniture(name):
    kind, ph, yaw = FURNITURE[name]; clean()
    f = glob.glob(os.path.join(CACHE, "polyhaven", ph, "*.gltf"))[0]
    bpy.ops.import_scene.gltf(filepath=f)
    finish(name, kind, list(bpy.data.objects), yaw)


def build_surfaces():
    """materials.blend: the textured materials of the floor and the walls. They are laid out in metres from the object's own
    coordinates (floor: seen from above; walls: from the six sides), so they need no UV maps and do not stretch."""
    clean()
    for name, (ph, size, paint, cover) in SURFACES.items():
        d = os.path.join(CACHE, "polyhaven", ph); mt = bpy.data.materials.new(name); mt.use_nodes = True; mt.use_fake_user = True; nt = mt.node_tree; b = nt.nodes["Principled BSDF"]
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1 / size, 1 / size, 1 / size); nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        def tex(f, data):
            n = nt.nodes.new("ShaderNodeTexImage"); n.image = bpy.data.images.load(os.path.join(d, f)); n.image.pack()
            if data: n.image.colorspace_settings.name = "Non-Color"
            if name != "floor": n.projection = "BOX"; n.projection_blend = 0.15
            nt.links.new(mp.outputs["Vector"], n.inputs["Vector"]); return n
        col = tex("Diffuse.jpg", False).outputs["Color"]
        if paint is not None:                                    # painted: the relief of the plaster stays, its stains mostly go
            mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.inputs["Factor"].default_value = cover; mx.inputs["B"].default_value = (*paint, 1.0)
            nt.links.new(col, mx.inputs["A"]); col = mx.outputs["Result"]
        nt.links.new(col, b.inputs["Base Color"]); nt.links.new(tex("Rough.jpg", True).outputs["Color"], b.inputs["Roughness"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(tex("nor_gl.jpg", True).outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, "materials.blend"); bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, copy=True)
    print(f"materials: {list(SURFACES)}, {os.path.getsize(path) / 1e6:.1f} MB")


def curves_of(action):
    try: return [fc for l in action.layers for s in l.strips for cb in s.channelbags for fc in cb.fcurves]
    except AttributeError: return list(action.fcurves)


def sample_animation(arm, src, f0, f1):
    """The pose of `src` (the animation file's armature, same bone names) at every frame f0..f1, as rotations of `arm`'s bones
    relative to ITS rest pose (the two rest poses differ: the animation's is not the avatar's T-pose) + the root's location.
    -> q (F, B, 4), loc (F, 3) in arm's armature space, bone names."""
    sc = bpy.context.scene; order = []
    def walk(b):
        order.append(b.name)
        for c in b.children: walk(c)
    for b in arm.data.bones:
        if b.parent is None: walk(b)
    order = [n for n in order if n in src.pose.bones]
    Q = np.zeros((f1 - f0 + 1, len(order), 4)); L = np.zeros((f1 - f0 + 1, 3)); Q[..., 0] = 1
    for i, f in enumerate(range(f0, f1 + 1)):
        sc.frame_set(f); bpy.context.view_layer.update(); des = {}
        to_arm = arm.matrix_world.inverted() @ src.matrix_world          # per frame: the animation also moves the armature OBJECT (the travel)
        for j, n in enumerate(order):
            b = arm.data.bones[n]; M = to_arm @ src.pose.bones[n].matrix; des[n] = M
            if b.parent is not None and b.parent.name in des:
                basis = b.convert_local_to_pose(M, b.matrix_local, parent_matrix=des[b.parent.name], parent_matrix_local=b.parent.matrix_local, invert=True)
            else:
                basis = b.convert_local_to_pose(M, b.matrix_local, invert=True); L[i] = basis.to_translation()[:]
            Q[i, j] = basis.to_quaternion().normalized()[:]
    return Q, L, order


def build_human(name):
    avatar, walk, idle = HUMANS[name]; clean(); d = os.path.join(CACHE, "rocketbox")
    bpy.ops.import_scene.fbx(filepath=os.path.join(d, "Avatars", "Adults", avatar, "Export", avatar + ".fbx"))
    objs = [o for o in bpy.data.objects if o.type in ("ARMATURE", "MESH")]
    for o in [o for o in bpy.data.objects if o not in objs]: bpy.data.objects.remove(o)
    arm = [o for o in objs if o.type == "ARMATURE"][0]
    # textures: the file names stored in the FBX point to the authors' disk; take them from the Textures folder, smaller, packed
    tdir = os.path.join(d, "Avatars", "Adults", avatar, "Textures")
    for im in list(bpy.data.images):
        f = os.path.join(tdir, os.path.basename(im.filepath.replace("\\", "/")))
        if not os.path.exists(f): continue
        im.filepath = f; im.reload()
        if im.size[0] > TEX_SIZE: im.scale(TEX_SIZE, TEX_SIZE)
        png = os.path.join(CACHE, "tmp_tex", os.path.splitext(os.path.basename(f))[0] + ".png"); os.makedirs(os.path.dirname(png), exist_ok=True)
        im.filepath_raw = png; im.file_format = "PNG"; im.save(); im.filepath = png; im.reload(); im.pack()
    for m in bpy.data.materials:                                 # plain skin / cloth; the "opacity" material (hair, lashes) uses its texture's alpha
        nt = m.node_tree; bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None: continue
        bsdf.inputs["Roughness"].default_value = 0.6
        for s in ("Specular IOR Level", "Metallic"):
            if s in bsdf.inputs:
                for l in list(bsdf.inputs[s].links): nt.links.remove(l)
        bsdf.inputs["Metallic"].default_value = 0.0
        if "opacity" in m.name.lower():
            tex = next((l.from_node for l in bsdf.inputs["Base Color"].links if l.from_node.type == "TEX_IMAGE"), None)
            if tex is not None: nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            for attr, val in (("blend_method", "HASHED"), ("surface_render_method", "DITHERED")):
                try: setattr(m, attr, val)
                except Exception: pass
    out = {}
    for key, rel in (("walk", walk), ("idle", idle)):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.fbx(filepath=os.path.join(d, "Animations", f"all_animations_max_motextr_{rel.split('/')[0]}", rel.split("/")[1] + ".max.fbx"))
        new = [o for o in bpy.data.objects if o not in before]; src = [o for o in new if o.type == "ARMATURE"][0]
        f0, f1 = [int(round(v)) for v in src.animation_data.action.frame_range]
        Q, L, order = sample_animation(arm, src, f0, f1); out[key + "_q"] = Q; out[key + "_loc"] = L; out["bones"] = np.array(order)
        for o in new: bpy.data.objects.remove(o)
    fps = float(bpy.context.scene.render.fps)
    # walking in place: the net travel of the root over the cycle is removed (the bobbing and the sway stay); its length is the stride
    Lw = out["walk_loc"]; n = len(Lw) - 1; travel = Lw[-1] - Lw[0]; out["walk_loc"] = Lw - np.linspace(0, 1, n + 1)[:, None] * travel[None]
    scale = float(np.mean([abs(s) for s in arm.matrix_world.to_scale()]))
    out["stride"] = float(np.linalg.norm(travel)) * scale; out["fps"] = fps
    # the walk's direction in the world (before the asset is turned): it must end up along +X
    wdir = (arm.matrix_world.to_3x3() @ (arm.data.bones[order[0]].matrix_local.to_3x3() @ Vector(travel))); yaw = -math.degrees(math.atan2(wdir.y, wdir.x))
    for pb in arm.pose.bones: pb.rotation_mode = "QUATERNION"
    arm.animation_data_clear()
    # the centre of the asset: under the pelvis of the standing pose, on the floor
    root_b = arm.data.bones[order[0]]; p = arm.matrix_world @ root_b.matrix_local.to_translation()
    R = Matrix.Rotation(math.radians(yaw), 4, "Z"); pc = R @ Vector((p.x, p.y, 0.0))
    np.savez_compressed(os.path.join(OUT, name + ".npz"), **out)
    finish(name, "human", objs, yaw, extra={"centre": (pc.x, pc.y, 0.0)})
    print(f"   walk: {len(out['walk_q'])} frames at {fps:.0f} fps, stride {out['stride']:.2f} m -> {out['stride'] / (n / fps):.2f} m/s as recorded; idle: {len(out['idle_q'])} frames; {len(order)} bones")


if __name__ == "__main__":
    names = sys.argv[1:] or ["materials"] + list(FURNITURE) + list(HUMANS)
    for n in names: build_surfaces() if n == "materials" else build_furniture(n) if n in FURNITURE else build_human(n)
