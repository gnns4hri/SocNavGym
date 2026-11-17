"""A small URDF reader: links, joints, visuals and forward kinematics. Only what the vision system needs (the robot's dynamics
stay SocNavGym's: the base moves as the simulator says; the joints are moved in position with a speed limit).
Supported: joints fixed / revolute / continuous / prismatic; visuals box / cylinder / sphere / mesh, with material colours.
Mesh file names: absolute, relative to the URDF's folder, or package://<pkg>/... (looked for under the folders given in
`packages`, under the URDF's folder and its parents)."""
import os, math
import xml.etree.ElementTree as ET
import numpy as np


def rpy_matrix(xyz, rpy):
    r, p, y = rpy; cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    M = np.eye(4); M[:3, 3] = xyz
    M[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]]
    return M


def axis_angle(axis, a):
    x, y, z = np.asarray(axis, float) / (np.linalg.norm(axis) + 1e-12); c, s = math.cos(a), math.sin(a); C = 1 - c; M = np.eye(4)
    M[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return M


def _origin(e):
    o = e.find("origin") if e is not None else None
    if o is None: return np.eye(4)
    f = lambda s: [float(v) for v in (s or "0 0 0").split()]
    return rpy_matrix(f(o.get("xyz")), f(o.get("rpy")))


class Joint:
    def __init__(self, name, type, parent, child, origin, axis, lower, upper, velocity):
        self.name, self.type, self.parent, self.child, self.origin, self.axis = name, type, parent, child, origin, axis
        self.lower, self.upper, self.velocity = lower, upper, velocity

    def transform(self, q):
        if self.type in ("revolute", "continuous"): return self.origin @ axis_angle(self.axis, q)
        if self.type == "prismatic":
            T = np.eye(4); T[:3, 3] = np.asarray(self.axis, float) * q; return self.origin @ T
        return self.origin


class URDF:
    def __init__(self, path, packages=()):
        self.path = os.path.abspath(os.path.expanduser(path)); self.dir = os.path.dirname(self.path); self.packages = [os.path.abspath(os.path.expanduser(p)) for p in packages]
        root = ET.parse(self.path).getroot(); self.name = root.get("name", "robot")
        colors = {m.get("name"): [float(v) for v in m.find("color").get("rgba").split()] for m in root.findall("material") if m.find("color") is not None}
        self.links = {}; self.joints = []
        for l in root.findall("link"):
            vis = []
            for v in l.findall("visual"):
                g = v.find("geometry"); geo = None
                if g is None: continue
                if g.find("box") is not None: geo = {"type": "box", "size": [float(x) for x in g.find("box").get("size").split()]}
                elif g.find("cylinder") is not None: geo = {"type": "cylinder", "radius": float(g.find("cylinder").get("radius")), "length": float(g.find("cylinder").get("length"))}
                elif g.find("sphere") is not None: geo = {"type": "sphere", "radius": float(g.find("sphere").get("radius"))}
                elif g.find("mesh") is not None:
                    m = g.find("mesh"); geo = {"type": "mesh", "filename": self.resolve(m.get("filename")), "scale": [float(x) for x in (m.get("scale") or "1 1 1").split()]}
                if geo is None: continue
                mt = v.find("material"); col = None
                if mt is not None: col = [float(x) for x in mt.find("color").get("rgba").split()] if mt.find("color") is not None else colors.get(mt.get("name"))
                vis.append({"geometry": geo, "origin": _origin(v), "color": col})
            self.links[l.get("name")] = vis
        for j in root.findall("joint"):
            lim = j.find("limit"); ax = j.find("axis"); t = j.get("type")
            lo = float(lim.get("lower", 0)) if lim is not None else 0.0; up = float(lim.get("upper", 0)) if lim is not None else 0.0
            if t == "continuous": lo, up = -math.pi, math.pi
            self.joints.append(Joint(j.get("name"), t, j.find("parent").get("link"), j.find("child").get("link"), _origin(j),
                                     [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0], lo, up,
                                     float(lim.get("velocity", 0) or 0) if lim is not None else 0.0))
        children = {j.child for j in self.joints}; roots = [n for n in self.links if n not in children]
        assert len(roots) == 1, f"{path}: one root link expected, found {roots}"
        self.root = roots[0]; self.movable = [j for j in self.joints if j.type in ("revolute", "continuous", "prismatic")]
        self.order = []; todo = [self.root]                      # joints in an order in which a parent comes before its children
        while todo:
            p = todo.pop(0)
            for j in self.joints:
                if j.parent == p: self.order.append(j); todo.append(j.child)

    def resolve(self, f):
        if f.startswith("package://"):
            rel = f[len("package://"):]; pkg, rest = rel.split("/", 1); d = self.dir; tried = [os.path.join(p, rel) for p in self.packages] + [os.path.join(p, rest) for p in self.packages if os.path.basename(p) == pkg]
            for _ in range(6): tried += [os.path.join(d, rel), os.path.join(d, rest)] if os.path.basename(d) == pkg else [os.path.join(d, rel)]; d = os.path.dirname(d)
            for t in tried:
                if os.path.exists(t): return t
            return rel
        if f.startswith("file://"): f = f[len("file://"):]
        return f if os.path.isabs(f) else os.path.join(self.dir, f)

    def fk(self, q, base=None):
        """q: {joint name: position} (missing: 0) -> {link name: 4x4 pose in the world}, with the root link at `base`."""
        T = {self.root: np.eye(4) if base is None else np.asarray(base, float)}
        for j in self.order: T[j.child] = T[j.parent] @ j.transform(float(q.get(j.name, 0.0)))
        return T
