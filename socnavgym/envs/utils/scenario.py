"""Hand-made scenarios for SocNavEnv_v2: a YAML file (or a dict) that fixes every item of a scene and how each person
behaves, instead of sampling them. Used for reproducible edge cases (a person stepping out of a doorway, someone standing
on the goal, a head-on encounter in a corridor ...).

    env.reset(seed=s, options={"scenario": "doorway.yaml"})   # this reset only (a path or a dict)
    env.set_scenario("doorway.yaml")                           # every following reset (None: back to random scenes)

The seed still matters: the learned humans sample their paths, ORCA / SFM add small noise, and unspecified values (a
person's policy, the goal's orientation) are drawn from the config. A scenario plus a seed reproduces exactly.
Everything after the scene is built is unchanged: observations, collisions, rewards (sn26 included).
Spontaneous group formation / dispersal and human-laptop interactions are switched off in scenario episodes.

Format (all lengths in metres, angles in radians, times in seconds; the map is centred on (0, 0)):

    name: doorway_step_out                 # optional
    description: ...                       # optional, free text
    map:
      size: [10, 6]                        # x, y extent (default: the config's largest map)
      boundary: true                       # four walls around the map (default true)
      walls: [[x1, y1, x2, y2], ...]       # extra wall segments; an optional 5th value is the thickness
      no_goal_zones: [[x, y, length, width, theta], ...]   # optional: where "wander" goals must not be drawn
    robot:
      x: -4
      y: 0
      theta: 0
      radius: 0.25                         # optional (default: the config's robot_radius, without the random margin)
      goal: {x: 4, y: 0, theta: 0, radius: 0.5, angle_threshold: 0.5}   # theta / radius / angle_threshold optional
    furniture:
      - {type: table, x: 0, y: 1.2, theta: 0, length: 1.5, width: 1.0}   # sizes optional (config defaults)
      - {type: chair, x: 0.9, y: 1.2, theta: 3.14}
      - {type: plant, x: 1, y: -1, radius: 0.3}
      - {type: laptop, x: 0, y: 1.2, theta: 0}
    people:
      - name: stander                      # optional (default p1, p2, ...); used by triggers and in reports
        x: 0
        y: -2
        theta: 1.57
        behaviour: static                  # never moves
      - name: walker
        x: 3
        y: 2
        behaviour: goals                   # walks to its goals with its policy, reacting to others
        policy: learned                    # learned | orca | sfm (default: drawn from the config's human_policy)
        speed: 1.1                         # preferred speed (default: as configured)
        avoids_robot: true                 # true | false | a probability (default: the config's prob_to_avoid_robot)
        goals: [[-3, 2], [-3, -2]]
        then: stop                         # after the last goal: stop (default) | loop | wander (random goals)
      - name: crosser
        x: 2
        y: -2
        behaviour: scripted                # follows the path exactly, ignoring everyone (also the robot)
        path: [[0.0, 2, -2], [1.5, 2, 0], [3.0, 2, 2]]   # (t, x, y), t counted from the start; linear in between
        then: stop                         # stop (default) | loop
        start: {robot_within: 2.5, delay: 0.3}           # trigger (optional; without it the person starts at t = 0)
    groups:                                # conversation groups: static people facing the group's centre
      - {centre: [1, 2], members: 3, radius: 0.6, phase: 0.0}
      - {members: [[0.5, 0], [1.1, 0]]}    # or explicit positions (they face their common centre)

Triggers (`start:`): the person waits (standing still, visible, an obstacle) until the condition holds, then starts its
behaviour once. One condition per trigger, plus an optional `delay` (seconds after the condition first holds):
    time: 2.0                                         simulated time since the start of the episode
    robot_within: 2.5                                 robot centre within 2.5 m of this person's centre
    robot_within: {distance: 2.5, of: [1.0, 1.0]}     ... of a point ({of: name}: of another person)
    robot_crosses: [[0, -2], [0, 2]]                  the robot's path crosses this segment
    after: {person: name, delay: 1.0}                 when another person starts (+ delay)
Triggers are checked once per step, after the robot moves. Every step's `info` has "scenario_events" (what fired) and,
when the robot touches a person, "scenario_collision" (names and behaviours: a collision with a scripted person may be
the script's fault rather than the robot's).
"""
import copy
import math
import os
import random

import numpy as np
import yaml

from socnavgym.envs.utils.chair import Chair
from socnavgym.envs.utils.human import Human
from socnavgym.envs.utils.laptop import Laptop
from socnavgym.envs.utils.plant import Plant
from socnavgym.envs.utils.robot import Robot
from socnavgym.envs.utils.table import Table
from socnavgym.envs.utils.wall import Wall

BEHAVIOURS = ("static", "goals", "scripted")
POLICIES = ("learned", "orca", "sfm")
CONDITIONS = ("time", "robot_within", "robot_crosses", "after")


class ScenarioError(ValueError):
    """The scenario cannot be built as written (bad field, overlapping items, more items than the config allows)."""


# ---------------------------------------------------------------------------------------------------- reading / checking
def read(spec):
    """A scenario as a dict: from a path (YAML), or a copy of the given dict."""
    if isinstance(spec, (str, os.PathLike)):
        with open(spec) as f: d = yaml.safe_load(f)
        if not isinstance(d, dict): raise ScenarioError(f"{spec}: not a scenario (expected a mapping)")
        d.setdefault("name", os.path.splitext(os.path.basename(str(spec)))[0])
        return d
    if isinstance(spec, dict): return copy.deepcopy(spec)
    raise ScenarioError(f"a scenario is a path or a dict, not {type(spec).__name__}")


def _num(v, what):
    try: x = float(v)
    except (TypeError, ValueError): raise ScenarioError(f"{what}: expected a number, got {v!r}")
    if not math.isfinite(x): raise ScenarioError(f"{what}: not finite")
    return x


def _xy(v, what):
    if not isinstance(v, (list, tuple)) or len(v) != 2: raise ScenarioError(f"{what}: expected [x, y], got {v!r}")
    return (_num(v[0], what), _num(v[1], what))


def _seg_cross(p, q, a, b):
    """Do segments p-q and a-b intersect (touching counts)?"""
    def orient(u, v, w): return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    def on(u, v, w): return min(u[0], v[0]) - 1e-12 <= w[0] <= max(u[0], v[0]) + 1e-12 and min(u[1], v[1]) - 1e-12 <= w[1] <= max(u[1], v[1]) + 1e-12
    d1, d2, d3, d4 = orient(a, b, p), orient(a, b, q), orient(p, q, a), orient(p, q, b)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0: return True
    return (d1 == 0 and on(a, b, p)) or (d2 == 0 and on(a, b, q)) or (d3 == 0 and on(p, q, a)) or (d4 == 0 and on(p, q, b))


class Person:
    """One person of a scenario and its state during an episode."""
    def __init__(self, d, i):
        if not isinstance(d, dict): raise ScenarioError(f"people[{i}]: expected a mapping")
        self.name = str(d.get("name", f"p{i + 1}"))
        w = f"person {self.name}"
        unknown = set(d) - {"name", "x", "y", "theta", "behaviour", "policy", "speed", "avoids_robot", "goals", "then", "path", "start", "group"}
        if unknown: raise ScenarioError(f"{w}: unknown field(s) {sorted(unknown)}")
        self.behaviour = d.get("behaviour", "goals" if "goals" in d else "scripted" if "path" in d else "static")
        if self.behaviour not in BEHAVIOURS: raise ScenarioError(f"{w}: behaviour must be one of {BEHAVIOURS}")
        self.group = d.get("group")
        self.policy = d.get("policy")
        if self.policy is not None and self.policy not in POLICIES: raise ScenarioError(f"{w}: policy must be one of {POLICIES}")
        self.speed = None if d.get("speed") is None else _num(d["speed"], f"{w}: speed")
        if self.speed is not None and self.speed < 0: raise ScenarioError(f"{w}: speed must be >= 0")
        av = d.get("avoids_robot")
        self.avoid_prob = None if av is None else (1.0 if av is True else 0.0 if av is False else _num(av, f"{w}: avoids_robot"))
        self.goals, self.path = [], None
        if self.behaviour == "goals":
            g = d.get("goals")
            if not g: raise ScenarioError(f"{w}: behaviour goals needs a non-empty goals list")
            self.goals = [_xy(p, f"{w}: goals[{k}]") for k, p in enumerate(g)]
            self.then = d.get("then", "stop")
            if self.then not in ("stop", "loop", "wander"): raise ScenarioError(f"{w}: then must be stop, loop or wander")
        elif self.behaviour == "scripted":
            p = d.get("path")
            if not p or len(p) < 1: raise ScenarioError(f"{w}: behaviour scripted needs a path of [t, x, y] points")
            pts = []
            for k, q in enumerate(p):
                if not isinstance(q, (list, tuple)) or len(q) != 3: raise ScenarioError(f"{w}: path[{k}] must be [t, x, y]")
                pts.append([_num(v, f"{w}: path[{k}]") for v in q])
            self.path = np.array(pts, float)
            if np.any(np.diff(self.path[:, 0]) < 0): raise ScenarioError(f"{w}: path times must not decrease")
            self.then = d.get("then", "stop")
            if self.then not in ("stop", "loop"): raise ScenarioError(f"{w}: then must be stop or loop for a scripted person")
        else:
            for k in ("goals", "path", "start", "policy", "speed"):
                if k in d: raise ScenarioError(f"{w}: '{k}' makes no sense for a static person")
        if "x" in d or "y" in d: self.x0, self.y0 = _num(d.get("x"), f"{w}: x"), _num(d.get("y"), f"{w}: y")
        elif self.path is not None: self.x0, self.y0 = float(self.path[0, 1]), float(self.path[0, 2])
        else: raise ScenarioError(f"{w}: x and y are required")
        if "theta" in d: self.th0 = _num(d["theta"], f"{w}: theta")
        else:                                           # facing where it will go first
            tgt = self.goals[0] if self.goals else (self.path[min(1, len(self.path) - 1), 1:] if self.path is not None else None)
            self.th0 = 0.0 if tgt is None or math.hypot(tgt[0] - self.x0, tgt[1] - self.y0) < 1e-9 else math.atan2(tgt[1] - self.y0, tgt[0] - self.x0)
        if self.path is not None and math.hypot(self.path[0, 1] - self.x0, self.path[0, 2] - self.y0) > 1e-6:
            raise ScenarioError(f"{w}: the path must start where the person stands ({self.x0:g}, {self.y0:g})")
        self.start_spec = copy.deepcopy(d.get("start"))
        self.start = None if d.get("start") is None else Trigger(d["start"], w)
        if self.start is not None and self.behaviour == "static": raise ScenarioError(f"{w}: a static person has nothing to start")

    def at(self, t):
        """Scripted position at time t since the start (linear interpolation, held before / after the path)."""
        P = self.path
        if len(P) == 1 or t <= P[0, 0]: return float(P[0, 1]), float(P[0, 2])
        if self.then == "loop" and P[-1, 0] > P[0, 0]: t = P[0, 0] + (t - P[0, 0]) % (P[-1, 0] - P[0, 0])
        if t >= P[-1, 0]: return float(P[-1, 1]), float(P[-1, 2])
        k = int(np.searchsorted(P[:, 0], t, side="right")) - 1
        t0, t1 = P[k, 0], P[k + 1, 0]; u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return float(P[k, 1] + u * (P[k + 1, 1] - P[k, 1])), float(P[k, 2] + u * (P[k + 1, 2] - P[k, 2]))


class Trigger:
    def __init__(self, d, w):
        if not isinstance(d, dict): raise ScenarioError(f"{w}: start must be a mapping, e.g. {{robot_within: 2.0}}")
        keys = [k for k in d if k in CONDITIONS]
        unknown = set(d) - set(CONDITIONS) - {"delay"}
        if unknown: raise ScenarioError(f"{w}: start: unknown field(s) {sorted(unknown)}")
        if len(keys) != 1: raise ScenarioError(f"{w}: start needs exactly one of {CONDITIONS}")
        self.kind = keys[0]; v = d[self.kind]; self.delay = _num(d.get("delay", 0.0), f"{w}: start: delay")
        if self.delay < 0: raise ScenarioError(f"{w}: start: delay must be >= 0")
        self.point = self.person = None
        if self.kind == "time": self.value = _num(v, f"{w}: start: time")
        elif self.kind == "robot_within":
            if isinstance(v, dict):
                self.value = _num(v.get("distance"), f"{w}: start: robot_within: distance")
                of = v.get("of")
                if isinstance(of, str): self.person = of
                elif of is not None: self.point = _xy(of, f"{w}: start: robot_within: of")
            else: self.value = _num(v, f"{w}: start: robot_within")
        elif self.kind == "robot_crosses":
            if not isinstance(v, (list, tuple)) or len(v) != 2: raise ScenarioError(f"{w}: start: robot_crosses must be [[x1, y1], [x2, y2]]")
            self.value = (_xy(v[0], f"{w}: start: robot_crosses"), _xy(v[1], f"{w}: start: robot_crosses"))
        else:
            if isinstance(v, str): v = {"person": v}
            if not isinstance(v, dict) or "person" not in v: raise ScenarioError(f"{w}: start: after must be {{person: name, delay: s}}")
            self.person = str(v["person"]); self.value = _num(v.get("delay", 0.0), f"{w}: start: after: delay")

    def describe(self):
        if self.kind == "time": s = f"at t = {self.value:g} s"
        elif self.kind == "robot_within": s = f"robot within {self.value:g} m of " + (self.person or (f"({self.point[0]:g}, {self.point[1]:g})" if self.point else "it"))
        elif self.kind == "robot_crosses": s = "robot crosses a line"
        else: s = f"{self.value:g} s after {self.person} starts"
        return s + (f" (+{self.delay:g} s)" if self.delay else "")


# ---------------------------------------------------------------------------------------------------- the scenario
class Scenario:
    def __init__(self, spec):
        d = read(spec); self.spec = d
        unknown = set(d) - {"name", "description", "config", "seed", "map", "robot", "furniture", "people", "groups", "source"}
        if unknown: raise ScenarioError(f"unknown top-level field(s) {sorted(unknown)}")
        self.name = str(d.get("name", "scenario"))
        m = d.get("map") or {}
        if not isinstance(m, dict): raise ScenarioError("map: expected a mapping")
        self.size = None if m.get("size") is None else _xy(m["size"], "map: size")
        self.boundary = bool(m.get("boundary", True))
        self.walls = []
        for k, w in enumerate(m.get("walls") or []):
            if not isinstance(w, (list, tuple)) or len(w) not in (4, 5): raise ScenarioError(f"map: walls[{k}] must be [x1, y1, x2, y2] or [x1, y1, x2, y2, thickness]")
            w = [_num(v, f"map: walls[{k}]") for v in w]
            if math.hypot(w[2] - w[0], w[3] - w[1]) < 1e-6: raise ScenarioError(f"map: walls[{k}] has zero length")
            self.walls.append(w)
        self.zones = []
        for k, z in enumerate(m.get("no_goal_zones") or []):
            if not isinstance(z, (list, tuple)) or len(z) not in (4, 5): raise ScenarioError(f"map: no_goal_zones[{k}] must be [x, y, length, width] or [x, y, length, width, theta]")
            self.zones.append([_num(v, f"map: no_goal_zones[{k}]") for v in z] + ([0.0] if len(z) == 4 else []))
        r = d.get("robot")
        if not isinstance(r, dict): raise ScenarioError("robot: required (x, y, theta and goal)")
        self.robot = dict(x=_num(r.get("x"), "robot: x"), y=_num(r.get("y"), "robot: y"), theta=_num(r.get("theta", 0.0), "robot: theta"),
                          radius=None if r.get("radius") is None else _num(r["radius"], "robot: radius"))
        g = r.get("goal")
        if not isinstance(g, dict): raise ScenarioError("robot: goal: required ({x: .., y: ..})")
        self.goal = dict(x=_num(g.get("x"), "robot: goal: x"), y=_num(g.get("y"), "robot: goal: y"),
                         theta=None if g.get("theta") is None else _num(g["theta"], "robot: goal: theta"),
                         radius=None if g.get("radius") is None else _num(g["radius"], "robot: goal: radius"),
                         angle_threshold=None if g.get("angle_threshold") is None else _num(g["angle_threshold"], "robot: goal: angle_threshold"))
        self.furniture = []
        for k, f in enumerate(d.get("furniture") or []):
            if not isinstance(f, dict) or f.get("type") not in ("table", "chair", "plant", "laptop"):
                raise ScenarioError(f"furniture[{k}]: type must be table, chair, plant or laptop")
            unknown = set(f) - {"type", "x", "y", "theta", "length", "width", "radius"}
            if unknown: raise ScenarioError(f"furniture[{k}]: unknown field(s) {sorted(unknown)}")
            self.furniture.append(dict(type=f["type"], x=_num(f.get("x"), f"furniture[{k}]: x"), y=_num(f.get("y"), f"furniture[{k}]: y"),
                                       theta=_num(f.get("theta", 0.0), f"furniture[{k}]: theta"),
                                       **{q: _num(f[q], f"furniture[{k}]: {q}") for q in ("length", "width", "radius") if f.get(q) is not None}))
        people = list(d.get("people") or [])
        for k, gr in enumerate(d.get("groups") or []):                     # groups -> static people facing the centre
            if not isinstance(gr, dict): raise ScenarioError(f"groups[{k}]: expected a mapping")
            mem = gr.get("members")
            if isinstance(mem, (list, tuple)):
                P = [_xy(p, f"groups[{k}]: members") for p in mem]
                c = _xy(gr["centre"], f"groups[{k}]: centre") if gr.get("centre") is not None else (float(np.mean([p[0] for p in P])), float(np.mean([p[1] for p in P])))
            else:
                n = int(_num(mem, f"groups[{k}]: members")); c = _xy(gr.get("centre"), f"groups[{k}]: centre")
                rad = _num(gr.get("radius", 0.6), f"groups[{k}]: radius"); ph = _num(gr.get("phase", 0.0), f"groups[{k}]: phase")
                P = [(c[0] + rad * math.cos(ph + 2 * math.pi * j / n), c[1] + rad * math.sin(ph + 2 * math.pi * j / n)) for j in range(n)]
            for j, p in enumerate(P):
                people.append(dict(name=f"{gr.get('name', f'group{k + 1}')}_{j + 1}", x=p[0], y=p[1], theta=math.atan2(c[1] - p[1], c[0] - p[0]),
                                   behaviour="static", group=k + 1))
        self.people = [Person(p, i) for i, p in enumerate(people)]
        names = [p.name for p in self.people]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup: raise ScenarioError(f"people: duplicate name(s) {dup}")
        for p in self.people:
            if p.start is not None and p.start.person is not None and p.start.person not in names:
                raise ScenarioError(f"person {p.name}: start refers to unknown person {p.start.person!r}")
            if p.start is not None and p.start.person == p.name and p.start.kind == "after": raise ScenarioError(f"person {p.name}: starts after itself")
        self.by_name = {p.name: p for p in self.people}

    # ------------------------------------------------------------------------------------------------ building the scene
    def build(self, env):
        """Fills env with this scene (replaces the random sampling of try_reset). Raises ScenarioError if impossible."""
        self.env = env
        X, Y = self.size if self.size is not None else (float(env.MAX_MAP_X), float(env.MAX_MAP_Y))
        if X <= 0 or Y <= 0: raise ScenarioError("map: size must be positive")
        env.MAP_X, env.MAP_Y = X, Y
        env.shape = "no-walls" if not self.boundary and not self.walls else "custom"
        env.RESOLUTION_X, env.RESOLUTION_Y = env.CANVAS_X, env.CANVAS_Y      # the drawing has the configured size whatever the map
        r, g = self.robot, self.goal
        env.ROBOT_RADIUS = r["radius"] if r["radius"] is not None and getattr(env, "ROBOT_SHAPE", "circle") == "circle" else env.INITIAL_ROBOT_RADIUS
        if getattr(env, "ROBOT_SHAPE", "circle") == "circle": env.ROBOT_LENGTH = env.ROBOT_WIDTH = 2 * env.ROBOT_RADIUS   # (a rectangular robot keeps the config's size)
        env.GOAL_RADIUS = g["radius"] if g["radius"] is not None else env.INITIAL_GOAL_RADIUS
        env.GOAL_THRESHOLD = env.GOAL_RADIUS
        if g["angle_threshold"] is not None: env.GOAL_ORIENTATION_THRESHOLD = g["angle_threshold"]   # else: as drawn from the config

        # walls (boundary as in the env's rectangular rooms; segments centred on their line)
        th = env.WALL_THICKNESS
        walls = []
        if self.boundary:
            walls += [Wall(None, X/2 - th/2, 0, -np.pi/2, Y, th), Wall(None, 0, -Y/2 + th/2, -np.pi, X, th),
                      Wall(None, -X/2 + th/2, 0, np.pi/2, Y, th), Wall(None, 0, Y/2 - th/2, 0, X, th)]
        for w in self.walls:
            x1, y1, x2, y2 = w[:4]
            walls.append(Wall(None, (x1 + x2) / 2, (y1 + y2) / 2, math.atan2(y2 - y1, x2 - x1), math.hypot(x2 - x1, y2 - y1), w[4] if len(w) > 4 else th))
        for w in walls: env.walls.append(w); env.objects.append(w)
        for z in self.zones:                                                  # invisible: only keeps random goals out
            env.objects.append(Wall(-1, z[0], z[1], z[4], z[2], z[3]))

        env.robot = Robot(id=0, x=r["x"], y=r["y"], theta=r["theta"], radius=env.ROBOT_RADIUS, goal_x=None, goal_y=None, goal_a=None, type=env.ROBOT_TYPE,
                          shape=getattr(env, "ROBOT_SHAPE", "circle"), length=getattr(env, "ROBOT_LENGTH", None), width=getattr(env, "ROBOT_WIDTH", None))
        env.objects.append(env.robot)
        env.robot_orca = copy.deepcopy(env.robot)
        env.has_orca_robot_reached_goal = False; env.has_orca_robot_collided = False
        env.orca_robot_reach_time = None; env.orca_robot_path_length = 0

        # people
        need_learned = False
        for p in self.people:
            pol = p.policy
            if p.behaviour == "goals" and pol is None:
                pol = random.choices(list(env.HUMAN_POLICY_WEIGHTS), weights=list(env.HUMAN_POLICY_WEIGHTS.values()))[0]
            p.real_policy = pol or "orca"
            need_learned |= p.behaviour == "goals" and p.real_policy == "learned"
            static = p.behaviour == "static"
            h = Human(id=env.id, x=p.x0, y=p.y0, theta=p.th0, width=env.HUMAN_DIAMETER, speed=0.0, goal_radius=env.HUMAN_GOAL_RADIUS,
                      goal_x=p.x0, goal_y=p.y0, policy=p.real_policy if static else "wait", fov=env.HUMAN_FOV,
                      prob_to_avoid_robot=env.PROB_TO_AVOID_ROBOT if p.avoid_prob is None else p.avoid_prob,
                      type="static" if static else "dynamic", pos_noise_std=env.HUMAN_POS_NOISE_STD, angle_noise_std=env.HUMAN_ANGLE_NOISE_STD)
            h.scn = p; p.h = h; env.id += 1
            p.started = p.start is None and not static; p.fire_at = None if p.start is not None else 0.0
            p.goal_idx = 0; p.finished = False; p.t0 = 0.0
            (env.static_humans if static else env.dynamic_humans).append(h); env.objects.append(h)
            env.goals[h.id] = Plant(id=None, x=p.x0, y=p.y0, radius=env.HUMAN_GOAL_RADIUS)
        # furniture
        lists = dict(table=(env.tables, Table), chair=(env.chairs, Chair), plant=(env.plants, Plant), laptop=(env.laptops, Laptop))
        for f in self.furniture:
            lst, cls = lists[f["type"]]
            if f["type"] == "plant":
                o = Plant(id=env.id, x=f["x"], y=f["y"], radius=f.get("radius", env.PLANT_RADIUS))
            else:
                L = dict(table=(env.TABLE_LENGTH, env.TABLE_WIDTH), chair=(env.CHAIR_LENGTH, env.CHAIR_WIDTH), laptop=(env.LAPTOP_LENGTH, env.LAPTOP_WIDTH))[f["type"]]
                o = cls(id=env.id, x=f["x"], y=f["y"], theta=f["theta"], width=f.get("width", L[1]), length=f.get("length", L[0]))
            o.scn_label = f["type"]; lst.append(o); env.objects.append(o); env.id += 1
        for w in env.walls: w.id = env.id; env.id += 1

        # counts (the observation is padded to the config's maxima: a scene cannot exceed them)
        env.NUMBER_OF_STATIC_HUMANS, env.NUMBER_OF_DYNAMIC_HUMANS = len(env.static_humans), len(env.dynamic_humans)
        env.NUMBER_OF_PLANTS, env.NUMBER_OF_TABLES, env.NUMBER_OF_CHAIRS, env.NUMBER_OF_LAPTOPS = len(env.plants), len(env.tables), len(env.chairs), len(env.laptops)
        for k in ("NUMBER_OF_H_H_DYNAMIC_INTERACTIONS", "NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING", "NUMBER_OF_H_H_STATIC_INTERACTIONS",
                  "NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING", "NUMBER_OF_H_L_INTERACTIONS", "NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING", "TOTAL_H_L_INTERACTIONS"):
            setattr(env, k, 0)
        env.humans_in_h_h_dynamic_interactions, env.humans_in_h_h_static_interactions = [], []
        env.humans_in_h_h_dynamic_interactions_non_dispersing, env.humans_in_h_h_static_interactions_non_dispersing = [], []
        env.total_humans = len(self.people)
        self._check(env)

        # goals
        env.goals[env.robot.id] = Plant(id=None, x=g["x"], y=g["y"], radius=env.GOAL_RADIUS)
        env.robot.goal_x, env.robot.goal_y = g["x"], g["y"]
        env.robot.goal_a = g["theta"] if g["theta"] is not None else random.uniform(-np.pi, np.pi)
        env.robot_orca.goal_x, env.robot_orca.goal_y, env.robot_orca.goal_a = env.robot.goal_x, env.robot.goal_y, env.robot.goal_a
        for p in self.people:
            if p.behaviour == "goals" and p.started: self._go(p, 0)

        if need_learned and env.LEARNED_HUMANS is None:                      # the config has no learned humans: add the controller
            from socnavgym.envs.utils.learned_humans import LearnedHumanController
            env.LEARNED_HUMANS = LearnedHumanController(speed_range=getattr(env, "LEARNED_SPEED_RANGE", None) or env.HUMAN_SPEED_RANGE or (0.6, 1.4))
        self.events, self.new_events = [], []
        self.prev_robot = (env.robot.x, env.robot.y)

    def _check(self, env):
        lim = [("people", len(self.people), env.MAX_HUMANS), ("tables", len(env.tables), env.MAX_TABLES), ("chairs", len(env.chairs), env.MAX_CHAIRS),
               ("plants", len(env.plants), env.MAX_PLANTS), ("laptops", len(env.laptops), env.MAX_LAPTOPS)]
        nw = env.observation_space["walls"].shape[0] // 8
        segs = sum(1 if env.WALL_SEGMENT_SIZE <= 0 else int(w.length // env.WALL_SEGMENT_SIZE) + (w.length % env.WALL_SEGMENT_SIZE != 0) for w in env.walls)
        lim.append(("wall segments", segs, nw))
        over = [f"{n} {k} (the config allows {m})" for k, n, m in lim if n > m]
        if over: raise ScenarioError("too many items for the config's observation size: " + "; ".join(over) + ". Use a config with larger max_* values.")
        r = env.robot; X, Y = env.MAP_X / 2, env.MAP_Y / 2
        # (people may stand outside the map, as the env's own groups sometimes do; the robot and its goal may not)
        for nm, x, y in [("robot", r.x, r.y), ("robot goal", self.goal["x"], self.goal["y"])]:
            if not (-X <= x <= X and -Y <= y <= Y): raise ScenarioError(f"{nm} at ({x:g}, {y:g}) is outside the map ({2 * X:g} x {2 * Y:g} m centred on 0, 0)")
        solid = [(o, f"{getattr(o, 'scn_label', o.name)} at ({o.x:g}, {o.y:g})") for o in env.walls + env.tables + env.chairs + env.plants + env.laptops]
        for o, nm in solid:
            if r.collides(o): raise ScenarioError(f"the robot overlaps the {nm}")
        clear = env.ROBOT_RADIUS + env.HUMAN_DIAMETER / 2 + env.SPAWN_CLEARANCE
        for p in self.people:
            if math.hypot(p.x0 - r.x, p.y0 - r.y) < clear:
                raise ScenarioError(f"person {p.name} is within {clear:.2f} m of the robot (it could touch the robot by turning)")
            for o, nm in solid:
                if o.name != "laptop" and p.h.collides(o): raise ScenarioError(f"person {p.name} overlaps the {nm}")
        for i, p in enumerate(self.people):
            for q in self.people[i + 1:]:
                if math.hypot(p.x0 - q.x0, p.y0 - q.y0) < 0.25: raise ScenarioError(f"persons {p.name} and {q.name} are on top of each other")

    # ------------------------------------------------------------------------------------------------ during an episode
    def _go(self, p, k):
        p.goal_idx = k; x, y = p.goals[k]; h = p.h
        h.set_goal(x, y); self.env.goals[h.id] = Plant(id=None, x=x, y=y, radius=self.env.HUMAN_GOAL_RADIUS)

    def _start(self, p, now):
        env, h = self.env, p.h; p.started = True; p.t0 = now
        if p.behaviour == "goals":
            h.policy = p.real_policy; self._go(p, 0)
            if p.speed is not None: h.scenario_speed = p.speed; h.pref_speed = p.speed
            else: h.scenario_speed = None
        self.new_events.append((round(now, 3), p.name, "starts" + ("" if p.start is None else f" ({p.start.describe()})")))

    def after_reset(self, env):
        """After the env's own per-human set-up (preferred speeds), before the learned controller is reset."""
        for p in self.people:
            h = p.h
            if p.behaviour == "goals" and p.started:
                h.policy = p.real_policy
                if p.speed is not None: h.scenario_speed = p.speed; h.pref_speed = p.speed
            if p.behaviour == "goals" and not p.started or p.behaviour == "scripted": h.scenario_speed = 0.0
        self.new_events = [(0.0, p.name, "starts") for p in self.people if p.started]

    def _condition(self, p, now):
        env, t = self.env, p.start; r = env.robot
        if t.kind == "time": return now >= t.value - 1e-9
        if t.kind == "robot_within":
            if t.point is not None: cx, cy = t.point
            elif t.person is not None: q = self.by_name[t.person].h; cx, cy = q.x, q.y
            else: cx, cy = p.h.x, p.h.y
            return math.hypot(r.x - cx, r.y - cy) <= t.value
        if t.kind == "robot_crosses": return _seg_cross(self.prev_robot, (r.x, r.y), *t.value)
        q = self.by_name[t.person]
        return q.started and now >= q.t0 + t.value - 1e-9

    def before_humans(self, env):
        """Called in step() after the robot has moved and before the humans' velocities are computed."""
        dt = env.TIMESTEP; now = (env.ticks + 1) * dt; self.now = now
        for _ in range(len(self.people) + 1):                                # chained 'after' triggers may fire in the same step
            fired = False
            for p in self.people:
                if p.started or p.behaviour == "static": continue
                if p.fire_at is None and self._condition(p, now):
                    p.fire_at = now + p.start.delay
                    if p.start.delay: self.new_events.append((round(now, 3), p.name, f"trigger: {p.start.describe()}"))
                if p.fire_at is not None and now >= p.fire_at - 1e-9: self._start(p, now); fired = True
            if not fired: break
        self.prev_robot = (env.robot.x, env.robot.y)
        for p in self.people:
            h = p.h
            if p.behaviour == "static": continue
            if p.behaviour == "scripted" and p.started:
                x, y = p.at(now - p.t0); ax, ay = p.at(now - p.t0 + 1.0)
                h.scripted_target = (x, y); h.policy = "scripted"
                h.goal_x, h.goal_y = ax, ay                                    # for the others' ORCA: where it is heading
                h.scenario_speed = math.hypot(x - h.x, y - h.y) / dt
            elif not p.started or p.finished:
                h.scripted_target = (h.x, h.y); h.policy = "wait"; h.goal_x, h.goal_y = h.x, h.y; h.scenario_speed = 0.0

    def after_goals(self, env):
        """Called in step() where the env would draw new goals for humans that reached theirs."""
        HX, HY = env.MAP_X / 2. - env.MARGIN, env.MAP_Y / 2. - env.MARGIN
        for p in self.people:
            if p.behaviour != "goals" or not p.started or p.finished or not p.h.has_reached_goal(): continue
            if p.goal_idx + 1 < len(p.goals): self._go(p, p.goal_idx + 1)
            elif p.then == "loop": self._go(p, 0)
            elif p.then == "wander":
                o = env.sample_goal(env.HUMAN_GOAL_RADIUS, HX, HY)
                if o is not None: p.h.set_goal(o.x, o.y); env.goals[p.h.id] = o
            else:
                p.finished = True; p.h.policy = "wait"; p.h.speed = 0.0
                self.new_events.append((round((env.ticks + 1) * env.TIMESTEP, 3), p.name, "reached its last goal: stops"))

    def step_info(self, env, info):
        info["scenario_events"] = self.new_events; self.events += self.new_events; self.new_events = []
        if info.get("COLLISION_HUMAN"):
            info["scenario_collision"] = [(p.name, p.behaviour) for p in self.people if p.h.collides(env.robot)]

    def labels(self):
        """(name, behaviour, state, x, y) of every person, for viewers."""
        out = []
        for p in self.people:
            done = p.finished or (p.behaviour == "scripted" and p.started and p.then == "stop" and getattr(self, "now", 0.0) - p.t0 >= p.path[-1, 0])
            st = "static" if p.behaviour == "static" else "done" if done else (p.behaviour if p.started else "waiting")
            out.append((p.name, p.behaviour, st, float(p.h.x), float(p.h.y)))
        return out


# ---------------------------------------------------------------------------------------------------- exporting a scene
def export(env, name="exported", config=None, seed=None, description=None):
    """The env's current scene as a scenario dict (call it right after reset() to capture an episode's start).
    Exact for the robot, goal, walls, furniture and every person's position, heading, policy, speed and avoidance;
    approximations: people keep only their current goal and then wander (random goals), moving groups become individuals
    walking to the group's goal, people at laptops become static people, and positions are rounded to 1 mm."""
    R = lambda v, n=3: round(float(v), n)
    scn = getattr(env, "_scenario", None)
    d = {"name": name}
    if description: d["description"] = description
    if config: d["config"] = config
    if seed is not None: d["seed"] = int(seed)
    walls = []
    for w in env.walls:
        c, s = math.cos(w.orientation), math.sin(w.orientation)
        seg = [R(w.x - c * w.length / 2), R(w.y - s * w.length / 2), R(w.x + c * w.length / 2), R(w.y + s * w.length / 2)]
        if abs(w.thickness - env.WALL_THICKNESS) > 1e-9: seg.append(R(w.thickness))
        walls.append(seg)
    zones = [[R(o.x), R(o.y), R(o.length), R(o.thickness), R(o.orientation, 4)] for o in env.objects if o.name == "wall" and o.id == -1]
    d["map"] = {"size": [R(env.MAP_X), R(env.MAP_Y)], "boundary": False, "walls": walls}
    if zones: d["map"]["no_goal_zones"] = zones
    r = env.robot
    d["robot"] = {"x": R(r.x), "y": R(r.y), "theta": R(r.orientation, 4), "radius": R(env.ROBOT_RADIUS),
                  "goal": {"x": R(r.goal_x), "y": R(r.goal_y), "theta": R(r.goal_a, 4), "radius": R(env.GOAL_RADIUS),
                           "angle_threshold": R(env.GOAL_ORIENTATION_THRESHOLD, 4)}}
    furn = []
    for t in env.tables: furn.append({"type": "table", "x": R(t.x), "y": R(t.y), "theta": R(t.orientation, 4), "length": R(t.length), "width": R(t.width)})
    for t in env.chairs: furn.append({"type": "chair", "x": R(t.x), "y": R(t.y), "theta": R(t.orientation, 4), "length": R(t.length), "width": R(t.width)})
    for t in env.plants: furn.append({"type": "plant", "x": R(t.x), "y": R(t.y), "radius": R(t.radius)})
    for t in list(env.laptops) + [i.laptop for i in env.h_l_interactions]:
        furn.append({"type": "laptop", "x": R(t.x), "y": R(t.y), "theta": R(t.orientation, 4), "length": R(t.length), "width": R(t.width)})
    if furn: d["furniture"] = furn
    L = env.LEARNED_HUMANS
    people = []

    def walker(h, nm, goal, policy):
        e = {"name": nm, "x": R(h.x), "y": R(h.y), "theta": R(h.orientation, 4), "behaviour": "goals", "policy": policy}
        if L is not None and policy == "learned" and h.id in getattr(L, "pref_speed", {}): e["speed"] = R(L.pref_speed[h.id])
        elif getattr(h, "pref_speed", None) is not None: e["speed"] = R(h.pref_speed)
        else: e["speed"] = R(env.human_speed(h))
        if L is not None and policy == "learned" and h.id in getattr(L, "sees_robot", {}): e["avoids_robot"] = bool(L.sees_robot[h.id])
        else: e["avoids_robot"] = R(h.prob_to_avoid_robot)
        e["goals"] = [[R(goal[0]), R(goal[1])]]; e["then"] = "wander"
        return e

    k = 0
    for h in env.static_humans:
        k += 1; nm = h.scn.name if getattr(h, "scn", None) is not None else f"p{k}"
        people.append({"name": nm, "x": R(h.x), "y": R(h.y), "theta": R(h.orientation, 4), "behaviour": "static"})
    for h in env.dynamic_humans:
        k += 1
        p = getattr(h, "scn", None)
        if p is not None and scn is not None:                                 # a scenario person: keep its own description
            people.append(_person_spec(p, h)); continue
        pol = h.policy if h.policy in POLICIES else "orca"
        people.append(walker(h, f"p{k}", (h.goal_x, h.goal_y), pol))
    for gi, i in enumerate(env.static_interactions):
        for j, h in enumerate(i.humans):
            people.append({"name": f"group{gi + 1}_{j + 1}", "x": R(h.x), "y": R(h.y), "theta": R(h.orientation, 4), "behaviour": "static"})
    for gi, i in enumerate(env.moving_interactions):
        for j, h in enumerate(i.humans):
            people.append(walker(h, f"moving_group{gi + 1}_{j + 1}", (i.goal_x, i.goal_y), "orca"))
    for j, i in enumerate(env.h_l_interactions):
        h = i.human; people.append({"name": f"at_laptop{j + 1}", "x": R(h.x), "y": R(h.y), "theta": R(h.orientation, 4), "behaviour": "static"})
    if people: d["people"] = people
    return d


def _person_spec(p, h):
    e = {"name": p.name, "x": round(p.x0, 3), "y": round(p.y0, 3), "theta": round(p.th0, 4), "behaviour": p.behaviour}
    if p.behaviour == "goals":
        e["policy"] = p.real_policy
        if p.speed is not None: e["speed"] = p.speed
        if p.avoid_prob is not None: e["avoids_robot"] = p.avoid_prob
        e["goals"] = [list(g) for g in p.goals]; e["then"] = p.then
    elif p.behaviour == "scripted":
        e["path"] = [[float(v) for v in q] for q in p.path]; e["then"] = p.then
    if p.start_spec is not None: e["start"] = copy.deepcopy(p.start_spec)
    return e


class _Dumper(yaml.SafeDumper):
    pass


def _list_repr(dumper, data):          # short lists of numbers in one line: [x, y]
    flow = all(isinstance(v, (int, float, bool)) for v in data) or all(isinstance(v, list) and all(isinstance(u, (int, float)) for u in v) and len(v) <= 5 for v in data) and len(data) <= 12
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _list_repr)


def dump(d, path=None, header=None):
    """Writes a scenario dict as YAML (numbers lists on one line). Returns the text."""
    txt = yaml.dump(d, Dumper=_Dumper, sort_keys=False, width=140, allow_unicode=True)
    if header: txt = "".join(f"# {l}\n" for l in header.splitlines()) + txt
    if path is not None:
        with open(path, "w") as f: f.write(txt)
    return txt
