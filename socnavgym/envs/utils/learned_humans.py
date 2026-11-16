"""Data-driven human motion for SocNavGym ("human_policy: learned").

Dynamic humans are moved by a generative trajectory model trained on real human motion (THÖR-MAGNI, LocoVR, ETH/UCY;
see /workspace/humanmotion). The simulator gives the model what a real person knows -- their destination (the human's
goal) and their preferred walking speed -- plus what they see: their own recent motion, nearby people (and the robot,
for humans that avoid it), and a local occupancy patch of the static scene. The robot's own forecaster, if any, is a
separate model that never receives destinations or speeds.

Every `replan` steps, one batched call samples a 4.8 s future for every learned human; the first `replan` positions are
executed. Self-contained on purpose (no dependency on the training code): the architecture below must match the
checkpoint (`MotionModel` in humanmotion/forecaster2.py).
"""
import math
import os
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn

T_OBS, T_PRED, K_NB, NB_RADIUS = 32, 48, 12, 8.0
BASE_RES, PATCH, PATCH_RES, PATCH_BACK = 0.05, 64, 0.10, 1.6
SCALE = 3.0
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_human_models", "motion_sim.pt")


# ------------------------------------------------------------------ static scene -> occupancy raster
def _rect(x, y, th, length, width):
    c, s = np.cos(th), np.sin(th)
    ax, ay, bx, by = length / 2 * c, length / 2 * s, -width / 2 * s, width / 2 * c
    return np.array([[x + ax + bx, y + ay + by], [x - ax + bx, y - ay + by], [x - ax - bx, y - ay - by], [x + ax - bx, y + ay - by]])


class SceneRaster:
    """Occupancy of the static scene (walls, furniture, outside the map, L-room cut-out) at 5 cm; channel 1 unused."""
    FINE = 0.01

    def __init__(self, env, margin=1.0):
        hx, hy = env.MAP_X / 2 + margin, env.MAP_Y / 2 + margin
        W, H = int(np.ceil(2 * hx / self.FINE)), int(np.ceil(2 * hy / self.FINE))
        occ = np.ones((H, W), np.uint8)
        px = lambda xy: np.stack([(np.asarray(xy, float)[..., 0] + hx) / self.FINE, (hy - np.asarray(xy, float)[..., 1]) / self.FINE], -1)
        a, b = px([-env.MAP_X / 2, env.MAP_Y / 2]), px([env.MAP_X / 2, -env.MAP_Y / 2])
        occ[int(round(a[1])):int(round(b[1])), int(round(a[0])):int(round(b[0]))] = 0
        if getattr(env, "shape", None) == "L":
            sx, sy = [(1, 1), (-1, 1), (1, -1), (-1, -1)][env.location]
            cx, cy = sx * (env.MAP_X / 2 - env.L_X / 2), sy * (env.MAP_Y / 2 - env.L_Y / 2)
            a, b = px([cx - env.L_X / 2, cy + env.L_Y / 2]), px([cx + env.L_X / 2, cy - env.L_Y / 2])
            occ[int(round(a[1])):int(round(b[1])), int(round(a[0])):int(round(b[0]))] = 1
        SH = 4
        polys = [_rect(w.x, w.y, w.orientation, w.length, w.thickness) for w in env.walls]
        polys += [_rect(o.x, o.y, o.orientation, o.length, o.width) for o in list(env.tables) + list(env.chairs) + list(env.laptops)]
        for p in polys:
            cv2.fillPoly(occ, [np.round(px(p) * (1 << SH)).astype(np.int32)], 1, shift=SH)
        for p in env.plants:
            c = px([p.x, p.y]); cv2.circle(occ, tuple(np.round(c * (1 << SH)).astype(int)), int(round(p.radius / self.FINE * (1 << SH))), 1, -1, shift=SH)
        k = max(1, int(round(BASE_RES / self.FINE)))
        occ = cv2.dilate(occ, np.ones((k, k), np.uint8)).astype(np.float32)
        f = self.FINE / BASE_RES
        self.grid = cv2.resize(occ, (int(round(W * f)), int(round(H * f))), interpolation=cv2.INTER_AREA)
        self.x0, self.y0 = -hx + 0.5 * BASE_RES, -hy + 0.5 * BASE_RES

    def patch(self, x, y, th):
        """(2, PATCH, PATCH): columns forward from -PATCH_BACK, rows from left (+y) to right; outside = occupied."""
        c, s = np.cos(th), np.sin(th); H = self.grid.shape[0]
        def cell(fx, fy):
            wx, wy = x + c * fx - s * fy, y + s * fx + c * fy
            return np.array([(wx - self.x0) / BASE_RES, (H - 1) - (wy - self.y0) / BASE_RES])
        fx0, fy0 = -PATCH_BACK + 0.5 * PATCH_RES, PATCH * PATCH_RES / 2 - 0.5 * PATCH_RES
        p00, p10, p01 = cell(fx0, fy0), cell(fx0 + PATCH_RES, fy0), cell(fx0, fy0 - PATCH_RES)
        M = np.array([[p10[0] - p00[0], p01[0] - p00[0], p00[0]], [p10[1] - p00[1], p01[1] - p00[1], p00[1]]], np.float32)
        occ = cv2.warpAffine(self.grid, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        return np.stack([occ, np.zeros_like(occ)])


# ------------------------------------------------------------------ model (must match humanmotion/forecaster2.MotionModel)
def _vel(p, v):
    d = torch.zeros_like(p); d[..., 1:, :] = (p[..., 1:, :] - p[..., :-1, :]) * 10
    ok = torch.zeros_like(v); ok[..., 1:] = v[..., 1:] & v[..., :-1]
    return d * ok[..., None]


class MapCNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        c = 32
        self.net = nn.Sequential(nn.Conv2d(2, c, 5, 2, 2), nn.GELU(), nn.Conv2d(c, c * 2, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(c * 2, d, 3, 2, 1), nn.GELU(), nn.Conv2d(d, d, 3, 1, 1))
        self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)

    def forward(self, patch):
        return self.net(patch).flatten(2).transpose(1, 2) + self.pos


class Encoder2(nn.Module):
    def __init__(self, d=128, layers=3, dropout=0.1):
        super().__init__()
        self.ego = nn.Sequential(nn.Linear(T_OBS * 7, 256), nn.GELU(), nn.Linear(256, d))
        self.intent = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.nb = nn.Sequential(nn.Linear(T_OBS * 5 + 4, 256), nn.GELU(), nn.Linear(256, d))
        self.kind = nn.Embedding(4, d)
        self.map = MapCNN(d)
        el = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)
        self.out = nn.LayerNorm(d)

    def forward(self, b, patch, dest_on, speed_on, map_on):
        B = b["hist"].shape[0]; dev = b["hist"].device
        hv = b["hvalid"]; hvf = hv.float()
        hs, hc = torch.sin(b["head"]) * b["head_valid"], torch.cos(b["head"]) * b["head_valid"]
        ego = torch.cat([b["hist"] * hvf[..., None], _vel(b["hist"], hv), hs[..., None], hc[..., None], hvf[..., None]], -1).flatten(1)
        dst = b["dest"] * dest_on[:, None]
        intent = torch.stack([dst[:, 0] / 5, dst[:, 1] / 5, dst.norm(dim=-1) / 5, b["speed"] * speed_on, dest_on.float(), speed_on.float()], -1)
        nv = b["nb_valid"]; nvf = nv.float()
        nfeat = torch.cat([b["nb"] * nvf[..., None], _vel(b["nb"], nv), nvf[..., None]], -1).flatten(2)
        oh = torch.nn.functional.one_hot(b["nb_type"].clamp(0, 2), 3).float()
        nfeat = torch.cat([nfeat, oh, b["nb_group"].float()[..., None]], -1)
        toks = [self.ego(ego)[:, None] + self.kind.weight[0], self.intent(intent)[:, None] + self.kind.weight[1],
                self.nb(nfeat) + self.kind.weight[2], self.map(patch) + self.kind.weight[3]]
        masks = [torch.zeros(B, 1, dtype=torch.bool, device=dev), ~(dest_on | speed_on)[:, None], ~b["nb_mask"],
                 (~map_on)[:, None].expand(B, 64)]
        return self.out(self.tf(torch.cat(toks, 1), src_key_padding_mask=torch.cat(masks, 1))[:, 0])


class MotionModel(nn.Module):
    def __init__(self, d=128, layers=3, dropout=0.1):
        super().__init__()
        self.enc = Encoder2(d, layers, dropout)
        self.temb = nn.Sequential(nn.Linear(16, d), nn.GELU())
        self.v = nn.Sequential(nn.Linear(2 * d + T_PRED * 2, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, T_PRED * 2))

    def tfeat(self, tau):
        a = tau[:, None] * (2 ** torch.arange(8, device=tau.device).float())[None] * math.pi
        return self.temb(torch.cat([torch.sin(a), torch.cos(a)], -1))

    @torch.no_grad()
    def sample(self, b, patch, dest_on, speed_on, map_on, n=1, steps=10, temp=1.0, gen=None):
        cond = self.enc(b, patch, dest_on, speed_on, map_on)
        B = cond.shape[0]; cond = cond.repeat_interleave(n, 0)
        x = torch.randn(B * n, T_PRED * 2, generator=gen) * temp
        for k in range(steps):
            tau = torch.full((B * n,), k / steps)
            x = x + self.v(torch.cat([cond, self.tfeat(tau), x], -1)) / steps
        return (x * SCALE).view(B, n, T_PRED, 2)


# ------------------------------------------------------------------ controller used by SocNavEnv_v2
class LearnedHumanController:
    def __init__(self, model_path=None, replan=4, temperature=0.5, speed_range=(0.6, 1.4), flow_steps=10, threads=1):
        # temperature 0.5: closed-loop tests at 1.0 gave jittery steps (p90 speed 1.9 m/s) and 1.2% of steps inside furniture; 0.5 -> 1.6 m/s, 0.1%
        path = model_path or DEFAULT_MODEL
        if not os.path.isfile(path):
            raise FileNotFoundError(f"learned human model not found: {path}")
        torch.set_num_threads(threads)
        self.model = MotionModel(); self.model.load_state_dict(torch.load(path, map_location="cpu")); self.model.eval()
        self.replan, self.temp, self.speed_range, self.flow_steps = replan, temperature, tuple(speed_range), flow_steps

    def reset(self, env, learned_humans):
        self.env = env
        self.ids = [h.id for h in learned_humans]
        self.raster = SceneRaster(env)
        rng = env.np_random
        self.gen = torch.Generator().manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
        self.rng = rng; self.pref_speed = {}; self.sees_robot = {}
        for h in learned_humans: self._init_human(h)
        self.hist = {}; self.yaw = {}; self.robot_hist = deque(maxlen=T_OBS)
        self.plan = {}; self.plan_goal = {}; self.since = 0
        self.record()

    def _init_human(self, h):
        """Per-human draws, also for humans that become learned walkers mid-episode (e.g. a dispersing group)."""
        if h.id in self.pref_speed: return
        ps = getattr(h, "pref_speed", None)                # env-wide speed_range, if configured, wins
        self.pref_speed[h.id] = float(ps) if ps is not None else float(self.rng.uniform(*self.speed_range))
        # whether the human reacts to the robot: drawn once per human (Human.avoids_robot re-draws on every access)
        self.sees_robot[h.id] = bool(self.rng.random() < h.prob_to_avoid_robot)

    def _all_humans(self):
        out = [(h, None) for h in self.env.static_humans + self.env.dynamic_humans]
        for i in self.env.moving_interactions + self.env.static_interactions:
            out += [(h, id(i)) for h in i.humans]
        out += [(i.human, None) for i in self.env.h_l_interactions]
        return out

    def record(self):
        for h, _ in self._all_humans():
            self.hist.setdefault(h.id, deque(maxlen=T_OBS)).append((h.x, h.y))
            self.yaw.setdefault(h.id, deque(maxlen=T_OBS)).append(h.orientation)
        r = self.env.robot; self.robot_hist.append((r.x, r.y))

    def invalidate(self, hid):
        self.plan.pop(hid, None)

    def next_positions(self, humans):
        """Call once per env step, before moving the humans. Returns {id: (x, y)} for the given learned humans."""
        need = [h for h in humans if not self.plan.get(h.id) or self.plan_goal.get(h.id) != (h.goal_x, h.goal_y)]
        if self.since % self.replan == 0: need = list(humans)
        self.since += 1
        if need: self._replan(need)
        return {h.id: self.plan[h.id].pop(0) for h in humans}

    def _replan(self, humans):
        for h in humans: self._init_human(h)
        allh = self._all_humans(); group_of = {h.id: g for h, g in allh}
        items, poses = [], []
        for h in humans:
            P = np.array(self.hist[h.id], np.float32); Y = np.array(self.yaw[h.id], np.float32)
            o = P[-1]; th = float(h.orientation)
            if len(P) > 5 and np.hypot(*(P[-1] - P[-6])) > 0.15: th = float(np.arctan2(*(P[-1] - P[-6])[::-1]))
            c, s = np.cos(th), np.sin(th)
            tf = lambda Q: np.stack([c * (Q[..., 0] - o[0]) + s * (Q[..., 1] - o[1]), -s * (Q[..., 0] - o[0]) + c * (Q[..., 1] - o[1])], -1).astype(np.float32)
            def pad(Q):                                     # left-pad a (k, 2) history to T_OBS with a validity mask
                out = np.zeros((T_OBS, 2), np.float32); v = np.zeros(T_OBS, bool)
                if len(Q): out[-len(Q):] = tf(Q); v[-len(Q):] = True
                return out, v
            hist, hv = pad(P)
            head = np.zeros(T_OBS, np.float32); hdv = np.zeros(T_OBS, bool)
            yy = np.arctan2(np.sin(Y - th), np.cos(Y - th)); head[-len(yy):] = yy; hdv[-len(yy):] = True
            cands = []
            for h2, g2 in allh:
                if h2.id == h.id: continue
                d = np.hypot(h2.x - o[0], h2.y - o[1])
                if d < NB_RADIUS: cands.append((d, np.array(self.hist[h2.id], np.float32), 0, int(g2 is not None and g2 == group_of[h.id])))
            if self.sees_robot[h.id]:
                R = np.array(self.robot_hist, np.float32); d = np.hypot(*(R[-1] - o))
                if d < NB_RADIUS: cands.append((d, R, 1, 0))
            cands = sorted(cands, key=lambda c_: c_[0])[:K_NB]
            nb = np.zeros((K_NB, T_OBS, 2), np.float32); nbv = np.zeros((K_NB, T_OBS), bool)
            nbt = np.zeros(K_NB, np.int64); nbg = np.zeros(K_NB, np.int64); nbm = np.zeros(K_NB, bool)
            for q, (_, Q, ty, gg) in enumerate(cands):
                nb[q], nbv[q] = pad(Q); nbt[q] = ty; nbg[q] = gg; nbm[q] = True
            goal = np.array([h.goal_x, h.goal_y], np.float32)
            items.append(dict(hist=hist, hvalid=hv, head=head, head_valid=hdv, nb=nb, nb_valid=nbv, nb_type=nbt, nb_group=nbg,
                              nb_mask=nbm, dest=tf(goal), speed=np.float32(self.pref_speed[h.id])))
            poses.append((o[0], o[1], th))
        b = {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in items[0]}
        patch = torch.as_tensor(np.stack([self.raster.patch(*p) for p in poses]))
        on = torch.ones(len(items), dtype=torch.bool)
        fut = self.model.sample(b, patch, on, on, on, n=1, steps=self.flow_steps, temp=self.temp, gen=self.gen)[:, 0].numpy()
        for h, (x, y, th), f in zip(humans, poses, fut):
            c, s = np.cos(th), np.sin(th)
            w = np.stack([x + c * f[:, 0] - s * f[:, 1], y + s * f[:, 0] + c * f[:, 1]], 1)
            self.plan[h.id] = [tuple(p) for p in w[:max(self.replan, 1) + 2]]
            self.plan_goal[h.id] = (h.goal_x, h.goal_y)
