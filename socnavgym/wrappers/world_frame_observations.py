"""World-frame observations for SocNavGym-v2 (re-implemented for v2's padded dict observations).

Entity rows keep v2's layout [x, y, sin, cos, radius, rel_speed, rel_ang_speed, gaze], but x, y and the angle are
expressed in the world frame instead of the robot frame. The robot observation becomes
[goal_x, goal_y, goal_radius, sin(goal_a), cos(goal_a), goal_angle_threshold, robot_radius, 1, robot_x, robot_y, sin(robot_a), cos(robot_a)]
with the goal in the world frame.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

ENTITIES = ("humans", "tables", "laptops", "plants", "chairs", "walls")


class WorldFrameObservations(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        base = env.observation_space
        u = env.unwrapped
        half = max(u.MAX_MAP_X, u.MAX_MAP_Y) / 2
        d = {}
        for k, sp in base.spaces.items():
            if k == "robot":
                low = np.concatenate([sp.low, [-half, -half, -1, -1]]).astype(np.float32); low[0:2] = -half; low[3:5] = -1
                high = np.concatenate([sp.high, [half, half, 1, 1]]).astype(np.float32); high[0:2] = half; high[3:5] = 1; high[7] = 1
                d[k] = spaces.Box(low=low, high=high, dtype=np.float32)
            elif k not in ENTITIES: d[k] = sp                  # e.g. "range" (the LiDAR's distances have no frame to change): as it is
            else:
                low, high = sp.low.copy().reshape(-1, 8), sp.high.copy().reshape(-1, 8)
                low[:, 0:2], high[:, 0:2] = -half, half
                d[k] = spaces.Box(low=low.reshape(-1), high=high.reshape(-1), dtype=np.float32)
        self.observation_space = spaces.Dict(d)

    def observation(self, obs):
        r = self.env.unwrapped.robot
        c, s = np.cos(r.orientation), np.sin(r.orientation)
        def to_world(xy):                                      # robot frame -> world
            return np.stack([r.x + c * xy[:, 0] - s * xy[:, 1], r.y + s * xy[:, 0] + c * xy[:, 1]], 1)
        out = {}
        ro = obs["robot"].astype(np.float32)
        g = to_world(ro[None, 0:2])[0]
        sa, ca = ro[3] * c + ro[4] * s, ro[4] * c - ro[3] * s  # goal angle = robot angle + relative angle
        out["robot"] = np.concatenate([[g[0], g[1], ro[2], sa, ca], ro[5:8], [r.x, r.y, s, c]]).astype(np.float32)
        for e in ENTITIES:
            if e not in obs: continue
            o = obs[e].reshape(-1, 8).copy()
            present = np.any(o != 0, axis=1)
            if present.any():
                p = o[present]
                p[:, 0:2] = to_world(p[:, 0:2])
                sn, cs = p[:, 2] * c + p[:, 3] * s, p[:, 3] * c - p[:, 2] * s
                p[:, 2], p[:, 3] = sn, cs
                o[present] = p
            out[e] = o.reshape(-1).astype(np.float32)
        for k in obs:                                          # anything else (e.g. "range", still relative to the robot's heading): as it is
            if k not in out: out[k] = obs[k]
        return out
