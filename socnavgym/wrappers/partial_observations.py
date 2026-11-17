"""Limited field of view / sensor range for SocNavGym-v2 (re-implemented for v2's padded dict observations)."""
import gymnasium as gym
import numpy as np

ENTITIES = ("humans", "tables", "laptops", "plants", "chairs", "walls")


class PartialObservations(gym.ObservationWrapper):
    def __init__(self, env, fov_angle: float, range: float) -> None:
        """
        Args:
            fov_angle (float): in radians; entities are visible within [-fov_angle/2, +fov_angle/2] of the robot heading (+x).
            range (float): maximum sensing distance in metres (to the entity centre).
        Entities outside the field of view or range are removed: their slot is set to zeros, i.e. exactly what an
        absent entity looks like in v2's padded observations. The goal (robot observation) is always visible.
        """
        super().__init__(env)
        assert 0 <= fov_angle <= 2 * np.pi, "FOV angle should be between 0 and 2*pi"
        self.fov_angle, self.range = fov_angle, range

    def observation(self, obs):
        obs = {k: v.copy() for k, v in obs.items()}
        for e in ENTITIES:
            if e not in obs: continue
            o = obs[e].reshape(-1, 8)
            d = np.hypot(o[:, 0], o[:, 1]); ang = np.abs(np.arctan2(o[:, 1], o[:, 0]))
            hidden = (d > self.range) | (ang > self.fov_angle / 2)
            o[hidden] = 0.0
            obs[e] = o.reshape(-1)
        return obs
