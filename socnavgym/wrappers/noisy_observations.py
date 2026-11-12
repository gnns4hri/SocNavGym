"""Gaussian observation noise for SocNavGym-v2 (re-implemented for v2's padded dict observations; the v1 version
relied on v1-only internals such as get_padded_observations and the robot one-hot encoding)."""
import gymnasium as gym
import numpy as np

ENTITIES = ("humans", "tables", "laptops", "plants", "chairs", "walls")


class NoisyObservations(gym.ObservationWrapper):
    def __init__(self, env, mean, std_dev, apply_noise_to=("robot", "humans", "tables", "laptops", "plants", "walls")) -> None:
        """Adds N(mean, std_dev) to positions: the goal (robot[0:2], robot frame) and the x, y of every present entity
        listed in apply_noise_to. Padded (empty) slots stay zero."""
        super().__init__(env)
        self.mean, self.std_dev = mean, std_dev
        self.apply_noise_to = list(apply_noise_to)
        for e in self.apply_noise_to:
            assert e == "robot" or e in ENTITIES, f"apply_noise_to entries must be 'robot' or one of {ENTITIES}"

    def observation(self, obs):
        obs = {k: v.copy() for k, v in obs.items()}
        rng = self.env.unwrapped.np_random
        if "robot" in self.apply_noise_to:
            obs["robot"][0:2] += rng.normal(self.mean, self.std_dev, 2).astype(np.float32)
        for e in self.apply_noise_to:
            if e == "robot" or e not in obs: continue
            o = obs[e].reshape(-1, 8)
            present = np.any(o != 0, axis=1)
            o[present, 0:2] += rng.normal(self.mean, self.std_dev, (int(present.sum()), 2)).astype(np.float32)
            obs[e] = o.reshape(-1)
        return obs
