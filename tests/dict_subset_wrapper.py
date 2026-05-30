from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DictToFlatWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that extracts a subset of keys from a Dict observation
    space and concatenates their values into a single flat Box (1-D vector).

    All selected keys must have Box spaces with dtype float32 (or safely
    castable to it).

    Compatible with Stable-Baselines3 using "MlpPolicy".

    Args:
        env:  The environment to wrap.
        keys: Ordered list of keys to extract and concatenate.

    Example:
        env = YourDictEnv()
        env = DictToFlatWrapper(env, keys=["position", "velocity"])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100_000)
    """

    def __init__(self, env: gym.Env, keys: list[str]):
        super().__init__(env)

        original_space = env.observation_space
        assert isinstance(original_space, spaces.Dict), (
            f"DictToFlatWrapper requires a Dict observation space, "
            f"got {type(original_space)}"
        )

        missing = [k for k in keys if k not in original_space.spaces]
        assert not missing, (
            f"Keys not found in observation space: {missing}. "
            f"Available keys: {list(original_space.spaces.keys())}"
        )

        for k in keys:
            assert isinstance(original_space.spaces[k], spaces.Box), (
                f"Key '{k}' has space {type(original_space.spaces[k])}; "
                f"only Box spaces can be concatenated into a flat vector."
            )

        self.keys = keys

        # Pre-compute flat size and per-key slice indices for fast obs()
        self._slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for k in keys:
            size = int(np.prod(original_space.spaces[k].shape))
            self._slices[k] = (cursor, cursor + size)
            cursor += size

        total = cursor

        # Build low/high bounds by concatenating each key's bounds
        low  = np.concatenate([original_space.spaces[k].low.flat  for k in keys]).astype(np.float32)
        high = np.concatenate([original_space.spaces[k].high.flat for k in keys]).astype(np.float32)

        self.observation_space = spaces.Box(
            low=low, high=high, shape=(total,), dtype=np.float32
        )

    def observation(self, obs: dict[str, Any]) -> np.ndarray:
        """Flatten and concatenate the selected keys into a single vector."""
        return np.concatenate(
            [obs[k].flatten().astype(np.float32) for k in self.keys]
        )


# --- Quick smoke-test --------------------------------------------------------

if __name__ == "__main__":
    class DummyDictEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict({
                "position": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                "velocity": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                "lidar":    spaces.Box( 0, 1, shape=(8,), dtype=np.float32),
            })
            self.action_space = spaces.Discrete(4)

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

        def step(self, action):
            obs = self.observation_space.sample()
            return obs, 0.0, False, False, {}

    env = DummyDictEnv()
    wrapped = DictToFlatWrapper(env, keys=["position", "velocity"])

    print("Wrapped obs space:", wrapped.observation_space)
    print("Expected shape   : (4,)  — 2 (position) + 2 (velocity)")

    obs, _ = wrapped.reset()
    print("Obs shape after reset:", obs.shape)
    assert obs.shape == (4,), f"Unexpected shape: {obs.shape}"
    print("All checks passed ✓")
