import pathlib
import unittest

import gym
import socnavgym


class TestEnvConfigs(unittest.TestCase):
    def test_env_configs(self):
        """
        Test env creation from config file contained in the cfg directory.
        """
        cfg_dir = pathlib.Path(__file__).parent.parent / "environment_configs"

        for cfg_file in cfg_dir.glob("*yaml"):
            cfg_filepath = cfg_dir / cfg_file
            env = gym.make("SocNavGym-v1", config=str(cfg_filepath))
            obs, _ = env.reset()

            self.assertTrue(
                env.observation_space.contains(obs),
                "the observation from reset is not valid",
            )
