import pytest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gym
import socnavgym
from socnavgym.wrappers import WorldFrameObservations, PartialObservations, NoisyObservations, DiscreteActions
from gym.utils.env_checker import check_env
import numpy as np
from glob import glob


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_configs():
    dir_name = os.path.dirname(os.path.abspath(__file__)) + "/../environment_configs/"
    configs = glob(dir_name + "/*")
    
    for config in configs:
        env = gym.make("SocNavGym-v1", config=config)
        obs, _ = env.reset()
        assert(obs in env.observation_space)