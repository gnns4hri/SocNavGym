import gym
from gym.spaces import Discrete
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
import numpy as np


class DiscreteActions(gym.ActionWrapper):
    """A wrapper class to take in discrete actions, and convert to continuous action space.
    """
    def __init__(self, env: SocNavEnv_v1):
        super().__init__(env)
        self.action_space = Discrete(7)

    def discrete_to_continuous_action(self, action:int):
        # Turning anti-clockwise
        if action == 0:
            return np.array([0, 0.0, 1.0], dtype=np.float32) 
        # Turning clockwise
        elif action == 1:
            return np.array([0, 0.0, -1.0], dtype=np.float32) 
        # Turning anti-clockwise and moving forward
        elif action == 2:
            return np.array([1, 0.0, 1.0], dtype=np.float32) 
        # Turning clockwise and moving forward
        elif action == 3:
            return np.array([1, 0.0, -1.0], dtype=np.float32) 
        # Move forward
        elif action == 4:
            return np.array([1, 0.0, 0.0], dtype=np.float32)
        # Move backward
        elif action == 5:
            return np.array([-1, 0.0, 0.0], dtype=np.float32)
        # No Op
        elif action == 6:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            raise NotImplementedError

    def action(self, action):
        return self.discrete_to_continuous_action(action)
    
