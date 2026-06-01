import sys

import numpy as np

import socnavgym
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2, EntityObs
from socnavgym.envs.utils.utils import point_to_segment_dist
from socnavgym.envs.rewards.reward_api import RewardAPI

class Reward(RewardAPI):
    def __init__(self, env: SocNavEnv_v2) -> None:
        super().__init__(env)
        self.reach_reward = 10.0
        self.out_of_map_reward = -5.0 
        self.max_steps_reward = -8.0 
        self.collision_reward = -10.0


    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        if self.check_out_of_map():
            return self.out_of_map_reward
        elif self.check_reached_goal():
            return self.reach_reward
        elif self.check_collision():
            return self.collision_reward
        elif self.check_timeout():
            return self.max_steps_reward
        else:
            return 0.0
