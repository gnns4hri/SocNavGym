import torch
import socnavgym
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2
from socnavgym.envs.utils.utils import point_to_segment_dist
from socnavgym.envs.utils import Human, Human_Human_Interaction, Human_Laptop_Interaction, Plant, Laptop, Wall
from socnavgym.envs.socnavenv_v2 import EntityObs
from typing import Dict
import numpy as np
import os

class RewardAPI:
    def __init__(self, env:SocNavEnv_v2) -> None:
        self.env = env
        self.sn_sequence = []
        self.info = {}  # record any information that should be returned in the info dict (in step function of the environment) here

        ## default values that should be calculated in your reward function to be returned
        self.info["DISCOMFORT_SNGNN"] = 0
        self.info["DISCOMFORT_DSRNN"] = 0
        self.info["distance_reward"] = 0
        self.info["alive_reward"] = 0
        self.info["sngnn_reward"] = 0

    def re_init(self, env:SocNavEnv_v2):
        self.__init__(env)

    def check_collision(self):
        # check for object-robot collisions

        for object in self.env.static_humans + self.env.dynamic_humans + self.env.plants + self.env.walls + self.env.tables + self.env.laptops:
            if self.env.robot.collides(object):
                return True

        # interaction-robot collision
        for i in (self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions):
            if i.collides(self.env.robot):
                return True

        return False

    def check_out_of_map(self):
        return (self.env.MAP_X/2 < self.env.robot.x) or (self.env.robot.x < -self.env.MAP_X/2) or (self.env.MAP_Y/2 < self.env.robot.y) or (self.env.robot.y < -self.env.MAP_Y/2)

    def check_reached_goal(self):
        distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x)**2 + (self.env.robot.goal_y - self.env.robot.y)**2)
        angular_distance_to_goal = np.abs(self.env.robot.goal_a - self.env.robot.orientation)
        return distance_to_goal < self.env.GOAL_THRESHOLD and angular_distance_to_goal < self.env.GOAL_ORIENTATION_THRESHOLD

    def check_timeout(self):
        return self.env.ticks > self.env.EPISODE_LENGTH

    def update_env(self, new_env:SocNavEnv_v2):
        self.env = new_env

    def compute_reward(self, action, prev_obs:EntityObs, curr_obs:EntityObs):
        raise NotImplementedError
