import socnavgym
from socnavgym.envs.rewards.reward_api import RewardAPI
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, EntityObs
from socnavgym.envs.utils.utils import point_to_segment_dist
import numpy as np


class Reward(RewardAPI):
    def __init__(self, env: SocNavEnv_v1) -> None:
        super().__init__(env)
        self.use_sngnn = True
        self.sngnn_factor = 1.0
        self.reach_reward = 1.0
        self.reach_reward = 1.0 
        self.out_of_map_reward = -1.0 
        self.max_steps_reward = -1.0 
        self.alive_reward = -0.00001 
        self.collision_reward = -1.0
        self.distance_reward_scaler = 5.0
        self.discomfort_distance = 0.6
        self.discomfort_penalty_factor = 0.5
        self.prev_distance = None

    def compute_dmin(self, action):
        dmin = float('inf')

        all_humans = []
        for human in self.env.static_humans + self.env.dynamic_humans : all_humans.append(human)

        for i in self.env.static_interactions + self.env.moving_interactions:
            for h in i.humans: all_humans.append(h)
        
        for i in self.env.h_l_interactions: all_humans.append(i.human)

        for human in all_humans:
            px = human.x - self.env.robot.x
            py = human.y - self.env.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for human in self.env.static_humans + self.env.dynamic_humans:
            px = human.x - self.env.robot.x
            py = human.y - self.env.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for interaction in (self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions):
            px = interaction.x - self.env.robot.x
            py = interaction.y - self.env.robot.y

            speed = 0
            if interaction.name == "human-human-interaction":
                for h in interaction.humans:
                    speed += h.speed
                speed /= len(interaction.humans)

            vx = speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist
        
        return dmin

    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        if self.check_out_of_map(): return self.out_of_map_reward
        elif self.check_reached_goal(): return self.reach_reward
        elif self.check_collision(): return self.collision_reward
        elif self.check_timeout(): return self.max_steps_reward
        else:
            sngnn_value = self.compute_sngnn_reward(action, prev_obs, curr_obs)
            sngnn_reward = (sngnn_value - 1.0) * self.sngnn_factor
            distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x)**2 + (self.env.robot.goal_y - self.env.robot.y)**2)
            distance_reward = 0.0
            if self.prev_distance is not None:
                distance_reward = -(distance_to_goal-self.prev_distance) * self.distance_reward_scaler
            self.prev_distance = distance_to_goal

            dsrnn_reward = 0.0  # calculating this reward only for comparison purposes. It will be stored in the info dict, but wont be returned
            dmin = self.compute_dmin(action)
            if dmin < self.discomfort_distance:
                dsrnn_reward = (dmin - self.discomfort_distance) * self.discomfort_penalty_factor * self.env.TIMESTEP

            self.info["DISCOMFORT_SNGNN"] = sngnn_value
            self.info["DISCOMFORT_DSRNN"] = dsrnn_reward
            self.info["distance_reward"] = distance_reward
            self.info["alive_reward"] = self.alive_reward
            self.info["sngnn_reward"] = sngnn_reward

            return sngnn_reward + distance_reward + self.alive_reward
