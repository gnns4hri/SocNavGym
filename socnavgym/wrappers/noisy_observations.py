import gym
from gym import spaces
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.utils.wall import Wall
import numpy as np
import copy

class NoisyObservations(gym.Wrapper):
    def __init__(self, env: SocNavEnv_v1, mean, std_dev, apply_noise_to=["robot", "humans", "tables", "laptops", "plants", "walls"]) -> None:
        """
        A Gaussian Noise of mean, and std_dev are added to the values of the observations that are received.

        apply_noise_to is a list with which you can control what observations would you want to add noise to
        """
        super().__init__(env)
        self.env = env
        self.mean = mean
        self.std_dev = std_dev
        self.max_noise = 0
        self.apply_noise_to = apply_noise_to
        for entity in self.apply_noise_to:
            assert(entity=="robot" or entity=="humans" or entity=="plants" or entity=="laptops" or entity=="tables" or entity=="walls"),"apply_noise_to only have the following names: \"goal\" \"humans\" \"tables\" \"laptops\" \"plants\" \"walls\""
    
    @property
    def observation_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {

            "robot": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2)-self.max_noise, -self.MAP_Y * np.sqrt(2)-self.max_noise, -self.ROBOT_RADIUS-self.max_noise], dtype=np.float32), 
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2)+self.max_noise, +self.MAP_Y * np.sqrt(2)+self.max_noise, +self.ROBOT_RADIUS+self.max_noise], dtype=np.float32),
                shape=((self.robot.one_hot_encoding.shape[0]+3, )),
                dtype=np.float32

            )
        }

        if self.env.is_entity_present["humans"]:
            d["humans"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2)-self.max_noise, -self.MAP_Y * np.sqrt(2)-self.max_noise, -1.0-self.max_noise, -1.0-self.max_noise, -self.HUMAN_DIAMETER/2-self.max_noise, -(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT)*np.sqrt(2)-self.max_noise, -2*np.pi/self.TIMESTEP -self.max_noise, 0-self.max_noise] * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2)+self.max_noise, +self.MAP_Y * np.sqrt(2)+self.max_noise, 1.0+self.max_noise, 1.0+self.max_noise, self.HUMAN_DIAMETER/2+self.max_noise, +(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT)*np.sqrt(2)+self.max_noise, +2*np.pi/self.TIMESTEP +self.max_noise, 1+self.max_noise] * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * ((self.MAX_HUMANS + self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans),)),
                dtype=np.float32
            )

        if self.env.is_entity_present["laptops"]:
            d["laptops"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2)-self.max_noise, -self.MAP_Y * np.sqrt(2)-self.max_noise, -1.0-self.max_noise, -1.0-self.max_noise, -self.LAPTOP_RADIUS-self.max_noise, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)-self.max_noise, -self.MAX_ROTATION-self.max_noise, 0-self.max_noise] * ((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2)+self.max_noise, +self.MAP_Y * np.sqrt(2)+self.max_noise, 1.0+self.max_noise, 1.0+self.max_noise, self.LAPTOP_RADIUS+self.max_noise, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)+self.max_noise, +self.MAX_ROTATION+self.max_noise, 1+self.max_noise] * ((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)),)),
                dtype=np.float32

            )

        if self.env.is_entity_present["tables"]:
            d["tables"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2)-self.max_noise, -self.MAP_Y * np.sqrt(2)-self.max_noise, -1.0-self.max_noise, -1.0-self.max_noise, -self.TABLE_RADIUS-self.max_noise, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)-self.max_noise, -self.MAX_ROTATION-self.max_noise, 0-self.max_noise] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2)+self.max_noise, +self.MAP_Y * np.sqrt(2)+self.max_noise, 1.0+self.max_noise, 1.0+self.max_noise, self.TABLE_RADIUS+self.max_noise, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)+self.max_noise, +self.MAX_ROTATION+self.max_noise, 1+self.max_noise] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*(self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES),)),
                dtype=np.float32

            )

        if self.env.is_entity_present["plants"]:
            d["plants"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2)-self.max_noise, -self.MAP_Y * np.sqrt(2)-self.max_noise, -1.0-self.max_noise, -1.0-self.max_noise, -self.PLANT_RADIUS-self.max_noise, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)-self.max_noise, -self.MAX_ROTATION-self.max_noise, 0-self.max_noise] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2)+self.max_noise, +self.MAP_Y * np.sqrt(2)+self.max_noise, 1.0+self.max_noise, 1.0+self.max_noise, self.PLANT_RADIUS+self.max_noise, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)+self.max_noise, +self.MAX_ROTATION+self.max_noise, 1+self.max_noise] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*(self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS),)),
                dtype=np.float32

            )
        

        if not self.get_padded_observations:
            total_segments = 0
            for w in self.walls:
                total_segments += w.length//self.WALL_SEGMENT_SIZE
                if w.length % self.WALL_SEGMENT_SIZE != 0: total_segments += 1
            
            if self.env.is_entity_present["walls"]:
                d["walls"] = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2)-self.max_noise, -self.MAP_Y * np.sqrt(2)-self.max_noise, -1.0-self.max_noise, -1.0-self.max_noise, -self.WALL_SEGMENT_SIZE-self.max_noise, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)-self.max_noise, -self.MAX_ROTATION-self.max_noise, 0-self.max_noise] * int(total_segments), dtype=np.float32),
                    high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2)+self.max_noise, +self.MAP_Y * np.sqrt(2)+self.max_noise, 1.0+self.max_noise, 1.0+self.max_noise, +self.WALL_SEGMENT_SIZE+self.max_noise, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2)+self.max_noise, +self.MAX_ROTATION+self.max_noise, 1+self.max_noise] * int(total_segments), dtype=np.float32),
                    shape=(((self.robot.one_hot_encoding.shape[0] + 8)*int(total_segments),)),
                    dtype=np.float32
                )

        return spaces.Dict(d)

    def generate_random_noise(self):
        noise = np.random.randn()*self.std_dev + self.mean
        self.max_noise = max(self.max_noise, abs(noise))
        return noise

    def add_noise(self, obs):
        noisy_obs = obs
        encoding_size = self.env.robot.one_hot_encoding.shape[0]
        if "robot" in self.apply_noise_to:
            # adding noise to goal
            noisy_obs["robot"][encoding_size] += self.generate_random_noise()
            noisy_obs["robot"][encoding_size+1] += self.generate_random_noise()
        
        # entity list contains the names of entities that we need to add noise to
        entity_list = []

        if not self.env.get_padded_observations: 
            for entity_name in self.apply_noise_to:
                if entity_name == "robot": continue
                entity_list.append(entity_name)
        else:
            # if padded observations are to be returned, then we cannot add noise to walls, thus removing walls from the entity_list
            for entity_name in self.apply_noise_to:
                if entity_name == "robot" or entity_name == "walls": continue
                entity_list.append(entity_name)
        for entity in entity_list:
            # if the entity is not present in the observation, then continue
            if entity not in noisy_obs.keys(): continue
            
            o = noisy_obs[entity].reshape(-1, self.env.entity_obs_dim)
            for i in range(o.shape[0]):
                for j in range(encoding_size, self.env.entity_obs_dim): # noise is only added to the non-one-hot components of the observation
                    o[i][j] += self.generate_random_noise()
            noisy_obs[entity] = o.flatten()
        return noisy_obs

    def step(self, action_pre):
        obs, reward, terminated, truncated, info = self.env.step(action_pre)
        obs = self.add_noise(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        obs = self.add_noise(obs)
        return obs, info

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self.env)
        obs, reward, terminated, truncated, info = env_copy.step(action_pre)
        obs = self.add_noise(obs)
        del env_copy
        return obs, reward, terminated, truncated, info
