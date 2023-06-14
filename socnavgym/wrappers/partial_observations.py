import gym
from gym import spaces
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.utils.wall import Wall
from socnavgym.envs.utils.utils import w2px, w2py
import sys
from typing import Dict
import numpy as np
import copy
import cv2

class PartialObservations(gym.Wrapper):
    def __init__(self, env: SocNavEnv_v1, fov_angle:float, range:float) -> None:
        """
        Args:
            env (SocNavEnv_v1): environment to be wrapped
            fov_angle (float): fov_angle is assumed to be in radians. The range of vision will be assumed from [-fov_angle/2, +fov_angle/2]. The robot heading is assumed to be where the X-axis lies.
            range (_type_): range is the sensor's maximum range (in meters)
        """
        super().__init__(env)
        self.env = env
        self.fov_angle = fov_angle
        self.range = range
        assert(self.fov_angle <= 2*np.pi and self.fov_angle>=0), "FOV angle should be between 0 and 2*pi"
        self.num_humans = 0
        self.num_plants = 0
        self.num_tables = 0
        self.num_laptops = 0
        self.num_walls = 0

    @property
    def observation_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """
        obs = self.latest_obs
        self.is_entity_present = {}
        for name in ["humans", "plants", "tables", "laptops", "walls"]: self.is_entity_present[name] = True
        for e_name in ["humans", "plants", "tables", "laptops", "walls"]:
            if e_name not in obs.keys():
                self.is_entity_present[e_name] = self.env.get_padded_observations
                continue
            e_obs = obs[e_name].reshape(-1, self.env.entity_obs_dim)
            partial_obs = np.array([], dtype=np.float32)
            for i in range(e_obs.shape[0]):
                # if the observation is a result of padding, skip it
                if ((e_obs[i][0] == 0.0) and (e_obs[i][1] == 0.0) and (e_obs[i][2] == 0.0) and (e_obs[i][3] == 0.0) and (e_obs[i][4] == 0.0) and (e_obs[i][5] == 0.0)):
                    continue
                if self.lies_in_range(e_obs[i]): partial_obs = np.concatenate((
                    partial_obs, e_obs[i]
                )).flatten()
            
            if e_name == "humans":
                self.num_humans = partial_obs.shape[0]//self.env.entity_obs_dim
            elif e_name == "plants":
                self.num_plants = partial_obs.shape[0]//self.env.entity_obs_dim
            elif e_name == "tables":
                self.num_tables = partial_obs.shape[0]//self.env.entity_obs_dim
            elif e_name == "laptops":
                self.num_laptops = partial_obs.shape[0]//self.env.entity_obs_dim
            elif e_name == "walls":
                self.num_walls = partial_obs.shape[0]//self.env.entity_obs_dim

        if not self.env.get_padded_observations and self.num_humans == 0: self.is_entity_present["humans"] = False            
        if not self.env.get_padded_observations and self.num_tables == 0: self.is_entity_present["tables"] = False            
        if not self.env.get_padded_observations and self.num_laptops == 0: self.is_entity_present["laptops"] = False            
        if not self.env.get_padded_observations and self.num_plants == 0: self.is_entity_present["plants"] = False            
        if not self.env.get_padded_observations and self.num_walls > 0: self.is_entity_present["walls"] = True  
        else: self.is_entity_present["walls"] = False            


        d = {

            "robot": spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -self.ROBOT_RADIUS], dtype=np.float32), 
                    high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), self.ROBOT_RADIUS], dtype=np.float32),
                    shape=((self.robot.one_hot_encoding.shape[0]+3, )),
                    dtype=np.float32

                )
        }

        if self.is_entity_present["humans"]:
            d["humans"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.HUMAN_DIAMETER/2, -(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -2*np.pi/self.TIMESTEP, 0] * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.num_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.HUMAN_DIAMETER/2, +(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +2*np.pi/self.TIMESTEP, 1] * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.num_humans), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.num_humans),)),
                dtype=np.float32
            )
        if self.is_entity_present["laptops"]:
            d["laptops"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.LAPTOP_RADIUS, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * ((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.num_laptops)), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.LAPTOP_RADIUS, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * ((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.num_laptops)), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.num_laptops)),)),
                dtype=np.float32

            )

        if self.is_entity_present["tables"]:
            d["tables"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.TABLE_RADIUS, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * (self.MAX_TABLES if self.get_padded_observations else self.num_tables), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.TABLE_RADIUS, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * (self.MAX_TABLES if self.get_padded_observations else self.num_tables), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*(self.MAX_TABLES if self.get_padded_observations else self.num_tables),)),
                dtype=np.float32

            )

        if self.is_entity_present["plants"]:
            d["plants"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.PLANT_RADIUS, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * (self.MAX_PLANTS if self.get_padded_observations else self.num_plants), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.PLANT_RADIUS, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * (self.MAX_PLANTS if self.get_padded_observations else self.num_plants), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*(self.MAX_PLANTS if self.get_padded_observations else self.num_plants),)),
                dtype=np.float32

            )

        if not self.get_padded_observations:
            total_segments = 0
            for w in self.walls:
                total_segments += w.length//self.WALL_SEGMENT_SIZE
                if w.length % self.WALL_SEGMENT_SIZE != 0: total_segments += 1
            if self.is_entity_present["walls"]:
                d["walls"] = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.WALL_SEGMENT_SIZE, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * self.num_walls, dtype=np.float32),
                    high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, +self.WALL_SEGMENT_SIZE, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * self.num_walls, dtype=np.float32),
                    shape=(((self.robot.one_hot_encoding.shape[0] + 8)*self.num_walls,)),
                    dtype=np.float32
                )

        return spaces.Dict(d)

    def lies_in_range(self, obs):
        """
        Checks whether the observation lies within -fov/2 to +fov/2 and within range

        Args:
        obs : an entity observation. Shape should be (self.env.entity_obs_dim, )
        """
        assert(obs.shape == (self.env.entity_obs_dim,)), "Wrong shape"
        if (np.arctan2(obs[7], obs[6]) >= -self.fov_angle/2) and (np.arctan2(obs[7], obs[6])<= self.fov_angle/2) and (np.linalg.norm([obs[6], obs[7]]) <= self.range):
            return True
        else:
            return False
        
    def lies_in_range_using_coordinates(self, x, y):
        """
        Checks whether the observation lies within -fov/2 to +fov/2 and within range

        Args:
        x : x-coordinate of entity in world frame
        y : y-coordinate of entity in world frame
        """
        coord = self.env.get_robot_frame_coordinates(np.array([[x, y]])).flatten()
        if (np.arctan2(coord[1], coord[0]) >= -self.fov_angle/2) and (np.arctan2(coord[1], coord[0])<= self.fov_angle/2) and (np.linalg.norm([coord[0], coord[1]]) <= self.range):
            return True
        else:
            return False

    def get_partial_observation(self, obs):
        d = {}
        d["robot"] = obs["robot"]
        for entity_name in ["humans", "plants", "tables", "laptops"]:
            if entity_name not in obs.keys(): continue
            o = obs[entity_name].reshape(-1, self.env.entity_obs_dim)
            partial_obs = np.array([], dtype=np.float32)
            for i in range(o.shape[0]):
                # if the observation is a result of padding, skip it
                if ((o[i][0] == 0.0) and (o[i][1] == 0.0) and (o[i][2] == 0.0) and (o[i][3] == 0.0) and (o[i][4] == 0.0) and (o[i][5] == 0.0)):
                    continue
                # if the observation lies in the frame of view, add it to the observation
                if self.lies_in_range(o[i]):
                    partial_obs = np.concatenate(
                        (partial_obs, o[i])
                    ).flatten()
                
            d[entity_name] = partial_obs
            if entity_name == "humans":
                self.num_humans = partial_obs.shape[0]//self.env.entity_obs_dim
            elif entity_name == "plants":
                self.num_plants = partial_obs.shape[0]//self.env.entity_obs_dim
            elif entity_name == "tables":
                self.num_tables = partial_obs.shape[0]//self.env.entity_obs_dim
            elif entity_name == "laptops":
                self.num_laptops = partial_obs.shape[0]//self.env.entity_obs_dim
        
        if "walls" in obs.keys():
            o = obs["walls"].reshape(-1, self.env.entity_obs_dim)
            partial_obs = np.array([], dtype=np.float32)
            for i in range(o.shape[0]):
                if self.lies_in_range(o[i]):
                    partial_obs = np.concatenate(
                        (partial_obs, o[i])
                    ).flatten()
            d["walls"] = partial_obs
            self.num_walls = partial_obs.shape[0]//self.env.entity_obs_dim

        if self.num_humans == 0 and "humans" in d.keys(): d.pop("humans") 
        if self.num_plants == 0 and "plants" in d.keys(): d.pop("plants") 
        if self.num_laptops == 0 and "laptops" in d.keys(): d.pop("laptops") 
        if self.num_tables == 0 and "tables" in d.keys(): d.pop("tables") 
        if self.num_walls == 0 and "walls" in d.keys(): d.pop("walls") 
        return d
    
    def get_interaction_info(self):
        human_human = []
        human_laptop = []
        curr_humans = 0
        curr_laptops = 0

        for h in self.env.static_humans + self.env.dynamic_humans:
            if self.lies_in_range_using_coordinates(h.x, h.y): curr_humans += 1
        
        for l in self.env.laptops:
            if self.lies_in_range_using_coordinates(l.x, l.y): curr_laptops += 1

        for i, interaction in enumerate(self.env.moving_interactions + self.env.static_interactions):
            interaction_indices = []
            count_of_humans = 0
            
            curr_count = 0
            for human in interaction.humans:
                if self.lies_in_range_using_coordinates(human.x, human.y):
                    interaction_indices.append(curr_humans + curr_count)
                    count_of_humans += 1
                    curr_count += 1
            
            for p in range(len(interaction_indices)):
                for q in range(p+1, len(interaction_indices)):
                    human_human.append((interaction_indices[p], interaction_indices[q]))
                    human_human.append((interaction_indices[q], interaction_indices[p]))
            
            curr_humans += count_of_humans
        curr_count = 0
        curr_laptop_count = 0
        for i, interaction in enumerate(self.env.h_l_interactions):
            if self.lies_in_range_using_coordinates(interaction.human.x, interaction.human.y) and self.lies_in_range_using_coordinates(interaction.laptop.x, interaction.laptop.y):
                human_laptop.append((curr_humans + curr_count, curr_laptops + curr_laptop_count))

            if self.lies_in_range_using_coordinates(interaction.human.x, interaction.human.y): curr_count += 1
            if self.lies_in_range_using_coordinates(interaction.laptop.x, interaction.laptop.y): curr_laptop_count += 1
        return human_human, human_laptop
    
    def pad_observations(self, obs):
        """
        pads the observations with 0s, so that all observations are of the same shape
        """
        # padding with zeros
        for entity_name in ["humans", "plants", "tables", "laptops"]:
            if entity_name in obs:
                obs[entity_name] = np.concatenate((obs[entity_name], np.zeros(self.observation_space[entity_name].shape[0] - obs[entity_name].shape[0])), dtype=np.float32)


    def step(self, action_pre):
        obs, reward, terminated, truncated, info = self.env.step(action_pre)
        self.latest_obs = obs
        if self.env.get_padded_observations: self.pad_observations(obs)
        obs = self.get_partial_observation(obs)
        human_human, human_laptop = self.get_interaction_info()
        info["interactions"]["human-human"] = human_human
        info["interactions"]["human-laptop"] = human_laptop
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.latest_obs = obs
        if self.env.get_padded_observations: self.pad_observations(obs)
        obs = self.get_partial_observation(obs)
        return obs, info
    
    def render(self, mode="human"):
        if not self.env.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(self.env.RESOLUTION_VIEW), int(self.env.RESOLUTION_VIEW))
            self.env.window_initialised = True
            self.env.world_image = (np.ones((int(self.env.RESOLUTION_Y),int(self.env.RESOLUTION_X),3))*255).astype(np.uint8)

        self.env.world_image = (np.ones((int(self.env.RESOLUTION_Y),int(self.env.RESOLUTION_X),3))*255).astype(np.uint8)
        
        self.env.robot.draw_range(
            self.env.world_image,
            self.range,
            self.fov_angle,
            self.env.PIXEL_TO_WORLD_X,
            self.env.PIXEL_TO_WORLD_Y,
            self.env.MAP_X,
            self.env.MAP_Y    
        )
        
        for wall in self.env.walls:
            wall.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)

        for table in self.env.tables:
            table.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)

        for laptop in self.env.laptops:
            laptop.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)
        
        for plant in self.env.plants:
            plant.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)

        cv2.circle(self.env.world_image, (w2px(self.env.robot.goal_x, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X), w2py(self.env.robot.goal_y, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_Y)), int(w2px(self.env.robot.x + self.env.GOAL_RADIUS, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X) - w2px(self.env.robot.x, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X)), (0, 255, 0), 2)
        
        for human in self.env.dynamic_humans:  # only draw goals for the dynamic humans
            cv2.circle(self.env.world_image, (w2px(human.goal_x, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X), w2py(human.goal_y, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_Y)), int(w2px(human.x + self.env.HUMAN_GOAL_RADIUS, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X) - w2px(human.x, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X)), (120, 0, 0), 2)
        
        for i in self.env.moving_interactions:
            cv2.circle(self.env.world_image, (w2px(i.goal_x, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X), w2py(i.goal_y, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_Y)), int(w2px(i.x + i.goal_radius, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X) - w2px(i.x, self.env.PIXEL_TO_WORLD_X, self.env.MAP_X)), (0, 0, 255), 2)
        
        for human in self.env.static_humans + self.env.dynamic_humans:
            human.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)
        
        self.env.robot.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)

        for i in (self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions):
            i.draw(self.env.world_image, self.env.PIXEL_TO_WORLD_X, self.env.PIXEL_TO_WORLD_Y, self.env.MAP_X, self.env.MAP_Y)

        ## uncomment to save the images 
        # cv2.imwrite("img"+str(self.env.count)+".jpg", self.env.world_image)
        # self.env.count+=1

        cv2.imshow("world", self.env.world_image)
        k = cv2.waitKey(self.env.MILLISECONDS)
        if k%255 == 27:
            sys.exit(0)
        

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self.env)
        obs, reward, terminated, truncated, info = env_copy.step(action_pre)
        obs = self.get_partial_observation(obs)
        if self.env.get_padded_observations: self.pad_observations(obs)
        del env_copy
        return obs, reward, terminated, truncated, info
