import gym
from gym import spaces
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.utils.wall import Wall
import numpy as np
import copy

class WorldFrameObservations(gym.Wrapper):
    def __init__(self, env:SocNavEnv_v1) -> None:
        super().__init__(env)
        self.env = env
        
    @property
    def observation_space(self):
        """
        Observation space includes the goal, and the world frame coordinates and speeds (linear & angular) of all the objects (including the robot) in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {

            "robot": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2, -self.env.MAP_X/2, -self.env.MAP_Y/2, -1.0, -1.0, -self.env.MAX_ADVANCE_ROBOT, -self.env.MAX_ADVANCE_ROBOT, -self.env.MAX_ROTATION, -self.env.ROBOT_RADIUS], dtype=np.float32), 
                high=np.array([1, 1, 1, 1, 1, 1, self.env.MAP_X/2 , self.env.MAP_Y/2, self.env.MAP_X/2, self.env.MAP_Y/2, 1.0, 1.0, self.env.MAX_ADVANCE_ROBOT, self.env.MAX_ADVANCE_ROBOT, self.env.MAX_ROTATION, self.env.ROBOT_RADIUS], dtype=np.float32),
                shape=((self.env.robot.one_hot_encoding.shape[0]+10, )),
                dtype=np.float32

            )
        }

        if self.env.is_entity_present["humans"]:
            d["humans"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.HUMAN_DIAMETER/2, -self.env.MAX_ADVANCE_HUMAN, -self.env.MAX_ADVANCE_HUMAN, 0] * ((self.env.MAX_HUMANS + (self.env.MAX_H_L_INTERACTIONS + self.env.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.env.MAX_H_H_DYNAMIC_INTERACTIONS + self.env.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.env.MAX_H_H_STATIC_INTERACTIONS + self.env.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.env.get_padded_observations else self.env.total_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.HUMAN_DIAMETER/2, +self.env.MAX_ADVANCE_HUMAN, +self.env.MAX_ADVANCE_HUMAN, 1.0] * ((self.env.MAX_HUMANS + (self.env.MAX_H_L_INTERACTIONS + self.env.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.env.MAX_H_H_DYNAMIC_INTERACTIONS + self.env.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.env.MAX_H_H_STATIC_INTERACTIONS + self.env.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.env.get_padded_observations else self.env.total_humans), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 8) * ((self.env.MAX_HUMANS + (self.env.MAX_H_L_INTERACTIONS + self.env.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.env.MAX_H_H_DYNAMIC_INTERACTIONS + self.env.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.env.MAX_H_H_STATIC_INTERACTIONS + self.env.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.env.get_padded_observations else self.env.total_humans),)),
                dtype=np.float32
            )

        if self.env.is_entity_present["laptops"]:
            d["laptops"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.LAPTOP_RADIUS, -0.05, -0.05, 0] * ((self.env.MAX_LAPTOPS + (self.env.MAX_H_L_INTERACTIONS + self.env.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.env.get_padded_observations else (self.env.NUMBER_OF_LAPTOPS + (self.env.NUMBER_OF_H_L_INTERACTIONS + self.env.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING))), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.LAPTOP_RADIUS, 0.05, 0.05, 1.0] * ((self.env.MAX_LAPTOPS + (self.env.MAX_H_L_INTERACTIONS + self.env.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.env.get_padded_observations else (self.env.NUMBER_OF_LAPTOPS + (self.env.NUMBER_OF_H_L_INTERACTIONS + self.env.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING))), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 8)*((self.env.MAX_LAPTOPS + (self.env.MAX_H_L_INTERACTIONS + self.env.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.env.get_padded_observations else (self.env.NUMBER_OF_LAPTOPS + (self.env.NUMBER_OF_H_L_INTERACTIONS + self.env.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING))),)),
                dtype=np.float32

            )

        if self.env.is_entity_present["tables"]:
            d["tables"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.TABLE_RADIUS, -0.05, -0.05, 0] * (self.env.MAX_TABLES if self.env.get_padded_observations else self.env.NUMBER_OF_TABLES), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.TABLE_RADIUS, 0.05, 0.05, 1.0] * (self.env.MAX_TABLES if self.env.get_padded_observations else self.env.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 8)*(self.env.MAX_TABLES if self.env.get_padded_observations else self.env.NUMBER_OF_TABLES),)),
                dtype=np.float32

            )

        if self.env.is_entity_present["plants"]:
            d["plants"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.PLANT_RADIUS, -0.05, -0.05, 0] * (self.env.MAX_PLANTS if self.env.get_padded_observations else self.env.NUMBER_OF_PLANTS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.PLANT_RADIUS, 0.05, 0.05, 1.0] * (self.env.MAX_PLANTS if self.env.get_padded_observations else self.env.NUMBER_OF_PLANTS), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 8)*(self.env.MAX_PLANTS if self.env.get_padded_observations else self.env.NUMBER_OF_PLANTS),)),
                dtype=np.float32

            )


        if not self.env.get_padded_observations:
            total_segments = 0
            for w in self.env.walls:
                total_segments += w.length//self.env.WALL_SEGMENT_SIZE
                if w.length % self.env.WALL_SEGMENT_SIZE != 0: total_segments += 1
            
            if self.env.is_entity_present["walls"]:
                d["walls"] = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.WALL_SEGMENT_SIZE, -0.05, -0.05, 0] * int(total_segments), dtype=np.float32),
                    high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, +self.env.WALL_SEGMENT_SIZE, 0.05, 0.05, 1.0] * int(total_segments), dtype=np.float32),
                    shape=(((self.env.robot.one_hot_encoding.shape[0] + 8)*int(total_segments),)),
                    dtype=np.float32
                )

        return spaces.Dict(d)


    def _get_entity_obs(self, object): 
            """
            Returning the observation for one individual object. Also to get the sin and cos of the angle rather than the angle itself.
            Input:
                object (one of socnavenv.envs.utils.object.Object's subclasses) : the object of interest
            Returns:
                numpy.ndarray : the observations of the given object.
            """
            # checking the coordinates and orientation of the object are not None
            assert((object.x is not None) and (object.y is not None) and (object.orientation is not None)), f"{object.name}'s coordinates or orientation are None type"

            def _get_wall_obs(wall:Wall, size:float):
                centers = []
                lengths = []

                left_x = wall.x - wall.length/2 * np.cos(wall.orientation)
                left_y = wall.y - wall.length/2 * np.sin(wall.orientation)

                right_x = wall.x + wall.length/2 * np.cos(wall.orientation)
                right_y = wall.y + wall.length/2 * np.sin(wall.orientation)

                segment_x = left_x + np.cos(wall.orientation)*(size/2)
                segment_y = left_y + np.sin(wall.orientation)*(size/2)

                for i in range(int(wall.length//size)):
                    centers.append((segment_x, segment_y))
                    lengths.append(size)
                    segment_x += np.cos(wall.orientation)*size
                    segment_y += np.sin(wall.orientation)*size

                if(wall.length % size != 0):
                    length = wall.length % size
                    centers.append((right_x - np.cos(wall.orientation)*length/2, right_y - np.sin(wall.orientation)*length/2))
                    lengths.append(length)
                
                obs = np.array([], dtype=np.float32)
                
                for center, length in zip(centers, lengths):
                    # wall encoding
                    obs = np.concatenate((obs, wall.one_hot_encoding))
                    # coorinates of the wall
                    obs = np.concatenate((obs, np.array([[center[0], center[1]]]).flatten()))
                    # sin and cos of angles
                    obs = np.concatenate((obs, np.array([np.sin(wall.orientation), np.cos(wall.orientation)])))
                    # radius of the wall = length/2
                    obs = np.concatenate((obs, np.array([length/2])))
                    # speeds based
                    if self.robot.type == "diff-drive":
                        relative_speeds = np.array([0, 0], dtype=np.float32)
                    elif self.robot.type == "holonomic":
                        relative_speeds = np.array([0, 0], dtype=np.float32)
                    else: raise NotImplementedError
                    obs = np.concatenate((obs, relative_speeds))
                    # gaze for walls is 0
                    obs = np.concatenate((obs, np.array([0])))
                    obs = obs.flatten().astype(np.float32)                
                return obs

            # if it is a wall, then return the observation
            if object.name == "wall":
                return _get_wall_obs(object, self.WALL_SEGMENT_SIZE)

            # initializing output array
            output = np.array([], dtype=np.float32)
            
            # object's one-hot encoding
            output = np.concatenate(
                (
                    output,
                    object.one_hot_encoding
                ),
                dtype=np.float32
            )

            # object's coordinates in the robot frame
            output = np.concatenate(
                        (
                            output,
                            np.array([[object.x, object.y]]).flatten() 
                        ),
                        dtype=np.float32
                    )

            # sin and cos of the relative angle of the object
            output = np.concatenate(
                        (
                            output,
                            np.array([np.sin(object.orientation), np.cos(object.orientation)]) 
                        ),
                        dtype=np.float32
                    )
            
            # object's radius
            radius = 0
            if object.name == "plant":
                radius = object.radius
            elif object.name == "human":
                radius = object.width/2
            elif object.name == "table" or object.name == "laptop":
                radius = np.sqrt((object.length/2)**2 + (object.width/2)**2)
            else: raise NotImplementedError

            output = np.concatenate(
                (
                    output,
                    np.array([radius], dtype=np.float32)
                ),
                dtype=np.float32
            )

            # relative speeds for static objects
            speeds = np.array([0.0, 0.0], dtype=np.float32) 
            
            if object.name == "human": # the only dynamic object
                speeds[0] = object.speed * np.cos(object.orientation)
                speeds[1] = object.speed * np.sin(object.orientation)
            
            output = np.concatenate(
                        (
                            output,
                            speeds
                        ),
                        dtype=np.float32
                    )

            # adding gaze
            gaze = 0

            if object.name == "human":
                robot_in_human_frame = self.get_human_frame_coordinates(object, np.array([[self.robot.x, self.robot.y]])).flatten()
                robot_x = robot_in_human_frame[0]
                robot_y = robot_in_human_frame[1]

                if np.arctan2(robot_y, robot_x) >= -self.HUMAN_GAZE_ANGLE/2 and np.arctan2(robot_y, robot_x)<= self.HUMAN_GAZE_ANGLE/2:
                    gaze = 1

            output = np.concatenate(
                        (
                            output,
                            np.array([gaze])
                        ),
                        dtype=np.float32
                    )
            
            assert(self.entity_obs_dim == output.flatten().shape[-1]), "The value of self.entity_obs_dim needs to be changed"
            return output.flatten()

    def _get_world_frame_obs(self):
        """
        Used to get the observations in the robot frame

        Returns:
            numpy.ndarray : observation as described in the observation space.
        """

        # the observations will go inside this dictionary
        d = {}
        
        # goal coordinates in the robot frame
        robot_obs = (np.array([
            self.env.robot.goal_x,
            self.env.robot.goal_y, 
            self.env.robot.x, 
            self.env.robot.y,
            np.sin(self.env.robot.orientation),
            np.cos(self.env.robot.orientation),
            self.env.robot.vel_x,
            self.env.robot.vel_y,
            self.env.robot.vel_a
        ], dtype=np.float32))
        
        # converting into the required shape
        robot_obs = robot_obs.flatten()

        # concatenating with the robot's one-hot-encoding
        robot_obs = np.concatenate((self.env.robot.one_hot_encoding, robot_obs), dtype=np.float32)
        # adding radius to robot observation
        robot_obs = np.concatenate((robot_obs, np.array([self.env.ROBOT_RADIUS])), dtype=np.float32).flatten()
        # placing it in a dictionary
        d["robot"] = robot_obs
        
        # getting the observations of humans
        human_obs = np.array([], dtype=np.float32)
        for human in self.static_humans + self.dynamic_humans:
            obs = self._get_entity_obs(human)
            human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
        
        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            if i.name == "human-human-interaction":
                for human in i.humans:
                    obs = self._get_entity_obs(human)
                    human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
            elif i.name == "human-laptop-interaction":
                obs = self._get_entity_obs(i.human)
                human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
       
        if self.get_padded_observations:
            # padding with zeros
            human_obs = np.concatenate((human_obs, np.zeros(self.observation_space["humans"].shape[0] - human_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        if self.env.is_entity_present["humans"]:
            d["humans"] = human_obs

    
        # getting the observations of laptops
        laptop_obs = np.array([], dtype=np.float32)
        for laptop in self.laptops:
            obs = self._get_entity_obs(laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
        
        for i in self.h_l_interactions:
            obs = self._get_entity_obs(i.laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
       
        if self.get_padded_observations:
            # padding with zeros
            laptop_obs = np.concatenate((laptop_obs, np.zeros(self.observation_space["laptops"].shape[0] -laptop_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        if self.env.is_entity_present["laptops"]:
            d["laptops"] = laptop_obs
    

        # getting the observations of tables
        table_obs = np.array([], dtype=np.float32)
        for table in self.tables:
            obs = self._get_entity_obs(table)
            table_obs = np.concatenate((table_obs, obs), dtype=np.float32)

        if self.get_padded_observations:
            # padding with zeros
            table_obs = np.concatenate((table_obs, np.zeros(self.observation_space["tables"].shape[0] -table_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        if self.env.is_entity_present["tables"]:
            d["tables"] = table_obs


        # getting the observations of plants
        plant_obs = np.array([], dtype=np.float32)
        for plant in self.plants:
            obs = self._get_entity_obs(plant)
            plant_obs = np.concatenate((plant_obs, obs), dtype=np.float32)

        if self.get_padded_observations:
            # padding with zeros
            plant_obs = np.concatenate((plant_obs, np.zeros(self.observation_space["plants"].shape[0] -plant_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        if self.env.is_entity_present["plants"]:
            d["plants"] = plant_obs

        # inserting wall observations to the dictionary
        if not self.get_padded_observations:
            wall_obs = np.array([], dtype=np.float32)
            for wall in self.walls:
                obs = self._get_entity_obs(wall)
                wall_obs = np.concatenate((wall_obs, obs), dtype=np.float32)
            if self.env.is_entity_present["walls"]:
                d["walls"] = wall_obs

        return d

    def step(self, action_pre):
        _, reward, terminated, truncated, info = self.env.step(action_pre)
        obs = self._get_world_frame_obs()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed)
        obs = self._get_world_frame_obs()
        return obs, info

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self.env)
        next_state, reward, terminated, truncated, info = env_copy.step(action_pre)
        current_env = copy.deepcopy(self.env)
        self.env = copy.deepcopy(env_copy)
        next_state = self._get_world_frame_obs()
        self.env = copy.deepcopy(current_env)
        del current_env
        del env_copy
        return next_state, reward, terminated, truncated, info