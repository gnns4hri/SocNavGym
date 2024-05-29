import os
import random
import sys
import time
from math import atan2
from typing import List, Dict
import copy
from importlib.machinery import SourceFileLoader
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import rvo2
import torch
import yaml
from gym import spaces
from shapely.geometry import Point, Polygon
from collections import namedtuple
import math
from math import ceil
from enum import Enum
EntityObs = namedtuple("EntityObs", ["id", "x", "y", "theta", "sin_theta", "cos_theta"])

from socnavgym.envs.utils.human import Human
from socnavgym.envs.utils.human_human import Human_Human_Interaction
from socnavgym.envs.utils.human_laptop import Human_Laptop_Interaction
from socnavgym.envs.utils.laptop import Laptop
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.plant import Plant
from socnavgym.envs.utils.robot import Robot
from socnavgym.envs.utils.table import Table
from socnavgym.envs.utils.utils import (get_coordinates_of_rotated_rectangle,
                                        get_nearest_point_from_rectangle,
                                        get_square_around_circle,
                                        convert_angle_to_minus_pi_to_pi,
                                        compute_time_to_collision,
                                        point_to_segment_dist, w2px, w2py)
from socnavgym.envs.utils.wall import Wall
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/utils/sngnnv2")
from socnavgym.envs.utils.sngnnv2.socnav import SocNavDataset
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import Human as otherHuman
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import Object as otherObject
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import SNScenario, SocNavAPI

DEBUG = 0
if 'debug' in sys.argv or "debug=2" in sys.argv:
    DEBUG = 2
elif "debug=1" in sys.argv:
    DEBUG = 1

MAX_ORIENTATION_CHANGE = math.pi/2.    

class SocNavGymObject(Enum):
    ROBOT = "ROBOT"
    DYNAMIC_HUMAN = "DYNAMIC_HUMAN"
    STATIC_HUMAN = "STATIC_HUMAN"
    PLANT = "PLANT"
    TABLE = "TABLE"
    LAPTOP = "LAPTOP"
    WALL = "WALL"
    HUMAN_HUMAN_INTERACTION_DYNAMIC = "HUMAN_HUMAN_INTERACTION_DYNAMIC"
    HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING = "HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING"
    HUMAN_HUMAN_INTERACTION_STATIC = "HUMAN_HUMAN_INTERACTION_STATIC"
    HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING = "HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING"
    HUMAN_LAPTOP_INTERACTION = "HUMAN_LAPTOP_INTERACTION"
    HUMAN_LAPTOP_INTERACTION_NON_DISPERSING = "HUMAN_LAPTOP_INTERACTION_NON_DISPERSING"

    def get_class(self):
        if self == SocNavGymObject.ROBOT: return Robot
        if self == SocNavGymObject.DYNAMIC_HUMAN: return Human
        if self == SocNavGymObject.STATIC_HUMAN: return Human
        if self == SocNavGymObject.PLANT: return Plant
        if self == SocNavGymObject.TABLE: return Table
        if self == SocNavGymObject.LAPTOP: return Laptop
        if self == SocNavGymObject.WALL: return Wall
        if self == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC: return Human_Human_Interaction
        if self == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING: return Human_Human_Interaction
        if self == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC: return Human_Human_Interaction
        if self == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING: return Human_Human_Interaction
        if self == SocNavGymObject.HUMAN_LAPTOP_INTERACTION: return Human_Laptop_Interaction
        if self == SocNavGymObject.HUMAN_LAPTOP_INTERACTION_NON_DISPERSING: return Human_Laptop_Interaction

class SocNavEnv_v1(gym.Env):
    """
    Class for the environment
    """
    metadata = {"render_modes": ["human", "rgb_array"],"render_fps": 4}
    
    # rendering params
    RESOLUTION_VIEW = None
    MILLISECONDS = None

    # episode params
    EPISODE_LENGTH = None
    TIMESTEP = None
    END_WITH_COLLISION = None

    # rewards
    REWARD_PATH = None

    # robot params
    ROBOT_RADIUS = None
    GOAL_RADIUS = None
    ROBOT_TYPE = None

    # human params
    HUMAN_DIAMETER = None
    HUMAN_GOAL_RADIUS = None
    HUMAN_POLICY=None
    HUMAN_GAZE_ANGLE=None

    # laptop params
    LAPTOP_WIDTH=None
    LAPTOP_LENGTH=None
    
    # plant params
    PLANT_RADIUS =None

    # table params
    TABLE_WIDTH = None
    TABLE_LENGTH = None

    # wall params
    WALL_THICKNESS = None

    # human-human interaction params
    INTERACTION_RADIUS = None
    INTERACTION_GOAL_RADIUS = None
    INTERACTION_NOISE_VARIANCE = None

    # human-laptop interaction params
    HUMAN_LAPTOP_DISTANCE = None

    def __init__(self, config:str=None) -> None:
        """
        Args : 
            config: Path to the environment config file
        """
        super().__init__()
        
        assert(config is not None), "Argument config_path is None. Please call gym.make(\"SocNavGym-v1\", config_path=path_to_config)"

        self.window_initialised = False
        self.has_configured = False
        # the number of steps taken in the current episode
        self.ticks = 0  
        # static humans in the environment
        self.static_humans:List[Human] = [] 
        # dynamic humans in the environment
        self.dynamic_humans:List[Human] = []
        # laptops in the environment
        self.laptops:List[Laptop] = [] 
        # walls in the environment
        self.walls:List[Wall] = []  
        # plants in the environment
        self.plants:List[Plant] = []  
        # tables in the environment
        self.tables:List[Table] = []  
        # dynamic interactions
        self.moving_interactions:List[Human_Human_Interaction] = []
        # static interactions
        self.static_interactions:List[Human_Human_Interaction] = []
        # human-laptop-interactions
        self.h_l_interactions:List[Human_Laptop_Interaction] = []

        
        # robot
        self.robot:Robot = None

        # environment parameters
        self.MARGIN = None
        self.MAX_ADVANCE_HUMAN = None
        self.MAX_ROTATION_HUMAN = None
        self.MAX_ADVANCE_ROBOT = None
        self.MAX_ROTATION = None
        self.SPEED_THRESHOLD = None
        
        # wall segment size
        self.WALL_SEGMENT_SIZE = None


        # defining the bounds of the number of entities
        self.MIN_STATIC_HUMANS = None
        self.MAX_STATIC_HUMANS = None 
        self.MIN_DYNAMIC_HUMANS = None
        self.MAX_DYNAMIC_HUMANS = None
        self.MAX_HUMANS = None
        self.MIN_TABLES = None
        self.MAX_TABLES = None
        self.MIN_PLANTS = None
        self.MAX_PLANTS = None
        self.MIN_LAPTOPS = None
        self.MAX_LAPTOPS = None
        self.MIN_H_H_DYNAMIC_INTERACTIONS = None
        self.MAX_H_H_DYNAMIC_INTERACTIONS = None
        self.MIN_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING = None
        self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING = None
        self.MIN_H_H_STATIC_INTERACTIONS = None
        self.MAX_H_H_STATIC_INTERACTIONS = None
        self.MIN_H_H_STATIC_INTERACTIONS_NON_DISPERSING = None
        self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING = None
        self.MIN_HUMAN_IN_H_H_INTERACTIONS = None
        self.MAX_HUMAN_IN_H_H_INTERACTIONS = None
        self.MIN_H_L_INTERACTIONS = None
        self.MAX_H_L_INTERACTIONS = None
        self.MIN_H_L_INTERACTIONS_NON_DISPERSING = None
        self.MAX_H_L_INTERACTIONS_NON_DISPERSING = None
        self.MIN_MAP_X = None
        self.MAX_MAP_X = None
        self.MIN_MAP_Y = None
        self.MIN_MAP_Y = None
        self.CROWD_DISPERSAL_PROBABILITY = None
        self.HUMAN_LAPTOP_DISPERSAL_PROBABILITY = None
        self.CROWD_FORMATION_PROBABILITY = None
        self.HUMAN_LAPTOP_FORMATION_PROBABILITY = None

        self.PROB_TO_AVOID_ROBOT = None  # probability that the human would consider the human while calculating it's velocity
        self.HUMAN_FOV = None

        # flag parameter that controls whether padded observations will be returned or not
        self.get_padded_observations = None

        # to check if the episode has finished
        self._is_terminated = True
        self._is_truncated = True
        
        # for rendering the world to an OpenCV image
        self.world_image = None
        
        # parameters for integrating multiagent particle environment's forces

        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # shape of the environment
        self.set_shape = None
        self.shape = None

        # rewards
        self.prev_distance = None

        # some shapes for general use

        # dimension of an entity observation
        self.entity_obs_dim = 14  
        # dimension of robot observation
        self.robot_obs_dim = 9

        # contact response parameters taken from OpenAI's multiagent particle environment
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # img list in case of recording videos
        self.img_list = None

        # cuda device
        self.cuda_device = None

        # configuring the environment parameters
        self._configure(config)


    def _configure(self, config_path):
        """
        To read from config file to set env parameters
        
        Args:
            config_path(str): path to config file
        """
        # loading config file
        with open(config_path, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        self.has_configured = True

        # resolution view
        self.RESOLUTION_VIEW = config["rendering"]["resolution_view"]
        assert(self.RESOLUTION_VIEW > 0), "resolution view should be greater than 0"
        
        # milliseconds
        self.MILLISECONDS = config["rendering"]["milliseconds"]
        assert(self.MILLISECONDS > 0), "milliseconds should be greater than zero"

        # episode parameters
        self.EPISODE_LENGTH = config["episode"]["episode_length"]
        assert(self.EPISODE_LENGTH > 0), "episode length should be greater than 0"
        self.TIMESTEP = config["episode"]["time_step"]
        self.END_WITH_COLLISION = config["episode"]["end_with_collision"]

        # robot
        self.ROBOT_RADIUS = config["robot"]["robot_radius"]
        self.GOAL_RADIUS = config["robot"]["goal_radius"]
        assert(self.ROBOT_RADIUS > 0 and self.GOAL_RADIUS > 0), "robot parameters in config file should be greater than 0"
        self.GOAL_THRESHOLD = self.ROBOT_RADIUS + self.GOAL_RADIUS
        self.ROBOT_TYPE = config["robot"]["robot_type"]
        assert(self.ROBOT_TYPE == "diff-drive" or self.ROBOT_TYPE == "holonomic")

        # human
        self.HUMAN_DIAMETER = config["human"]["human_diameter"]
        self.HUMAN_GOAL_RADIUS = config["human"]["human_goal_radius"]
        self.HUMAN_POLICY = config["human"]["human_policy"]
        assert(self.HUMAN_POLICY=="random" or self.HUMAN_POLICY == "orca" or self.HUMAN_POLICY == "sfm"), "human_policy should be \"random\", or \"orca\" or \"sfm\""
        self.HUMAN_GAZE_ANGLE = config["human"]["gaze_angle"]
        self.PROB_TO_AVOID_ROBOT = config["human"]["prob_to_avoid_robot"]
        self.HUMAN_FOV = config["human"]["fov_angle"]

        # laptop
        self.LAPTOP_WIDTH = config["laptop"]["laptop_width"]
        self.LAPTOP_LENGTH = config["laptop"]["laptop_length"]
        self.LAPTOP_RADIUS = np.sqrt((self.LAPTOP_LENGTH/2)**2 + (self.LAPTOP_WIDTH/2)**2)

        # plant
        self.PLANT_RADIUS = config["plant"]["plant_radius"]

        # table
        self.TABLE_WIDTH = config["table"]["table_width"]
        self.TABLE_LENGTH = config["table"]["table_length"]
        self.TABLE_RADIUS = np.sqrt((self.TABLE_LENGTH/2)**2 + (self.TABLE_WIDTH/2)**2)

        # wall
        self.WALL_THICKNESS = config["wall"]["wall_thickness"]

        # human-human-interaction
        self.INTERACTION_RADIUS = config["human-human-interaction"]["interaction_radius"]
        self.INTERACTION_GOAL_RADIUS = config["human-human-interaction"]["interaction_goal_radius"]
        self.INTERACTION_NOISE_VARIANCE = config["human-human-interaction"]["noise_variance"]

        # human-laptop-interaction
        self.HUMAN_LAPTOP_DISTANCE = config["human-laptop-interaction"]["human_laptop_distance"]
        
        # env
        self.MARGIN = config["env"]["margin"]
        self.MAX_ADVANCE_HUMAN = config["env"]["max_advance_human"]
        self.MAX_ROTATION_HUMAN = config["env"]["max_rotation_human"]
        self.MAX_ADVANCE_ROBOT = config["env"]["max_advance_robot"]
        self.MAX_ROTATION = config["env"]["max_rotation"]
        self.WALL_SEGMENT_SIZE = config["env"]["wall_segment_size"]
        self.SPEED_THRESHOLD = config["env"]["speed_threshold"]

        self.MIN_STATIC_HUMANS = config["env"]["min_static_humans"]
        self.MAX_STATIC_HUMANS = config["env"]["max_static_humans"]
        assert(self.MIN_STATIC_HUMANS <= self.MAX_STATIC_HUMANS), "min_static_humans should be less than or equal to max_static_humans"

        self.MIN_DYNAMIC_HUMANS = config["env"]["min_dynamic_humans"]
        self.MAX_DYNAMIC_HUMANS = config["env"]["max_dynamic_humans"]
        assert(self.MIN_DYNAMIC_HUMANS <= self.MAX_DYNAMIC_HUMANS), "min_dynamic_humans should be less than or equal to max_dynamic_humans"

        self.MAX_HUMANS = self.MAX_STATIC_HUMANS + self.MAX_DYNAMIC_HUMANS

        self.MIN_TABLES = config["env"]["min_tables"]
        self.MAX_TABLES = config["env"]["max_tables"]
        assert(self.MIN_TABLES <= self.MAX_TABLES), "min_tables should be less than or equal to max_tables"
        
        self.MIN_PLANTS = config["env"]["min_plants"]
        self.MAX_PLANTS = config["env"]["max_plants"]
        assert(self.MIN_PLANTS <= self.MAX_PLANTS), "min_plants should be less than or equal to max_plants"

        self.MIN_LAPTOPS = config["env"]["min_laptops"]
        self.MAX_LAPTOPS = config["env"]["max_laptops"]
        assert(self.MIN_LAPTOPS <= self.MAX_LAPTOPS), "min_laptops should be less than or equal to max_laptops"

        self.MIN_H_H_DYNAMIC_INTERACTIONS = config["env"]["min_h_h_dynamic_interactions"]
        self.MAX_H_H_DYNAMIC_INTERACTIONS = config["env"]["max_h_h_dynamic_interactions"]
        assert(self.MIN_H_H_DYNAMIC_INTERACTIONS <= self.MAX_H_H_DYNAMIC_INTERACTIONS)

        self.MIN_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING = config["env"]["min_h_h_dynamic_interactions_non_dispersing"]
        self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING = config["env"]["max_h_h_dynamic_interactions_non_dispersing"]
        assert(self.MIN_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING <= self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)

        self.MIN_H_H_STATIC_INTERACTIONS = config["env"]["min_h_h_static_interactions"]
        self.MAX_H_H_STATIC_INTERACTIONS = config["env"]["max_h_h_static_interactions"]
        assert(self.MIN_H_H_STATIC_INTERACTIONS <= self.MAX_H_H_STATIC_INTERACTIONS)

        self.MIN_H_H_STATIC_INTERACTIONS_NON_DISPERSING = config["env"]["min_h_h_static_interactions_non_dispersing"]
        self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING = config["env"]["max_h_h_static_interactions_non_dispersing"]
        assert(self.MIN_H_H_STATIC_INTERACTIONS_NON_DISPERSING <= self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)

        self.MIN_HUMAN_IN_H_H_INTERACTIONS = config["env"]["min_human_in_h_h_interactions"]
        self.MAX_HUMAN_IN_H_H_INTERACTIONS = config["env"]["max_human_in_h_h_interactions"]
        assert(self.MIN_HUMAN_IN_H_H_INTERACTIONS <= self.MAX_HUMAN_IN_H_H_INTERACTIONS), "min_human_in_h_h_interactions should be less than or equal to max_human_in_h_h_interactions"

        self.MIN_H_L_INTERACTIONS = config["env"]["min_h_l_interactions"]
        self.MAX_H_L_INTERACTIONS = config["env"]["max_h_l_interactions"]
        assert(self.MIN_H_L_INTERACTIONS <= self.MAX_H_L_INTERACTIONS), "min_h_l_interactions should be lesser than or equal to max_h_l_interactions"
        
        self.MIN_H_L_INTERACTIONS_NON_DISPERSING = config["env"]["min_h_l_interactions_non_dispersing"]
        self.MAX_H_L_INTERACTIONS_NON_DISPERSING = config["env"]["max_h_l_interactions_non_dispersing"]
        assert(self.MIN_H_L_INTERACTIONS_NON_DISPERSING <= self.MAX_H_L_INTERACTIONS_NON_DISPERSING), "min_h_l_interactions_non_dispersing should be lesser than or equal to max_h_l_interactions_non_dispersing"

        self.get_padded_observations = config["env"]["get_padded_observations"]
        assert(self.get_padded_observations == True or self.get_padded_observations == False), "get_padded_observations should be either True or False"

        self.set_shape = config["env"]["set_shape"]
        assert(self.set_shape == "random" or self.set_shape == "square" or self.set_shape == "rectangle" or self.set_shape == "L" or self.set_shape == "no-walls"), "set shape can be \"random\", \"square\", \"rectangle\", \"L\", or \"no-walls\""

        self.add_corridors = config["env"]["add_corridors"]
        assert(type(self.add_corridors) == bool)

        self.MIN_MAP_X = config["env"]["min_map_x"]
        self.MAX_MAP_X = config["env"]["max_map_x"]
        self.MIN_MAP_Y = config["env"]["min_map_y"]
        self.MAX_MAP_Y = config["env"]["max_map_y"]

        self.CROWD_DISPERSAL_PROBABILITY = config["env"]["crowd_dispersal_probability"]
        assert(self.CROWD_DISPERSAL_PROBABILITY >= 0 and self.CROWD_DISPERSAL_PROBABILITY <= 1.0), "Probability should be within [0, 1]"

        self.HUMAN_LAPTOP_DISPERSAL_PROBABILITY = config["env"]["human_laptop_dispersal_probability"]
        assert(self.HUMAN_LAPTOP_DISPERSAL_PROBABILITY >= 0 and self.HUMAN_LAPTOP_DISPERSAL_PROBABILITY <= 1.0), "Probability should be within [0, 1]"

        self.CROWD_FORMATION_PROBABILITY = config["env"]["crowd_formation_probability"]
        assert(self.CROWD_FORMATION_PROBABILITY >= 0 and self.CROWD_FORMATION_PROBABILITY <= 1.0), "Probability should be within [0, 1]"

        self.HUMAN_LAPTOP_FORMATION_PROBABILITY = config["env"]["human_laptop_formation_probability"]
        assert(self.HUMAN_LAPTOP_FORMATION_PROBABILITY >= 0 and self.HUMAN_LAPTOP_FORMATION_PROBABILITY <= 1.0), "Probability should be within [0, 1]"
        
        # cuda device
        self.cuda_device = config["env"]["cuda_device"]

        # reward
        self.REWARD_PATH = config["env"]["reward_file"]
        if self.REWARD_PATH == "sngnn":
            self.REWARD_PATH = os.path.dirname(os.path.abspath(__file__)) + "/rewards/" + "sngnn_reward.py"
        elif self.REWARD_PATH == "dsrnn":
            self.REWARD_PATH = os.path.dirname(os.path.abspath(__file__)) + "/rewards/" + "dsrnn_reward.py"
        
        module, self.REWARD_PATH = self.process_reward_path(self.REWARD_PATH)
        reward_module = SourceFileLoader(module, self.REWARD_PATH).load_module()
        reward_api_class = SourceFileLoader(module, self.REWARD_PATH).load_module().RewardAPI
        try:
            self.reward_class = reward_module.Reward
        except AttributeError:
            print(f"No class named Reward found in {self.REWARD_PATH}. Please name your reward function class as Reward!")
            sys.exit(0)
        
        assert(issubclass(self.reward_class, reward_api_class)), "Please make Reward class a subclass of RewardAPI class"
        self.reward_calculator = self.reward_class(self)
        self.reset()

    def process_reward_path(self, path:str):
        if not ".py" in path: path += ".py"
        if not os.path.exists(path):
            raise FileNotFoundError("Path to the reward file not found!")

        l = path.split("/")
        return l[-1][:3], path

    def set_padded_observations(self, val:bool):
        """
        To assign True/False to the parameter get_padded_observations. True will indicate that padding will be done. Else padding 
        Args: val (bool): True/False value that would enable/disable padding in the observations received henceforth 
        """
        self.get_padded_observations = val

    def randomize_params(self):
        """
        To randomly initialize the number of entities of each type. Specifically, this function would initialize the MAP_SIZE, NUMBER_OF_HUMANS, NUMBER_OF_PLANTS, NUMBER_OF_LAPTOPS and NUMBER_OF_TABLES
        """
        self.MAP_X = random.randint(self.MIN_MAP_X, self.MAX_MAP_X)
        
        if self.shape == "square" or self.shape == "L":
            self.MAP_Y = self.MAP_X
        else :
            self.MAP_Y = random.randint(self.MIN_MAP_Y, self.MAX_MAP_Y)
        
        self.RESOLUTION_X = int(1850 * self.MAP_X/(self.MAP_X + self.MAP_Y))
        self.RESOLUTION_Y = int(1850 * self.MAP_Y/(self.MAP_X + self.MAP_Y))
        self.NUMBER_OF_STATIC_HUMANS = random.randint(self.MIN_STATIC_HUMANS, self.MAX_STATIC_HUMANS)  # number of static humans in the env
        self.NUMBER_OF_DYNAMIC_HUMANS = random.randint(self.MIN_DYNAMIC_HUMANS, self.MAX_DYNAMIC_HUMANS)  # number of static humans in the env
        self.NUMBER_OF_PLANTS = random.randint(self.MIN_PLANTS, self.MAX_PLANTS)  # number of plants in the env
        self.NUMBER_OF_TABLES = random.randint(self.MIN_TABLES, self.MAX_TABLES)  # number of tables in the env
        self.NUMBER_OF_LAPTOPS = random.randint(self.MIN_LAPTOPS, self.MAX_LAPTOPS)  # number of laptops in the env. Laptops will be sampled on tables
        self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS = random.randint(self.MIN_H_H_DYNAMIC_INTERACTIONS, self.MAX_H_H_DYNAMIC_INTERACTIONS) # number of dynamic human-human interactions
        self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING = random.randint(self.MIN_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING, self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING) # number of dynamic human-human interactions that do not disperse
        self.NUMBER_OF_H_H_STATIC_INTERACTIONS = random.randint(self.MIN_H_H_STATIC_INTERACTIONS, self.MAX_H_H_STATIC_INTERACTIONS) # number of static human-human interactions
        self.NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING = random.randint(self.MIN_H_H_STATIC_INTERACTIONS_NON_DISPERSING, self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING) # number of static human-human interactions that do not disperse
        self.humans_in_h_h_dynamic_interactions = []
        self.humans_in_h_h_static_interactions = []
        self.humans_in_h_h_dynamic_interactions_non_dispersing = []
        self.humans_in_h_h_static_interactions_non_dispersing = []
        for _ in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS):
            self.humans_in_h_h_dynamic_interactions.append(random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS):
            self.humans_in_h_h_static_interactions.append(random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING):
            self.humans_in_h_h_dynamic_interactions_non_dispersing.append(random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING):
            self.humans_in_h_h_static_interactions_non_dispersing.append(random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))

        self.NUMBER_OF_H_L_INTERACTIONS = random.randint(self.MIN_H_L_INTERACTIONS, self.MAX_H_L_INTERACTIONS) # number of human laptop interactions
        self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING = random.randint(self.MIN_H_L_INTERACTIONS_NON_DISPERSING, self.MAX_H_L_INTERACTIONS_NON_DISPERSING) # number of human laptop interactions that do not disperse
        self.TOTAL_H_L_INTERACTIONS = self.NUMBER_OF_H_L_INTERACTIONS + self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING

        # total humans
        self.total_humans = self.NUMBER_OF_STATIC_HUMANS + self.NUMBER_OF_DYNAMIC_HUMANS
        for i in self.humans_in_h_h_dynamic_interactions: self.total_humans += i
        for i in self.humans_in_h_h_static_interactions: self.total_humans += i
        for i in self.humans_in_h_h_dynamic_interactions_non_dispersing: self.total_humans += i
        for i in self.humans_in_h_h_static_interactions_non_dispersing: self.total_humans += i
        self.total_humans += self.TOTAL_H_L_INTERACTIONS
        # randomly select the shape
        if self.set_shape == "random":
            self.shape = random.choice(["rectangle", "square", "L"])
        else: self.shape = self.set_shape
        

        # adding Gaussian Noise to ORCA parameters
        self.orca_neighborDist = 2*self.HUMAN_DIAMETER + np.random.randn()
        self.orca_timeHorizon = 5 + np.random.randn()
        self.orca_timeHorizonObst = 5 + np.random.randn()
        self.orca_maxSpeed = self.MAX_ADVANCE_HUMAN + np.random.randn()*0.01

        # adding Gaussian Noise to SFM parameters
        self.sfm_r0 = abs(0.05 + np.random.randn()*0.01)
        self.sfm_gamma = 0.25 + np.random.randn()*0.01
        self.sfm_n = 1 + np.random.randn()*0.1
        self.sfm_n_prime = 1 + np.random.randn()*0.1
        self.sfm_lambd = 1 + np.random.randn()*0.1

    @property
    def PIXEL_TO_WORLD_X(self):
        return self.RESOLUTION_X / self.MAP_X

    @property
    def PIXEL_TO_WORLD_Y(self):
        return self.RESOLUTION_Y / self.MAP_Y
    
    @property
    def MAX_OBSERVATION_LENGTH(self):
        return (self.MAX_HUMANS + self.MAX_LAPTOPS + self.MAX_PLANTS + self.MAX_TABLES) * self.entity_obs_dim + 8
    
    @property
    def observation_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {

            "robot": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -self.ROBOT_RADIUS], dtype=np.float32), 
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), self.ROBOT_RADIUS], dtype=np.float32),
                shape=((self.robot_obs_dim, )),
                dtype=np.float32

            )
        }

        if self.is_entity_present["humans"]:
            d["humans"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.HUMAN_DIAMETER/2, -(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -2*np.pi/self.TIMESTEP, 0] * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.HUMAN_DIAMETER/2, +(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +2*np.pi/self.TIMESTEP, 1] * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * ((self.MAX_HUMANS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + ((self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans),)),
                dtype=np.float32
            )

        if self.is_entity_present["laptops"]:
            d["laptops"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.LAPTOP_RADIUS, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * ((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.LAPTOP_RADIUS, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * ((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*((self.MAX_LAPTOPS + (self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)),)),
                dtype=np.float32

            )

        if self.is_entity_present["tables"]:
            d["tables"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.TABLE_RADIUS, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.TABLE_RADIUS, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*(self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES),)),
                dtype=np.float32

            )

        if self.is_entity_present["plants"]:
            d["plants"] =  spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.PLANT_RADIUS, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.PLANT_RADIUS, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8)*(self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS),)),
                dtype=np.float32

            )

        if not self.get_padded_observations:
            total_segments = 0
            for w in self.walls:
                total_segments += w.length//self.WALL_SEGMENT_SIZE
                if w.length % self.WALL_SEGMENT_SIZE != 0: total_segments += 1
            if self.is_entity_present["walls"]:
                d["walls"] = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.WALL_SEGMENT_SIZE, -(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), -self.MAX_ROTATION, 0] * int(total_segments), dtype=np.float32),
                    high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, +self.WALL_SEGMENT_SIZE, +(self.MAX_ADVANCE_ROBOT)*np.sqrt(2), +self.MAX_ROTATION, 1] * int(total_segments), dtype=np.float32),
                    shape=(((self.robot.one_hot_encoding.shape[0] + 8)*int(total_segments),)),
                    dtype=np.float32
                )

        return spaces.Dict(d)

    @property
    def action_space(self): # continuous action space 
        """Returns the action space of the robot. The action space contains three parameters, the linear velocity, perpendicular velocity, and the angular velocity. The values lie in the range of [-1, 1]. Velocities are obtained by mapping from [-1,1] to [-MAX_ADVANCE_ROBOT, +MAX_ADVANCE_ROBOT] for linear and perpendicular velocities, to [-self.MAX_ROTATION, +self.MAX_ROTATION] for angular velocity

        Returns:
            gym.spaces.Box: The action space
        """
        if self.robot.type == "holonomic":
            low  = np.array([-1, -1, -1], dtype=np.float32)
            high = np.array([+1, +1, +1], dtype=np.float32)
        
        elif self.robot.type == "diff-drive":  # lateral speed is 0 for differential drive robots
            low  = np.array([-1, 0, -1], dtype=np.float32)
            high = np.array([+1, 0, +1], dtype=np.float32)
        
        else: raise NotImplementedError
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def done(self):
        """
        Indicates whether the episode has finished
        Returns:
        bool: True if episode has finished, and False if the episode has not finished.
        """
        return self._is_terminated or self._is_truncated

    @property
    def transformation_matrix(self):
        """
        The transformation matrix that can convert coordinates from the global frame to the robot frame. This is calculated by inverting the transformation from the world frame to the robot frame
        
        That is,
        np.linalg.inv([[cos(theta)    -sin(theta)      h],
                       [sin(theta)     cos(theta)      k],
                       [0              0               1]]) where h, k are the coordinates of the robot in the global frame and theta is the angle of the X-axis of the robot frame with the X-axis of the global frame

        Note that the above matrix is invertible since the determinant is always 1.

        Returns:
        numpy.ndarray : the transformation matrix to convert coordinates from the world frame to the robot frame.
        """
        # check if the coordinates and orientation are not None
        assert(self.robot.x is not None and self.robot.y is not None and self.robot.orientation is not None), "Robot coordinates or orientation are None type"
        # initalizing the matrix
        tm = np.zeros((3,3), dtype=np.float32)
        # filling values as described
        tm[2,2] = 1
        tm[0,2] = self.robot.x
        tm[1,2] = self.robot.y
        tm[0,0] = tm[1,1] = np.cos(self.robot.orientation)
        tm[1,0] = np.sin(self.robot.orientation)
        tm[0,1] = -1*np.sin(self.robot.orientation)

        return np.linalg.inv(tm)
    
    @property
    def is_entity_present(self)->Dict[str, bool]:
        """
        Returns a dictionary that contains the keys as the entity types, and the values for each key would be a boolean value specifying whether the entity is present in the environment or not.
        """
        if not self.has_configured: raise Exception("Environment not configured")
        d = {}
        d["robot"] = True
        d["humans"] = True
        d["laptops"] = True
        d["tables"] = True
        d["plants"] = True
        d["walls"] = True

        if (not self.get_padded_observations) and (self.total_humans == 0): d["humans"] = False
        if (not self.get_padded_observations) and (self.NUMBER_OF_PLANTS == 0): d["plants"] = False
        if (not self.get_padded_observations) and (self.NUMBER_OF_TABLES == 0): d["tables"] = False
        if (not self.get_padded_observations) and ((self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS) == 0): d["laptops"] = False
        if (self.shape == "no-walls"): d["walls"] = False

        return d

    def human_transformation_matrix(self, human:Human):
        """
        The transformation matrix that can convert coordinates from the global frame to the human frame. This is calculated by inverting the transformation from the world frame to the human frame
        
        That is,
        np.linalg.inv([[cos(theta)    -sin(theta)      h],
                       [sin(theta)     cos(theta)      k],
                       [0              0               1]]) where h, k are the coordinates of the human in the global frame and theta is the angle of the X-axis of the human frame with the X-axis of the global frame

        Note that the above matrix is invertible since the determinant is always 1.

        Returns:
        numpy.ndarray : the transformation matrix to convert coordinates from the world frame to the human frame.
        """
        # check if the coordinates and orientation are not None
        assert(human.x is not None and human.y is not None and human.orientation is not None), "Human coordinates or orientation are None type"
        # initalizing the matrix
        tm = np.zeros((3,3), dtype=np.float32)
        # filling values as described
        tm[2,2] = 1
        tm[0,2] = human.x
        tm[1,2] = human.y
        tm[0,0] = tm[1,1] = np.cos(human.orientation)
        tm[1,0] = np.sin(human.orientation)
        tm[0,1] = -1*np.sin(human.orientation)

        return np.linalg.inv(tm)

    def get_robot_frame_coordinates(self, coord):
        """
        Given coordinates in the world frame, this method returns the corresponding robot frame coordinates.
        Args:
            coord (numpy.ndarray) :  coordinate input in the world frame expressed as np.array([[x,y]]). If there are multiple coordinates, then give input as 2-D array with shape (no. of points, 2).
        Returns:
            numpy.ndarray : Coordinates in the robot frame. Shape is same as the input shape.

        """
        # converting the coordinates to homogeneous coordinates
        homogeneous_coordinates = np.c_[coord, np.ones((coord.shape[0], 1))]
        # getting the robot frame coordinates by multiplying with the transformation matrix
        coord_in_robot_frame = (self.transformation_matrix@homogeneous_coordinates.T).T
        return coord_in_robot_frame[:, 0:2]
    
    def get_human_frame_coordinates(self, human, coord):
        """
        Given coordinates in the world frame, this method returns the corresponding human frame coordinates.
        Args:
            coord (numpy.ndarray) :  coordinate input in the world frame expressed as np.array([[x,y]]). If there are multiple coordinates, then give input as 2-D array with shape (no. of points, 2).
        Returns:
            numpy.ndarray : Coordinates in the human frame. Shape is same as the input shape.

        """
        # converting the coordinates to homogeneous coordinates
        homogeneous_coordinates = np.c_[coord, np.ones((coord.shape[0], 1))]
        # transformation matrix of the human
        transformation_matrix = self.human_transformation_matrix(human)
        # getting the human frame coordinates by multiplying with the transformation matrix
        coord_in_robot_frame = (transformation_matrix@homogeneous_coordinates.T).T
        return coord_in_robot_frame[:, 0:2]

    def is_entity_visible_in_human_frame(self, human:Human, entity:Object):
        # return True if the frame of view is very close to 2*pi
        if abs(human.fov - 2*(np.pi)) < 0.1: return True
        
        sector_points = []
        start_angle = human.orientation - human.fov/2
        end_angle = human.orientation + human.fov/2
        radius = 0
        for p in [(self.MAP_X/2, self.MAP_Y/2), (-self.MAP_X/2, self.MAP_Y/2), (self.MAP_X/2, -self.MAP_Y/2), (-self.MAP_X/2, -self.MAP_Y/2)]:
            radius = max(radius, np.linalg.norm([human.x-p[0], human.y-p[1]]))
        
        # hardcoded resolution of 1000 points
        for steps in range(1001):
            angle = ((end_angle-start_angle)/1000)*steps + start_angle
            sector_points.append([human.x + radius*np.cos(angle), human.y + radius*np.sin(angle)])
        
        # append the center of the sector
        sector_points.append([human.x, human.y])

        if entity.name == "plant" or entity.name == "robot":
            assert(entity.x != None and entity.y != None and entity.radius != None), "Attributes are None type"
            other_obj = Point((entity.x, entity.y)).buffer(entity.radius)
        
        elif entity.name == "human":
            assert(entity.x != None and entity.y != None and entity.width != None), "Attributes are None type"
            other_obj = Point((entity.x, entity.y)).buffer(entity.width/2)
        
        elif entity.name == "laptop" or entity.name == "table":
            assert(entity.x != None and entity.y != None and entity.width != None and entity.length != None and entity.orientation != None), "Attributes are None type"
            other_obj = Polygon(get_coordinates_of_rotated_rectangle(entity.x, entity.y, entity.orientation, entity.length, entity.width))

        elif entity.name == "wall":
            assert(entity.x != None and entity.y != None and entity.thickness != None and entity.length != None and entity.orientation != None), "Attributes are None type"
            other_obj = Polygon(get_coordinates_of_rotated_rectangle(entity.x, entity.y, entity.orientation, entity.length, entity.thickness))

        elif entity.name == "human-human-interaction":
            other_obj = Point((entity.x, entity.y)).buffer(entity.radius)
        
        elif entity.name == "human-laptop-interaction":
            other_obj = Point((entity.x, entity.y)).buffer(entity.distance/2)
        
        else: raise NotImplementedError

        sector = Polygon(sector_points)
        # if it lies within the range, then it would intersect the sector
        return other_obj.intersects(sector)

    def _get_entity_obs(self, object): 
        """
        Returning the observation for one individual object. Also to get the sin and cos of the relative angle rather than the angle itself.
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
                obs = np.concatenate((obs, self.get_robot_frame_coordinates(np.array([[center[0], center[1]]])).flatten()))
                # sin and cos of relative angles
                obs = np.concatenate((obs, np.array([(np.sin(wall.orientation - self.robot.orientation)), np.cos(wall.orientation - self.robot.orientation)])))
                # radius of the wall = length/2
                obs = np.concatenate((obs, np.array([length/2])))
                # relative speeds based on robot type
                relative_speeds = np.array([-np.sqrt(self.robot.vel_x**2 + self.robot.vel_y**2), -self.robot.vel_a], dtype=np.float32)
                obs = np.concatenate((obs, relative_speeds))
                # gaze for walls is 0
                obs = np.concatenate((obs, np.array([0.0])))
                obs = obs.flatten().astype(np.float32)
            
            wall_coordinates = self.get_robot_frame_coordinates(np.array([[wall.x, wall.y]])).flatten()
            self._current_observations[wall.id] = EntityObs(
                wall.id,
                wall_coordinates[0],
                wall_coordinates[1],
                wall.orientation - self.robot.orientation,
                np.sin(wall.orientation - self.robot.orientation),
                np.cos(wall.orientation - self.robot.orientation)
            )

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
                        self.get_robot_frame_coordinates(np.array([[object.x, object.y]])).flatten() 
                    ),
                    dtype=np.float32
                )

        # sin and cos of the relative angle of the object
        output = np.concatenate(
                    (
                        output,
                        np.array([(np.sin(object.orientation - self.robot.orientation)), np.cos(object.orientation - self.robot.orientation)]) 
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

        robot_vel_x = self.robot.vel_x * np.cos(self.robot.orientation) + self.robot.vel_y * np.cos(self.robot.orientation + np.pi/2)
        robot_vel_y = self.robot.vel_x * np.sin(self.robot.orientation) + self.robot.vel_y * np.sin(self.robot.orientation + np.pi/2)

        # relative speeds for static objects
        relative_speeds = np.array([-np.sqrt(self.robot.vel_x**2 + self.robot.vel_y**2), -self.robot.vel_a], dtype=np.float32)
        
        if object.name == "human": # the only dynamic object
            if object.type == "static":
                assert object.speed <= 0.001, "static human has speed" 
            # relative linear speed
            relative_speeds[0] = np.sqrt((object.speed*np.cos(object.orientation) - robot_vel_x)**2 + (object.speed*np.sin(object.orientation) - robot_vel_y)**2) 
            relative_speeds[1] = (np.arctan2(np.sin(object.orientation - self.robot.orientation), np.cos(object.orientation - self.robot.orientation)) - self._prev_observations[object.id].theta) / self.TIMESTEP
        
        output = np.concatenate(
                    (
                        output,
                        relative_speeds
                    ),
                    dtype=np.float32
                )

        # adding gaze
        gaze = 0.0

        if object.name == "human":
            robot_in_human_frame = self.get_human_frame_coordinates(object, np.array([[self.robot.x, self.robot.y]])).flatten()
            robot_x = robot_in_human_frame[0]
            robot_y = robot_in_human_frame[1]

            if np.arctan2(robot_y, robot_x) >= -self.HUMAN_GAZE_ANGLE/2 and np.arctan2(robot_y, robot_x)<= self.HUMAN_GAZE_ANGLE/2:
                gaze = 1.0

        output = np.concatenate(
                    (
                        output,
                        np.array([gaze])
                    ),
                    dtype=np.float32
                )
                
        output = output.flatten()
        self._current_observations[object.id] = EntityObs(
            object.id,
            output[6],
            output[7],
            np.arctan2(output[8], output[9]),
            output[8],
            output[9]
        )
        assert(self.entity_obs_dim == output.flatten().shape[-1]), "The value of self.entity_obs_dim needs to be changed"
        return output.flatten()

    def _get_obs(self):
        """
        Used to get the observations in the robot frame

        Returns:
            numpy.ndarray : observation as described in the observation space.
        """

        # the observations will go inside this dictionary
        d = {}
        
        # goal coordinates in the robot frame
        goal_in_robot_frame = self.get_robot_frame_coordinates(np.array([[self.robot.goal_x, self.robot.goal_y]], dtype=np.float32))
        # converting into the required shape
        robot_obs = goal_in_robot_frame.flatten()

        # concatenating with the robot's one-hot-encoding
        robot_obs = np.concatenate((self.robot.one_hot_encoding, robot_obs), dtype=np.float32)
        
        # adding the radius of the robot to the robot's observation
        robot_obs = np.concatenate((robot_obs, np.array([self.ROBOT_RADIUS], dtype=np.float32))).flatten()

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
        if self.is_entity_present["humans"]:
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
        if self.is_entity_present["laptops"]:
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
        if self.is_entity_present["tables"]:
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
        if self.is_entity_present["plants"]:
            d["plants"] = plant_obs

        # inserting wall observations to the dictionary
        if not self.get_padded_observations:
            wall_obs = np.array([], dtype=np.float32)
            for wall in self.walls:
                obs = self._get_entity_obs(wall)
                wall_obs = np.concatenate((wall_obs, obs), dtype=np.float32)
            if self.is_entity_present["walls"]:
                d["walls"] = wall_obs

        return d
    
    def get_desired_force(self, human:Human):
        e_d = np.array([(human.goal_x - human.x), (human.goal_y - human.y)], dtype=np.float32)
        if np.linalg.norm(e_d) != 0:
            e_d /= np.linalg.norm(e_d)
        f_d = self.MAX_ADVANCE_HUMAN * e_d
        return f_d

    def get_obstacle_force(self, human:Human, obstacle:Object, r0):
        # perpendicular distance
        distance = 0

        if obstacle.name == "plant" or obstacle.name=="robot":
            distance = np.sqrt((obstacle.x - human.x)**2 + (obstacle.y - human.y)**2) - obstacle.radius - human.width/2
            e_o = np.array([human.x - obstacle.x, human.y - obstacle.y])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
        
        elif obstacle.name == "human-human-interaction":
            distance = np.sqrt((obstacle.x - human.x)**2 + (obstacle.y - human.y)**2) - obstacle.radius - human.width/2
            e_o = np.array([human.x - obstacle.x, human.y - obstacle.y])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
        
        elif obstacle.name == "table" or obstacle.name == "laptop":
            px, py = get_nearest_point_from_rectangle(obstacle.x, obstacle.y, obstacle.length, obstacle.width, obstacle.orientation, human.x, human.y)      
            e_o = np.array([human.x - px, human.y - py])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
            distance = np.sqrt((human.x-px)**2 + (human.y-py)**2) - human.width/2
        
        elif obstacle.name == "wall":
            px, py = get_nearest_point_from_rectangle(obstacle.x, obstacle.y, obstacle.length, obstacle.thickness, obstacle.orientation, human.x, human.y)      
            e_o = np.array([human.x - px, human.y - py])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
            distance = np.sqrt((human.x-px)**2 + (human.y-py)**2) - human.width/2

        else : raise NotImplementedError

        f_o = np.exp(-distance/r0) * e_o
        return f_o

    def get_interaction_force(self, human1:Human, human2:Human, gamma, n, n_prime, lambd):
        e_ij = np.array([human2.x - human1.x, human2.y - human1.y])
        if np.linalg.norm(e_ij) != 0:
            e_ij /= np.linalg.norm(e_ij)

        v_ij = np.array([
            (human2.speed * np.cos(human2.orientation)) - (human1.speed * np.cos(human1.orientation)),
            (human2.speed * np.sin(human2.orientation)) - (human1.speed * np.sin(human1.orientation))
        ])

        D_ij = lambd *  v_ij + e_ij
        B = np.linalg.norm(D_ij) * gamma
        if np.linalg.norm(D_ij) != 0:
            t_ij = D_ij/np.linalg.norm(D_ij)
        theta_ij = np.arccos(np.clip(np.dot(e_ij, t_ij), -1, 1))
        n_ij = np.array([-e_ij[1], e_ij[0]])
        d_ij = np.sqrt((human1.x-human2.x)**2 + (human1.y-human2.y)**2)
        f_ij = -np.exp(-d_ij/B) * (np.exp(-((n_prime*B*theta_ij)**2))*t_ij + np.exp(-((n*B*theta_ij)**2))*n_ij)
        return f_ij

    def compute_sfm_velocity(self, human:Human, w1=1/np.sqrt(3), w2=1/np.sqrt(3), w3=1/np.sqrt(3)):
        f = np.array([0, 0], dtype=np.float32)
        f_d = np.zeros(2, dtype=np.float32)
        f_d = self.get_desired_force(human)
        f += w1*f_d

        visible_humans = []
        visible_tables = []
        visible_plants = []
        visible_laptops = []
        visible_h_l_interactions = []
        visible_moving_interactions = []
        visible_static_interactions = []
        # walls are always visible to the human
        visible_walls = []

        # fill in the visible entities
        for h in self.static_humans + self.dynamic_humans:
            # check if it is the same human
            if human.id == h.id: continue
            # if visible append to visible list
            elif self.is_entity_visible_in_human_frame(human, h): visible_humans.append(h)
        
        for plant in self.plants:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, plant): visible_plants.append(plant)
        
        for table in self.tables:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, table): visible_tables.append(table)
        
        for laptop in self.laptops:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, laptop): visible_laptops.append(laptop)

        for wall in self.walls:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, wall): visible_walls.append(wall)
        
        for interaction in self.moving_interactions:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, interaction): visible_moving_interactions.append(interaction)
        
        for interaction in self.static_interactions:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, interaction): visible_static_interactions.append(interaction)
        
        for interaction in self.h_l_interactions:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, interaction): visible_h_l_interactions.append(interaction)

        if human.avoids_robot:
            for obj in visible_plants + visible_walls + visible_tables + visible_laptops + [self.robot] + visible_static_interactions:
                f += w2 * self.get_obstacle_force(human, obj, self.sfm_r0)

        else:
            for obj in visible_plants + visible_walls + visible_tables + visible_laptops + visible_static_interactions:
                f += w2 * self.get_obstacle_force(human, obj, self.sfm_r0)

        for other_human in visible_humans:
            if other_human.id == human.id: continue
            else:
                f += w3 * self.get_interaction_force(human, other_human, self.sfm_gamma, self.sfm_n, self.sfm_n_prime, self.sfm_lambd)

        for i in (visible_moving_interactions + visible_h_l_interactions):
            if i.name == "human-human-interaction":
                for other_human in i.humans:
                    f += w3 * self.get_interaction_force(human, other_human, self.sfm_gamma, self.sfm_n, self.sfm_n_prime, self.sfm_lambd)

            elif i.name == "human-laptop-interaction":
                f += w3 * self.get_interaction_force(human, i.human, self.sfm_gamma, self.sfm_n, self.sfm_n_prime, self.sfm_lambd)
        
        velocity = (f/human.mass) * self.TIMESTEP
        if np.linalg.norm(velocity) > self.MAX_ADVANCE_HUMAN:
            if np.linalg.norm(velocity) != 0:
                velocity /= np.linalg.norm(velocity)
            velocity *= self.MAX_ADVANCE_HUMAN

        return velocity

    def compute_orca_velocity(self, human:Human):
        """
        This method takes in a human object, and computes the velocity using ORCA policy by taking into consideration only the entities that lie in the fov of the human

        Args:
            human (Human): the human whose velocity using ORCA is being calculated 
        Returns:
            The velocity of the human
        """

        # initialising the simulator 
        sim = rvo2.PyRVOSimulator(self.TIMESTEP, self.orca_neighborDist, self.total_humans, self.orca_timeHorizon, self.orca_timeHorizonObst, self.HUMAN_DIAMETER/2, self.orca_maxSpeed)

        # these lists would correspond to the entities that are visible to the human
        visible_humans = []
        visible_tables = []
        visible_plants = []
        visible_laptops = []
        visible_h_l_interactions = []
        visible_moving_interactions = []
        visible_static_interactions = []
        # walls are always visible to the human
        visible_walls = []

        # fill in the visible entities
        for h in self.static_humans + self.dynamic_humans:
            # check if it is the same human
            if human.id == h.id: continue
            # if visible append to visible list
            elif self.is_entity_visible_in_human_frame(human, h): visible_humans.append(h)
        
        for plant in self.plants:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, plant): visible_plants.append(plant)
        
        for table in self.tables:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, table): visible_tables.append(table)
        
        for laptop in self.laptops:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, laptop): visible_laptops.append(laptop)

        for wall in self.walls:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, wall): visible_walls.append(wall)
        
        for interaction in self.moving_interactions:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, interaction): visible_moving_interactions.append(interaction)
        
        for interaction in self.static_interactions:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, interaction): visible_static_interactions.append(interaction)
        
        for interaction in self.h_l_interactions:
            # if visible append to visible list
            if self.is_entity_visible_in_human_frame(human, interaction): visible_h_l_interactions.append(interaction)

        # adding the current human to the simulator
        thisHuman = sim.addAgent((human.x, human.y))
        # preferred velocity is towards the goal
        pref_vel = np.array([human.goal_x-human.x, human.goal_y-human.y], dtype=np.float32)
        # normalising the velocity
        if not np.linalg.norm(pref_vel) == 0:
            pref_vel /= np.linalg.norm(pref_vel)
        pref_vel *= self.MAX_ADVANCE_HUMAN
        # setting the preferred velocity
        sim.setAgentPrefVelocity(thisHuman, (pref_vel[0], pref_vel[1]))

        # adding visible humans as agents
        for human in visible_humans:
            h = sim.addAgent((human.x, human.y))
            # preferred velocity is towards the goal
            pref_vel = np.array([human.goal_x-human.x, human.goal_y-human.y], dtype=np.float32)
            # normalising the velocity
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            # setting the preferred velocity
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))


        # adding robot with a probability of avoiding the robot
        if np.random.random() <= human.prob_to_avoid_robot:
            h = sim.addAgent((self.robot.x, self.robot.y))
            # preferred velocity is towards the goal
            pref_vel = np.array([self.robot.goal_x-self.robot.x, self.robot.goal_y-self.robot.y], dtype=np.float32)
            # normalising the velocity
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_ROBOT
            # setting preferred velocity
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))
        
        # adding visible moving interactions
        for i in visible_moving_interactions:
            h = sim.addAgent((i.x, i.y))
            sim.setAgentRadius(h, self.INTERACTION_RADIUS+self.HUMAN_DIAMETER)
            sim.setAgentNeighborDist(h, 2*(self.INTERACTION_RADIUS + self.HUMAN_DIAMETER))
            pref_vel = np.array([i.goal_x-i.x, i.goal_y-i.y], dtype=np.float32)
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))

        # adding visible obstacles to the simulator
        for obj in visible_tables + visible_laptops + visible_plants + visible_walls:
            p = self.get_obstacle_corners(obj)
            sim.addObstacle(p)

        # adding static and human laptop interactions
        for i in visible_static_interactions + visible_h_l_interactions:
            if i.name == "human-laptop-interaction":
                p = self.get_obstacle_corners(i.human)
                sim.addObstacle(p)

            elif i.name == "human-human-interaction" and i.type == "stationary":
                p = self.get_obstacle_corners(i)
                sim.addObstacle(p)
        
        sim.processObstacles()
        sim.doStep()

        vel = sim.getAgentVelocity(thisHuman)
        del sim
        return vel
    
    def compute_orca_velocity_robot(self, robot:Robot):
        # initialising the simulator 
        sim = rvo2.PyRVOSimulator(self.TIMESTEP, self.orca_neighborDist, self.total_humans, self.orca_timeHorizon, self.orca_timeHorizonObst, self.HUMAN_DIAMETER/2, self.orca_maxSpeed)

        # these lists would correspond to the entities that are visible to the robot
        visible_humans = []
        visible_tables = []
        visible_plants = []
        visible_laptops = []
        visible_h_l_interactions = []
        visible_moving_interactions = []
        visible_static_interactions = []
        visible_walls = []

        # all entities are visible to the robot
        for h in self.static_humans + self.dynamic_humans:
            visible_humans.append(h)
        
        for plant in self.plants:
            visible_plants.append(plant)
        
        for table in self.tables:
            visible_tables.append(table)
        
        for laptop in self.laptops:
            visible_laptops.append(laptop)

        for wall in self.walls:
            visible_walls.append(wall)
        
        for interaction in self.moving_interactions:
            visible_moving_interactions.append(interaction)
        
        for interaction in self.static_interactions:
            visible_static_interactions.append(interaction)
        
        for interaction in self.h_l_interactions:
            visible_h_l_interactions.append(interaction)

        # adding the robot to the simulator
        envRobot = sim.addAgent((robot.x, robot.y))
        # preferred velocity is towards the goal
        pref_vel = np.array([robot.goal_x-robot.x, robot.goal_y-robot.y], dtype=np.float32)
        # normalising the velocity
        if not np.linalg.norm(pref_vel) == 0:
            pref_vel /= np.linalg.norm(pref_vel)
        pref_vel *= self.MAX_ADVANCE_ROBOT
        # setting the preferred velocity
        sim.setAgentPrefVelocity(envRobot, (pref_vel[0], pref_vel[1]))

        # adding visible humans as agents
        for human in visible_humans:
            h = sim.addAgent((human.x, human.y))
            # preferred velocity is towards the goal
            pref_vel = np.array([human.goal_x-human.x, human.goal_y-human.y], dtype=np.float32)
            # normalising the velocity
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            # setting the preferred velocity
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))
        
        # adding visible moving interactions
        for i in visible_moving_interactions:
            h = sim.addAgent((i.x, i.y))
            sim.setAgentRadius(h, self.INTERACTION_RADIUS+self.HUMAN_DIAMETER)
            sim.setAgentNeighborDist(h, 2*(self.INTERACTION_RADIUS + self.HUMAN_DIAMETER))
            pref_vel = np.array([i.goal_x-i.x, i.goal_y-i.y], dtype=np.float32)
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))

        # adding visible obstacles to the simulator
        for obj in visible_tables + visible_laptops + visible_plants + visible_walls:
            p = self.get_obstacle_corners(obj)
            sim.addObstacle(p)

        # adding static and human laptop interactions
        for i in visible_static_interactions + visible_h_l_interactions:
            if i.name == "human-laptop-interaction":
                p = self.get_obstacle_corners(i.human)
                sim.addObstacle(p)

            elif i.name == "human-human-interaction" and i.type == "stationary":
                p = self.get_obstacle_corners(i)
                sim.addObstacle(p)
        
        sim.processObstacles()
        sim.doStep()

        vel = sim.getAgentVelocity(envRobot)
        del sim
        return vel

    def get_obstacle_corners(self, obs:Object):
        if obs.name == "laptop" or obs.name == "table":
            return get_coordinates_of_rotated_rectangle(obs.x, obs.y, obs.orientation, obs.length, obs.width)
        
        elif obs.name == "wall":
            return get_coordinates_of_rotated_rectangle(obs.x, obs.y, obs.orientation, obs.length, obs.thickness)
        
        elif obs.name == "plant" or obs.name == "robot":
            return get_square_around_circle(obs.x, obs.y, obs.radius)
        
        elif obs.name == "human-laptop-interaction":
            return get_square_around_circle(obs.human.x, obs.human.y, self.HUMAN_DIAMETER/2)
        
        elif obs.name == "human":
            return get_square_around_circle(obs.x, obs.y, 2*obs.width)
        
        elif obs.name == "human-human-interaction":
            return get_square_around_circle(obs.x, obs.y, self.INTERACTION_RADIUS)

        else: raise NotImplementedError

    def compute_orca_interaction_velocities(self):
        """
        Returns the velocities of the all the moving interactions 
        """
        sim = rvo2.PyRVOSimulator(self.TIMESTEP, self.orca_neighborDist, self.total_humans, self.orca_timeHorizon, self.orca_timeHorizonObst, self.HUMAN_DIAMETER/2, self.orca_maxSpeed)
        interactionList = []
        for human in self.static_humans + self.dynamic_humans:
            h = sim.addAgent((human.x, human.y))
            pref_vel = np.array([human.goal_x-human.x, human.goal_y-human.y], dtype=np.float32)
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))

        for obj in self.tables + self.laptops + self.plants + self.walls:
            p = self.get_obstacle_corners(obj)
            sim.addObstacle(p)

        for i in (self.static_interactions + self.h_l_interactions):
            if i.name == "human-laptop-interaction":
                p = self.get_obstacle_corners(i.human)
                sim.addObstacle(p)

            elif i.name == "human-human-interaction" and i.type == "stationary":
                p = self.get_obstacle_corners(i)
                sim.addObstacle(p)

        for i in self.moving_interactions:
            h = sim.addAgent((i.x, i.y))
            sim.setAgentRadius(h, self.INTERACTION_RADIUS+self.HUMAN_DIAMETER)
            sim.setAgentNeighborDist(h, 2*(self.INTERACTION_RADIUS + self.HUMAN_DIAMETER))
            pref_vel = np.array([i.goal_x-i.x, i.goal_y-i.y], dtype=np.float32)
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))
            interactionList.append(h)

        sim.processObstacles()
        sim.doStep()
        
        interaction_vels = []
        
        for h in interactionList:
            interaction_vels.append(sim.getAgentVelocity(h))

        return interaction_vels

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        """
        Calculating environment forces  Reference : https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/core.py 
        """
       
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        
        # compute actual distance between entities
        delta_pos = np.array([entity_a.x - entity_b.x, entity_a.y - entity_b.y], dtype=np.float32) 
        # minimum allowable distance
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # calculating the radius based on the entitiy
        if entity_a.name == "plant" or entity_a.name == "robot": # circular shaped
            radius_a = entity_a.radius
        
        # width was assumed as the diameter of the human
        elif entity_a.name == "human": 
            radius_a = entity_a.width/2

        # initialized to 0. Walls are separately handled below
        elif entity_a.name == "wall":  
            radius_a = 0
        
        # approximating the rectangular objects with a circle that circumscribes it
        elif  entity_a.name == "table" or entity_a.name == "laptop":
            radius_a = np.sqrt((entity_a.length/2)**2 + (entity_a.width/2)**2)

        else: raise NotImplementedError

        # similarly calculating for entity b
        if entity_b.name == "plant" or entity_b.name == "robot":
            radius_b = entity_b.radius
        
        elif entity_b.name == "human":
            radius_b = entity_b.width/2
        
        elif entity_b.name == "wall":
            radius_b = 0
        
        elif  entity_b.name == "table" or entity_b.name == "laptop":
            radius_b = np.sqrt((entity_b.length/2)**2 + (entity_b.width/2)**2)
        
        else: raise NotImplementedError
        
        # if one of the entities is a wall, the center is taken to be the reflection of the point in the wall, and radius same as the other entity
        if entity_a.name == "wall":
            if entity_a.orientation == np.pi/2 or entity_a.orientation == -np.pi/2:
                # taking reflection about the striaght line parallel to y axis
                center_x = 2*entity_a.x - entity_b.x  + ((entity_a.thickness) if entity_b.x >= entity_a.x else (-entity_a.thickness))
                center_y = entity_b.y
                delta_pos = np.array([center_x - entity_b.x, center_y - entity_b.y], dtype=np.float32) 
            
            elif entity_a.orientation == 0 or entity_a.orientation == np.pi or entity_a.orientation == -1*np.pi:
                # taking reflection about a striaght line parallel to the x axis
                center_x = entity_b.x
                center_y = 2*entity_a.y - entity_b.y + ((entity_a.thickness) if entity_b.y >= entity_a.y else (-entity_a.thickness))
                delta_pos = np.array([center_x - entity_b.x, center_y - entity_b.y], dtype=np.float32) 

            else : raise NotImplementedError
            # setting the radius of the wall to be the radius of the entity. This is done because the wall's center was assumed to be the reflection of the other entity's center, so now to collide with the wall, it should collide with a circle of the same size.
            radius_a = radius_b
            dist = np.sqrt(np.sum(np.square(delta_pos)))

        elif entity_b.name == "wall":
            if entity_b.orientation == np.pi/2 or entity_b.orientation == -np.pi/2:
                center_x = 2*entity_b.x - entity_a.x  + ((entity_b.thickness) if entity_a.x >= entity_b.x else (-entity_b.thickness))
                center_y = entity_a.y
                delta_pos = np.array([entity_a.x - center_x, entity_a.y - center_y], dtype=np.float32) 
            
            elif entity_b.orientation == 0 or entity_b.orientation == np.pi or entity_b.orientation == -1*np.pi:
                center_x = entity_a.x
                center_y = 2*entity_b.y - entity_a.y + ((entity_b.thickness) if entity_a.y >= entity_b.y else (-entity_b.thickness))
                delta_pos = np.array([entity_a.x - center_x, entity_a.y - center_y], dtype=np.float32) 

            else : raise NotImplementedError

            radius_b = radius_a
            dist = np.sqrt(np.sum(np.square(delta_pos)))

        # minimum distance that is possible between two entities
        dist_min = radius_a + radius_b
        
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if not entity_a.is_static else None  # forces are applied only to dynamic objects
        force_b = -force if not entity_b.is_static else None  # forces are applied only to dynamic objects
        return [force_a, force_b]

    def handle_collision(self):
        all_humans:List[Human] = []
        for human in self.static_humans + self.dynamic_humans: all_humans.append(human)

        for i in self.moving_interactions + self.static_interactions:
            for human in i.humans: all_humans.append(human)
        
        for i in self.h_l_interactions: all_humans.append(i.human)

        for human in all_humans:
            # check collisions with objects
            for object in self.plants + self.walls + self.tables + self.laptops:
                if human.collides(object):
                    [fi, fj] = self.get_collision_force(human, object)
                    entity_vel = (fi / human.mass) * self.TIMESTEP
                    orientation = np.arctan2(entity_vel[1], entity_vel[0])
                    if human.set_new_orientation_with_limits(orientation, MAX_ORIENTATION_CHANGE, self.TIMESTEP):
                        human.speed = min(np.linalg.norm(entity_vel), self.MAX_ADVANCE_HUMAN)
                    else:
                        human.speed = 0
                    human.update(self.TIMESTEP)

    def discrete_to_continuous_action(self, action:int):
        """
        Function to return a continuous space action for a given discrete action
        """
        # Linear vel --> [-1,1]: -1: Stop; 1: Move forward with max velocity
        # Rotational vel --> [-1,1]: -1: Max clockwise rotation; 1: Max anti-clockwise rotation
        # Move forward with max_vel/2 and rotate anti-clockwise with max_rotation/4
        # if action == 0:
        #     return np.array([0, 0.25], dtype=np.float32) 
        # Move forward with max_vel/2 and rotate clockwise with max_rotation/4
        # elif action == 1:
        #     return np.array([0, -0.25], dtype=np.float32) 
        # Move forward with max_vel and rotate anti-clockwise with max_rotation/8
        # elif action == 2:
        #     return np.array([1, 0.125], dtype=np.float32) 
        # Move forward with max_vel and rotate clockwise with max_rotation/8
        # elif action == 3:
        #     return np.array([1, -0.125], dtype=np.float32) 
        # Move forward with max_vel
        # elif action == 4:
        #     return np.array([1, 0], dtype=np.float32)
        # Stop
        # elif action == 5:
        #     return np.array([-1, 0], dtype=np.float32)
        # Move forward with max_vel/10 and rotate anti-clockwise with max_rotation/2.5
        # elif action == 6:
        #     return np.array([-0.8, +0.4], dtype=np.float32)
        # Move forward with max_vel/10 and rotate clockwise with max_rotation/2.5
        # elif action == 7:
        #     return np.array([-0.8, -0.4], dtype=np.float32)
        
        # else:
        #     raise NotImplementedError


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

    
    def step(self, action_pre):
        """Computes a step in the current episode given the action.

        Args:
            action_pre (Union[numpy.ndarray, list]): An action that lies in the action space

        Returns:
            observation (numpy.ndarray) : the observation from the current action
            reward (float) : reward received on the current action
            terminated (bool) : whether the episode has finished or not
            truncated (bool) : whether the episode has finished due to time limit or not
            info (dict) : additional information
        """        
        # for converting the action to the velocity
        def process_action(act):
            """Converts the values from [-1,1] to the corresponding velocity values

            Args:
                act (np.ndarray): action from the action space

            Returns:
                np.ndarray: action with velocity values
            """            
            action = act.astype(np.float32)
            # action[0] = (float(action[0]+1.0)/2.0)*self.MAX_ADVANCE_ROBOT   # [-1, +1] --> [0, self.MAX_ADVANCE_ROBOT]
            action[0] = ((action[0]+0.0)/1.0)*self.MAX_ADVANCE_ROBOT  # [-1, +1] --> [-MAX_ADVANCE, +MAX_ADVANCE]
            if action[1] != 0.0 and self.robot.type == "diff-drive": raise AssertionError("Differential Drive robot cannot have lateral speed")
            action[1] = ((action[1]+0.0)/1.0)*self.MAX_ADVANCE_ROBOT  # [-1, +1] --> [-MAX_ADVANCE, +MAX_ADVANCE]
            action[2] = (float(action[2]+0.0)/1.0)*self.MAX_ROTATION  # [-1, +1] --> [-self.MAX_ROTATION, +self.MAX_ROTATION]
            # if action[0] < 0:               # Advance must be positive
            #     action[0] *= -1
            if action[0] > self.MAX_ADVANCE_ROBOT:     # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[0] = self.MAX_ADVANCE_ROBOT
            if action[0] < -self.MAX_ADVANCE_ROBOT:     # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[0] = -self.MAX_ADVANCE_ROBOT
            if action[1] > self.MAX_ADVANCE_ROBOT:     # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[1] = self.MAX_ADVANCE_ROBOT
            if action[1] < -self.MAX_ADVANCE_ROBOT:     # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[1] = -self.MAX_ADVANCE_ROBOT
            if action[2]   < -self.MAX_ROTATION:   # Rotation must be higher than -self.MAX_ROTATION
                action[2] =  -self.MAX_ROTATION
            elif action[2] > +self.MAX_ROTATION:  # Rotation must be lower than +self.MAX_ROTATION
                action[2] =  +self.MAX_ROTATION
            return action

        # if action is a list, converting it to numpy.ndarray
        if(type(action_pre) == list):
            action_pre = np.array(action_pre, dtype=np.float32)

        # call error if the environment wasn't reset after the episode ended
        if self._is_truncated or self._is_terminated:
            raise Exception('step call within a finished episode!')
    
        # calculating the velocity from action
        action = process_action(action_pre)

        # setting the robot's velocities
        self.robot.vel_x = action[0]        
        self.robot.vel_y = action[1]        
        self.robot.vel_a = action[2]        

        # update robot
        if self._collision:
            future_robot = copy.deepcopy(self.robot)
            future_robot.update(self.TIMESTEP)
            collision_human, collision_object, collision_wall = self.check_robot_collision(future_robot)
            collision = collision_human or collision_object or collision_wall
            execute_action = not collision
        else:
            execute_action = True

        if execute_action:
            self.robot.update(self.TIMESTEP)

        # update dummy robot with orca policy
        if (not self.has_orca_robot_collided) and (not self.has_orca_robot_reached_goal):
            vel = self.compute_orca_velocity_robot(self.robot_orca)
            if self.robot_orca.type == "holonomic":
                vel_x = vel[0] * np.cos(self.robot_orca.orientation) + vel[1] * np.sin(self.robot_orca.orientation)
                vel_y = -vel[0] * np.sin(self.robot_orca.orientation) + vel[1] * np.cos(self.robot_orca.orientation)
                vel_a = (np.arctan2(vel[1], vel[0]) - self.robot_orca.orientation)/self.TIMESTEP
            elif self.robot_orca.type == "diff-drive":
                vel_y = 0
                vel_a = (np.arctan2(vel[1], vel[0]) - self.robot_orca.orientation)/self.TIMESTEP
                vel_x = np.sqrt(vel[0]**2 + vel[1]**2)

            self.robot_orca.vel_x = np.clip(vel_x, -self.MAX_ADVANCE_ROBOT, self.MAX_ADVANCE_ROBOT)
            self.robot_orca.vel_y = np.clip(vel_y, -self.MAX_ADVANCE_ROBOT, self.MAX_ADVANCE_ROBOT)
            self.robot_orca.vel_a = np.clip(vel_a, -self.MAX_ROTATION, self.MAX_ROTATION)
            self.robot_orca.update(self.TIMESTEP)

        # update humans
        interaction_vels = self.compute_orca_interaction_velocities()
        for index, human in enumerate(self.dynamic_humans):
            if(human.goal_x==None or human.goal_y==None):
                raise AssertionError("Human goal not specified")
            if human.policy == "orca":
                velocity = self.compute_orca_velocity(human)
            elif human.policy == "sfm":
                velocity = self.compute_sfm_velocity(human)

            orientation = atan2(velocity[1], velocity[0])
            if human.set_new_orientation_with_limits(orientation, MAX_ORIENTATION_CHANGE, self.TIMESTEP):
                human.speed = np.linalg.norm(velocity)
                if human.speed < self.SPEED_THRESHOLD and not(self.crowd_forming and human.id in self.humans_forming_crowd.keys()): human.speed = 0
            else:
                human.speed = 0
            human.update(self.TIMESTEP)

        # updating moving humans in interactions
        for index, i in enumerate(self.moving_interactions):
            i.update(self.TIMESTEP, interaction_vels[index], self.MAX_ROTATION_HUMAN)
        
        # update the goals for humans if they have reached goal
        for human in self.dynamic_humans:
            if self.crowd_forming and human.id in self.humans_forming_crowd.keys(): continue  # handling the humans forming a crowd separately
            if self.h_l_forming and human.id == self.h_l_forming_human.id: continue  # handling the human forming the human-laptop-interaction separately
            HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
            HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
            if human.has_reached_goal():
                o = self.sample_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
                if o is not None:
                    human.set_goal(o.x, o.y)
                    self.goals[human.id] = o

        # update goals of interactions
        for i in self.moving_interactions:
            if i.has_reached_goal():
                HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
                HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
                o = self.sample_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
                if o is not None:
                    i.set_goal(o.x, o.y)
                    for human in i.humans:
                        self.goals[human.id] = o

        # complete the crowd formation if all the crowd-forming humans have reached their goals
        if self.crowd_forming:  # enter only when the environment is undergoing a crowd formation
            haveAllHumansReached = True
            for human in self.humans_forming_crowd.values():
                if human.has_reached_goal(offset=0):
                    # updating the orientation of humans so that the humans look towards each other
                    human.orientation = self.upcoming_interaction.humans[self.id_to_index[human.id]].orientation
                else:
                    haveAllHumansReached = False
            if haveAllHumansReached: self.finish_human_crowd_formation()
            else:
                if self.check_almost_crowd_formed():
                    self.almost_formed_crowd_count += 1
                else:
                    self.almost_formed_crowd_count = 0
                
                if self.almost_formed_crowd_count == 25:
                    self.finish_human_crowd_formation(make_approx_crowd=True)

        # complete human laptop interaction formation if the human has reached goal
        if self.h_l_forming:
            if self.h_l_forming_human.has_reached_goal(offset=0):
                self.finish_h_l_formation()

        # handling collisions
        self.handle_collision()

        # getting observations
        observation = self._get_obs()

        # computing rewards and done 
        reward, info = self.compute_reward_and_ticks(action)
        terminated = self._is_terminated
        truncated = self._is_truncated

        # updating the previous observations
        self.populate_prev_obs()

        self.cumulative_reward += reward

        # providing debugging information
        if DEBUG > 0 and self.ticks%50==0:
            self.render()
        elif DEBUG > 1:
            self.render()

        if DEBUG > 0 and (self._is_terminated or self._is_truncated):
            print(f'cumulative reward: {self.cumulative_reward}')

        # dispersing crowds
        if np.random.random() <= self.CROWD_DISPERSAL_PROBABILITY:
            t = np.random.randint(0, 2)
            self.dispersable_moving_crowd_indices = []
            self.dispersable_static_crowd_indices = []

            for ind, i in enumerate(self.moving_interactions):
                if i.can_disperse:
                    self.dispersable_moving_crowd_indices.append(ind)
            
            for ind, i in enumerate(self.static_interactions):
                if i.can_disperse:
                    self.dispersable_static_crowd_indices.append(ind)

            if t == 0 and len(self.dispersable_static_crowd_indices) > 0:
                index = random.choice(self.dispersable_static_crowd_indices)
                self.disperse_static_crowd(index)
            
            elif t == 1 and len(self.dispersable_moving_crowd_indices) > 0:
                index = random.choice(self.dispersable_moving_crowd_indices)
                self.disperse_moving_crowd(index)
        
        # disperse human-laptop
        self.dispersable_h_l_interaction_indices = []
        for ind, i in enumerate(self.h_l_interactions):
            if i.can_disperse:
                self.dispersable_h_l_interaction_indices.append(ind)
        
        if np.random.random() <= self.HUMAN_LAPTOP_DISPERSAL_PROBABILITY and len(self.dispersable_h_l_interaction_indices) > 0:
            index = random.choice(self.dispersable_h_l_interaction_indices)
            self.disperse_human_laptop(index)

        # forming interactions
        if np.random.random() <= self.CROWD_FORMATION_PROBABILITY and not self.crowd_forming and not self.h_l_forming:
            self.form_human_crowd()  # form a new human crowd
        
        if np.random.random() <= self.HUMAN_LAPTOP_FORMATION_PROBABILITY and not self.crowd_forming and not self.h_l_forming:
            self.form_human_laptop_interaction()  # form a new human-laptop interaction

        return observation, reward, terminated, truncated, info

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self)
        next_state, reward, terminated, truncated, info = env_copy.step(action_pre)
        del env_copy
        return next_state, reward, terminated, truncated, info

    def sample_goal(self, goal_radius, HALF_SIZE_X, HALF_SIZE_Y):
        start_time = time.time()
        while True:
            if self.check_timeout(start_time):
                break
            goal = Plant(
                id=None,
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                radius=goal_radius
            )
            
            collides = False
            all_objects = self.objects
            for obj in (all_objects + list(self.goals.values())): # check if spawned object collides with any of the exisiting objects. It will not be rendered as a plant.
                if obj is None: continue
                if(goal.collides(obj)):
                    collides = True
                    break

            if collides:
                del goal
            else:
                return goal
        return None

    def disperse_static_crowd(self, index):
        assert(len(self.static_interactions) > index)
        interaction = self.static_interactions[index]
        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
        static_human_count = 0
        new_dynamic_humans = []

        # set all the humans to dynamic first
        for human in interaction.humans: human.type = "dynamic"

        # add the humans from the interaction into the human list
        for human in interaction.humans:
            # randomly set the human to static or dynamic
            if (np.random.random() < 0.5): 
                human.type="static"  # default is dynamic
                static_human_count += 1

        if static_human_count <= 1:
            # remove the interaction from static interactions list
            self.static_interactions.pop(index)
            for human in interaction.humans:
                if human.type == "static": 
                    self.static_humans.append(human)
                    self.goals[human.id] = Plant(id=None, x=human.x, y=human.y, radius=self.HUMAN_GOAL_RADIUS)
                    human.set_goal(human.x, human.y)
                    human.speed = 0
                else : 
                    self.dynamic_humans.append(human)
                    new_dynamic_humans.append(human)          
        else:
            humans_to_remove = []
            for human in interaction.humans:
                if human.type == "dynamic":
                    humans_to_remove.append(human)
                    self.dynamic_humans.append(human)
                    new_dynamic_humans.append(human)

                else:
                    human.type = "dynamic"  # by default all the human types in any interaction is dynamic
            
            for human in humans_to_remove: interaction.humans.remove(human)

        # sample goals for individual humans
        success = 1
        for human in new_dynamic_humans:
            o = self.sample_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                success = 0
                break
            self.goals[human.id] = o
            human.set_goal(o.x, o.y)
            human.goal_radius = self.HUMAN_GOAL_RADIUS
            human.fov = self.HUMAN_FOV
            human.prob_to_avoid_robot = self.PROB_TO_AVOID_ROBOT

    def disperse_moving_crowd(self, index):
        assert(len(self.moving_interactions) > index)
        interaction = self.moving_interactions[index]
        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN

        # set all the humans to dynamic first
        for human in interaction.humans: human.type = "dynamic"

        static_humans = []
        dynamic_humans = []

        # add the humans from the interaction into the human list
        for human in interaction.humans:
            # randomly set the human to static or dynamic
            if (np.random.random() < 0.5): 
                human.type="static"  # default is dynamic
                static_humans.append(human)
            else:
                self.dynamic_humans.append(human)
                dynamic_humans.append(human)
        
        if(len(static_humans) <= 1): 
            for human in static_humans:
                self.goals[human.id] = Plant(id=None, x=human.x, y=human.y, radius=self.HUMAN_GOAL_RADIUS)
                human.set_goal(human.x, human.y)
                human.speed = 0
                self.static_humans.append(human)  # note this is the static humans list of the env, and not the local list created above

            # remove the interaction from moving interactions list
            self.moving_interactions.pop(index)
        else:
            interaction.humans.clear()
            for human in static_humans: 
                human.type = "dynamic"
                human.speed = 0
                interaction.humans.append(human)
            
            # randomly make this interaction static
            if np.random.random() <= 0.5:
                self.moving_interactions.pop(index)
                interaction.type = "stationary"
                for human in interaction.humans:
                    self.goals.pop(human.id)
                self.static_interactions.append(interaction)
            else:  # keeping the rest of the crowd moving to the same goal
                for human in interaction.humans: human.type = "dynamic"
                assert human.id in self.goals.keys()            
        
        # sample goals for individual humans
        success = 1
        for human in dynamic_humans:
            o = self.sample_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                success = 0
                break
            self.goals[human.id] = o
            human.set_goal(o.x, o.y)
            human.goal_radius = self.HUMAN_GOAL_RADIUS
            human.fov = self.HUMAN_FOV
            human.prob_to_avoid_robot = self.PROB_TO_AVOID_ROBOT

        return

    def disperse_human_laptop(self, index):
        assert(len(self.h_l_interactions) > index)
        interaction = self.h_l_interactions[index]

        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN

        self.dynamic_humans.append(interaction.human)  # does not make sense to keep the human static in front of the laptop, so pushing the human in dynamic humans list
        self.laptops.append(interaction.laptop)
        human = interaction.human
        self.h_l_interactions.pop(index)
        o = self.sample_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
        if o is not None:
            self.goals[human.id] = o
            human.set_goal(o.x, o.y)
            human.goal_radius = self.HUMAN_GOAL_RADIUS
            human.fov = self.HUMAN_FOV
            human.prob_to_avoid_robot = self.PROB_TO_AVOID_ROBOT

    def form_human_crowd(self):
        """Initiates the process of crowd formation
        """
        numHumans = random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS)
        if numHumans <= 1: # cannot form a crowd with one or less humans
            pass
        else:
            all_objects = self.static_humans + self.dynamic_humans + self.tables + self.laptops + self.plants + self.walls + self.static_interactions + self.moving_interactions + self.h_l_interactions + [self.robot]
            if(len(self.dynamic_humans) < numHumans):  # check if the number of moving humans are greater than the number of humans required to form a crowd
                pass
            else:
                start_time = time.time()  # for timeout purposes, recording the time
                indices = random.sample(range(len(self.dynamic_humans)), numHumans)  # randomly sample a few humans from the list of moving humans
                HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
                HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
                while True: # comes out of loop only when spawned object collides with none of current objects
                    if self.check_timeout(start_time, period=0.5):  # times out if a crowd location cannot be found in 0.5 seconds. Such a small period is set so that it does not interrupt the normal flow of the environment unnecessarily trying to form a crowd
                        break
                    x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                    y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                    i = Human_Human_Interaction(
                        x, y, "stationary", numHumans, self.INTERACTION_RADIUS, self.HUMAN_DIAMETER, self.MAX_ADVANCE_HUMAN, self.INTERACTION_GOAL_RADIUS, self.INTERACTION_NOISE_VARIANCE
                    )

                    collides = False
                    for obj in all_objects + list(self.goals.values()):
                        if(i.collides(obj)):
                            collides = True
                            break

                    if collides:
                        del i
                    else:
                        # setting goals of the selected humans to be the goals of the spawned interaction
                        # this will make the humans go near a particular point and form a crowd
                        for j in range(numHumans):
                            self.dynamic_humans[indices[j]].set_goal(i.humans[j].x, i.humans[j].y)  # setting goal
                            self.goals[self.dynamic_humans[indices[j]].id] = Plant(None, i.humans[j].x, i.humans[j].y, self.HUMAN_GOAL_RADIUS)  # recording the goals in the goal dict
                            self.dynamic_humans[indices[j]].policy = 'orca'  # this works better, crowds get formed with a higher probability when using orca policy
                        self.upcoming_interaction = i  # storing the new human-human interaction object
                        self.humans_forming_crowd:Dict[int, Human] = {}  # dictionary mapping human.id -> human
                        self.id_to_index = {}
                        for ind, j in enumerate(indices):
                            self.humans_forming_crowd[self.dynamic_humans[j].id] = self.dynamic_humans[j]
                            self.id_to_index[self.dynamic_humans[j].id] = ind

                        self.crowd_forming = True  # flag variable that indicates that the environment is undergoing a crowd formation
                        self.almost_formed_crowd_count = 0
                        break

    def check_almost_crowd_formed(self):
        """Checks if the humans forming the crowd are close by or not. If the humans haven't reached the goal, and are still nearby for 25 steps, then it is approximated as a crowd.

        Returns:
            bool: True when all the humans forming a crowd are very close, and False otherwise
        """ 
        i_x = 0  
        i_y = 0
        for human in self.humans_forming_crowd.values():
            i_x += human.x
            i_y += human.y
        
        i_x /= len(self.humans_forming_crowd)
        i_y /= len(self.humans_forming_crowd)

        can_form_crowd = True
        for human in self.humans_forming_crowd.values():
            if np.sqrt((i_x - human.x)**2 + (i_y - human.y)**2) > self.INTERACTION_RADIUS+self.HUMAN_DIAMETER/2:
                can_form_crowd = False
        return can_form_crowd
    
    def finish_human_crowd_formation(self, make_approx_crowd=False):
        """Finishes making the crowd. Removes the humans from the human_list and adds the new crowd in the interaction list.
        If make_approx_crowd is True, then the self.upcoming_interaction is ignored and a crowd is formed using the current location of the crowd forming humans.

        Args:
            make_approx_crowd (bool, optional): Parameter that decides whether self.upcoming_interaction has to be used or not. Defaults to False.
        """
        if not make_approx_crowd:
            l = len(self.dynamic_humans)
            count = 0  # counting for sanity check
            for human in self.humans_forming_crowd.values():
                self.dynamic_humans.remove(human)  # remove human from list
                count += 1
                self.upcoming_interaction.humans[self.id_to_index[human.id]] = human  # replace the human in the interaction with current human
                self.goals.pop(human.id)  # remove the human's id from the goal dictionary
            
            assert(len(self.dynamic_humans) + count == l)  # sanity check
            self.upcoming_interaction.arrange_humans()  # arrange the humans properly
            # randomly make the crowd static or moving
            if random.random() < 0.5:
                self.upcoming_interaction.type = "moving"

            if self.upcoming_interaction.type == "moving":
                # sampling goal for the moving interaction
                HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
                HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
                o = self.sample_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
                if o is not None:
                    # setting goal
                    self.upcoming_interaction.set_goal(o.x, o.y)
                    for human in self.upcoming_interaction.humans:
                        self.goals[human.id] = o
                
                self.moving_interactions.append(self.upcoming_interaction)
            
            else:
                self.static_interactions.append(self.upcoming_interaction)  # add the new interaction to the static interaction list

        else:
            i_x = 0  # x coordinate of center of geometry
            i_y = 0  # y coordinate of center of geometry

            # Calculate the value of center of geomety
            for human in self.humans_forming_crowd.values():
                i_x += human.x
                i_y += human.y
            
            i_x /= len(self.humans_forming_crowd)
            i_y /= len(self.humans_forming_crowd)
            
            del self.upcoming_interaction
            # making crowd of humans
            i = Human_Human_Interaction(
                i_x, i_y, "stationary", len(self.humans_forming_crowd), self.INTERACTION_RADIUS, self.HUMAN_DIAMETER, self.MAX_ADVANCE_HUMAN, self.INTERACTION_GOAL_RADIUS, self.INTERACTION_NOISE_VARIANCE
            )
            l = len(self.dynamic_humans)
            count = 0  # count for sanity check
            for ind, human in enumerate(self.humans_forming_crowd.values()):
                i.humans[ind] = human
                self.dynamic_humans.remove(human)  # remove human from list
                count += 1
                human.orientation = np.arctan2(i_y-human.y, i_x-human.x)  # setting orientation to face the center of geometry
            assert(len(self.dynamic_humans) + count == l)  # sanity check
            self.static_interactions.append(i)  # adding to interaction to list
        
        # setting the crowd formation related variables to default values
        self.crowd_forming = False
        self.almost_formed_crowd_count = 0

    def form_human_laptop_interaction(self):
        """Initiates the process of human-laptop-interaction formation
        """
        if len(self.laptops) == 0 or len(self.dynamic_humans) == 0:  # if no empty laptop, or no free moving human, do nothing
            pass
        else:
            index = random.sample(range(len(self.dynamic_humans)), 1)[0]  # sampling a random human 
            laptop_index = random.sample(range(len(self.laptops)), 1)[0]  # sampling a random laptop
            self.h_l_forming_human = self.dynamic_humans[index]
            self.h_l_forming_human.policy = 'orca'  # using orca policy seems to work better

            # list of all the objects for collision check
            all_objects = self.static_humans + self.dynamic_humans + self.tables + self.laptops + self.plants + self.walls + self.static_interactions + self.moving_interactions + self.h_l_interactions + [self.robot]

            # creating the human-laptop-interaction object
            self.upcoming_h_l_interaction = Human_Laptop_Interaction(
                self.laptops[laptop_index], self.LAPTOP_WIDTH+self.HUMAN_LAPTOP_DISTANCE, self.HUMAN_DIAMETER
            )

            # collision check
            collides = False

            for obj in all_objects + list(self.goals.values()):
                if self.upcoming_h_l_interaction.collides(obj, human_only=True):
                    collides = True
                    break
            
            if not collides:
                # setting the goal for the human
                self.h_l_forming_human.set_goal(self.upcoming_h_l_interaction.human.x, self.upcoming_h_l_interaction.human.y)
                self.goals[self.h_l_forming_human.id] = Plant(None, self.upcoming_h_l_interaction.human.x, self.upcoming_h_l_interaction.human.y, self.HUMAN_GOAL_RADIUS)
                # setting the flag variable for human-laptop-interaction formation to be true
                self.h_l_forming = True

    def finish_h_l_formation(self):
        """Completes the formation of human-laptop-interaction, and updates the lists accordingly
        """
        # remove the goal corresponding to the human
        self.goals.pop(self.h_l_forming_human.id)
        l = len(self.dynamic_humans)  # storing length for assertion
        self.dynamic_humans.remove(self.h_l_forming_human)  # removing human from list
        assert(len(self.dynamic_humans) == l-1)  # sanity check

        self.laptops.remove(self.upcoming_h_l_interaction.laptop)  # removing laptop from list
        self.upcoming_h_l_interaction.human = self.h_l_forming_human  
        
        self.upcoming_h_l_interaction.arrange_human()  # arranging the human to make it look aesthetically pleasing

        self.h_l_interactions.append(self.upcoming_h_l_interaction)  # adding the new human laptop interaction

        self.h_l_forming = False  # resetting the flag variable

    def populate_prev_obs(self):
        """
        Used to fill the dictionary storing the previous observations
        """
        
        # adding humans, tables, laptops, plants, walls
        for entity in self.static_humans + self.dynamic_humans + self.tables + self.laptops + self.plants + self.walls:
            coordinates = self.get_robot_frame_coordinates(np.array([[entity.x, entity.y]], dtype=np.float32)).flatten()
            sin_theta = np.sin(entity.orientation - self.robot.orientation)
            cos_theta = np.cos(entity.orientation - self.robot.orientation)
            theta = np.arctan2(sin_theta, cos_theta)
            self._prev_observations[entity.id] = EntityObs(
                entity.id,
                coordinates[0],
                coordinates[1],
                theta,
                sin_theta,
                cos_theta
            )

        # adding human-human interactions
        for i in self.moving_interactions + self.static_interactions:
            for entity in i.humans:
                coordinates = self.get_robot_frame_coordinates(np.array([[entity.x, entity.y]], dtype=np.float32)).flatten()
                sin_theta = np.sin(entity.orientation - self.robot.orientation)
                cos_theta = np.cos(entity.orientation - self.robot.orientation)
                theta = np.arctan2(sin_theta, cos_theta)
                self._prev_observations[entity.id] = EntityObs(
                    entity.id,
                    coordinates[0],
                    coordinates[1],
                    theta,
                    sin_theta,
                    cos_theta
                )
        
        # adding human-laptop interactions
        for i in self.h_l_interactions:
            entity = i.human
            coordinates = self.get_robot_frame_coordinates(np.array([[entity.x, entity.y]], dtype=np.float32)).flatten()
            sin_theta = np.sin(entity.orientation - self.robot.orientation)
            cos_theta = np.cos(entity.orientation - self.robot.orientation)
            theta = np.arctan2(sin_theta, cos_theta)
            self._prev_observations[entity.id] = EntityObs(
                entity.id,
                coordinates[0],
                coordinates[1],
                theta,
                sin_theta,
                cos_theta
            )

            entity = i.laptop
            coordinates = self.get_robot_frame_coordinates(np.array([[entity.x, entity.y]], dtype=np.float32)).flatten()
            sin_theta = np.sin(entity.orientation - self.robot.orientation)
            cos_theta = np.cos(entity.orientation - self.robot.orientation)
            theta = np.arctan2(sin_theta, cos_theta)
            self._prev_observations[entity.id] = EntityObs(
                entity.id,
                coordinates[0],
                coordinates[1],
                theta,
                sin_theta,
                cos_theta
            )
        
    def compute_reward_and_ticks(self, action):
        """
        Function to compute the reward and also calculate if the episode has finished
        """
        self.ticks += 1

        # calculate the distance to the goal
        distance_to_goal = np.sqrt((self.robot.goal_x - self.robot.x)**2 + (self.robot.goal_y - self.robot.y)**2)

        # calculate the distance to goal for the orca robot
        if (not self.has_orca_robot_collided) and (not self.has_orca_robot_reached_goal):
            distance_to_goal_orca_robot = np.sqrt((self.robot_orca.goal_x - self.robot_orca.x)**2 + (self.robot_orca.goal_y - self.robot_orca.y)**2)
            # check for object-robot collisions
            orca_robot_collision = False

            for object in self.static_humans + self.dynamic_humans + self.plants + self.walls + self.tables + self.laptops:
                if(self.robot_orca.collides(object)): 
                    orca_robot_collision = True
                    
            # interaction-robot collision
            for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
                if orca_robot_collision:
                    break
                if i.collides(self.robot):
                    orca_robot_collision = True
                    break

            if orca_robot_collision:
                self.has_orca_robot_collided = True

            if distance_to_goal_orca_robot < self.GOAL_THRESHOLD:
                self.has_orca_robot_reached_goal = True
                self.orca_robot_reach_time = self.ticks
            
            orca_robot_speed = np.linalg.norm(np.array([self.robot_orca.vel_x, self.robot_orca.vel_y], dtype=np.float32))
            self.orca_robot_path_length += orca_robot_speed * self.TIMESTEP


        collision_human, collision_object, collision_wall = self.check_robot_collision(self.robot)

        collision = collision_object or collision_human        
        
        dmin = float('inf')

        self.all_humans = []
        for human in self.static_humans + self.dynamic_humans : self.all_humans.append(human)

        for i in self.static_interactions + self.moving_interactions:
            for h in i.humans: self.all_humans.append(h)
        
        for i in self.h_l_interactions: self.all_humans.append(i.human)

        for human in self.all_humans:
            px = human.x - self.robot.x
            py = human.y - self.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.TIMESTEP + self.robot.orientation) - action[1] * np.cos(action[2]*self.TIMESTEP + self.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.TIMESTEP + self.robot.orientation) - action[1] * np.sin(action[2]*self.TIMESTEP + self.robot.orientation + np.pi/2)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER/2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for human in self.static_humans + self.dynamic_humans:
            px = human.x - self.robot.x
            py = human.y - self.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.TIMESTEP + self.robot.orientation) - action[1] * np.cos(action[2]*self.TIMESTEP + self.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.TIMESTEP + self.robot.orientation) - action[1] * np.sin(action[2]*self.TIMESTEP + self.robot.orientation + np.pi/2)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER/2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for interaction in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            px = interaction.x - self.robot.x
            py = interaction.y - self.robot.y

            speed = 0
            if interaction.name == "human-human-interaction":
                for h in interaction.humans:
                    speed += h.speed
                speed /= len(interaction.humans)


            vx = speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.TIMESTEP + self.robot.orientation) - action[1] * np.cos(action[2]*self.TIMESTEP + self.robot.orientation + np.pi/2)
            vy = speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.TIMESTEP + self.robot.orientation) - action[1] * np.sin(action[2]*self.TIMESTEP + self.robot.orientation + np.pi/2)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER/2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist       

        info = {
            "OUT_OF_MAP": False,
            "COLLISION_HUMAN": False,
            "COLLISION_OBJECT": False,
            "COLLISION": False,
            "DISCOMFORT_SNGNN": 0.0,
            "DISCOMFORT_DSRNN": 0.0,
            'sngnn_reward': 0.0,
            'distance_reward': 0.0,

            # metrics with different nomenclature
            "SUCCESS": False,
            "COLLISION_WALL": False,
            "TIMEOUT": False,
            "FAILURE_TO_PROGRESS": self.failure_to_progress,
            "STALLED_TIME": self.stalled_time,
            "TIME_TO_REACH_GOAL": None,  # will be None if the robot didn't reach the goal.
            "STL": None,
            "SPL": None
        }

        # calculate the reward and record necessary information
        if self.MAP_X/2 < self.robot.x or self.robot.x < -self.MAP_X/2 or self.MAP_Y/2 < self.robot.y or self.robot.y < -self.MAP_Y/2:
            self._is_terminated = self._is_terminated or self.END_WITH_COLLISION
            self._collision = True
            info["OUT_OF_MAP"] = True

        elif distance_to_goal < self.GOAL_THRESHOLD:
            self._is_terminated = True
            info["SUCCESS"] = True
            info["TIME_TO_REACH_GOAL"] = self.ticks * self.TIMESTEP

        elif collision is True:
            self._is_terminated = self._is_terminated or self.END_WITH_COLLISION
            self._collision = True
            info["COLLISION"] = True

            if collision_human:
                info["COLLISION_HUMAN"] = True
            
            if collision_object:
                info["COLLISION_OBJECT"] = True

        elif self.ticks > self.EPISODE_LENGTH:
            self._is_truncated = True
            info["TIMEOUT"] = True

        if collision_wall:
            info["COLLISION_WALL"] = True

        if distance_to_goal > self.prev_goal_distance:
            self.failure_to_progress += 1
            info["FAILURE_TO_PROGRESS"] = self.failure_to_progress
        self.prev_goal_distance = distance_to_goal

        robot_speed = np.linalg.norm(np.array([action[0], action[1]], dtype=np.float32))
        if robot_speed == 0:
            self.stalled_time += self.TIMESTEP
            info["STALLED_TIME"] = self.stalled_time
        
        self.robot_path_length += self.TIMESTEP * robot_speed
        info["PATH_LENGTH"] = self.robot_path_length

        self.reward_calculator.update_env(self)
        reward = self.reward_calculator.compute_reward(action, self._prev_observations, self._current_observations)
        for k, v in self.reward_calculator.info.items():
            info[k] = v
        
        # velocity based info
        self.v_min = min(self.v_min, robot_speed)
        self.v_max = max(self.v_max, robot_speed)
        self.v_avg *= (self.ticks-1)
        self.v_avg += robot_speed
        self.v_avg /= self.ticks

        info["V_MIN"] = self.v_min
        info["V_AVG"] = self.v_avg
        info["V_MAX"] = self.v_max
        
        # acceleration based info
        vel = np.array([action[0], action[1]], dtype=np.float32)
        a = (vel - self.prev_vel) / self.TIMESTEP
        self.prev_vel = vel
        a_magnitude = np.linalg.norm(a)
        
        self.a_min = min(self.a_min, a_magnitude)
        self.a_max = max(self.a_max, a_magnitude)
        self.a_avg *= (self.ticks - 1)
        self.a_avg += a_magnitude
        self.a_avg /= self.ticks

        info["A_MIN"] = self.a_min
        info["A_AVG"] = self.a_avg
        info["A_MAX"] = self.a_max

        jerk = (a - self.prev_a) / self.TIMESTEP
        self.prev_a = a
        jerk_magnitude = np.linalg.norm(jerk)
        self.jerk_min = min(self.jerk_min, jerk_magnitude)
        self.jerk_max = max(self.jerk_max, jerk_magnitude)
        self.jerk_avg *= (self.ticks - 1)
        self.jerk_avg += jerk_magnitude
        self.jerk_avg /= self.ticks

        info["JERK_MIN"] = self.jerk_min
        info["JERK_AVG"] = self.jerk_avg
        info["JERK_MAX"] = self.jerk_max

        # calculating the closest distance to humans and time to collision
        closest_human_dist = float('inf')
        time_to_collision = None
        robot_vx = action[0]*np.cos(self.robot.orientation) + action[1]*np.cos(self.robot.orientation + np.pi/2)
        robot_vy = action[0]*np.sin(self.robot.orientation) + action[1]*np.sin(self.robot.orientation + np.pi/2)

        for h in self.static_humans + self.dynamic_humans:
            closest_human_dist = min(closest_human_dist, np.sqrt((self.robot.x-h.x)**2 + (self.robot.y-h.y)**2))
            t = compute_time_to_collision(
                self.robot.x, 
                self.robot.y, 
                robot_vx,
                robot_vy,
                h.x,
                h.y,
                h.speed * np.cos(h.orientation),
                h.speed * np.sin(h.orientation),
                self.ROBOT_RADIUS,
                self.HUMAN_DIAMETER/2
            )

            if t != -1:
                if time_to_collision is None:
                    time_to_collision = ceil(t / self.TIMESTEP)
                else:
                    time_to_collision = min(time_to_collision, ceil(t / self.TIMESTEP))

        for i in self.moving_interactions + self.static_interactions:
            for h in i.humans:
                closest_human_dist = min(closest_human_dist, np.sqrt((self.robot.x-h.x)**2 + (self.robot.y-h.y)**2))
                t = compute_time_to_collision(
                    self.robot.x, 
                    self.robot.y, 
                    robot_vx,
                    robot_vy,
                    h.x,
                    h.y,
                    h.speed * np.cos(h.orientation),
                    h.speed * np.sin(h.orientation),
                    self.ROBOT_RADIUS,
                    self.HUMAN_DIAMETER/2
                )

                if t != -1:
                    if time_to_collision is None:
                        time_to_collision = ceil(t / self.TIMESTEP)
                    else:
                        time_to_collision = min(time_to_collision, ceil(t / self.TIMESTEP))
        
        for i in self.h_l_interactions:
            closest_human_dist = min(closest_human_dist, np.sqrt((self.robot.x-i.human.x)**2 + (self.robot.y-i.human.y)**2))
            h = i.human
            t = compute_time_to_collision(
                self.robot.x, 
                self.robot.y, 
                robot_vx,
                robot_vy,
                h.x,
                h.y,
                h.speed * np.cos(h.orientation),
                h.speed * np.sin(h.orientation),
                self.ROBOT_RADIUS,
                self.HUMAN_DIAMETER/2
            )

            if t != -1:
                if time_to_collision is None:
                    time_to_collision = ceil(t / self.TIMESTEP)
                else:
                    time_to_collision = min(time_to_collision, ceil(t / self.TIMESTEP))
        
        if time_to_collision is None:
            info["TIME_TO_COLLISION"] = -1
        else:
            info["TIME_TO_COLLISION"] = time_to_collision

        info["MINIMUM_DISTANCE_TO_HUMAN"] = closest_human_dist

        if (closest_human_dist - self.ROBOT_RADIUS) >= 0.45:  # same value used in SEAN 2.0
            self.compliant_count += 1
        
        info["PERSONAL_SPACE_COMPLIANCE"] = self.compliant_count / self.ticks

        # Success weighted by time length
        info["STL"] = 0
        info["SPL"] = 0
        if info["SUCCESS"]:
            if self.has_orca_robot_collided or not self.has_orca_robot_reached_goal:
                info["STL"] = 1
                info["SPL"] = 1
            else:
                stl_value = float(self.orca_robot_reach_time / self.ticks)
                if stl_value > 1: stl_value = 1
                info["STL"] = stl_value

                spl_value = float(self.orca_robot_path_length / info["PATH_LENGTH"])
                if spl_value > 1: spl_value = 1
                info["SPL"] = spl_value

        closest_obstacle_dist = float('inf')
        obstacle_dist_sum = 0
        obstacle_count = 0
        for p in self.plants:
            d = np.sqrt((self.robot.x-p.x)**2 + (self.robot.y-p.y)**2)-self.PLANT_RADIUS
            closest_obstacle_dist = min(closest_obstacle_dist, d)
            obstacle_dist_sum += d
            obstacle_count += 1


        for table in self.tables:
            p_x, p_y = get_nearest_point_from_rectangle(table.x, table.y, table.length, table.width, table.orientation, self.robot.x, self.robot.y)
            d = np.sqrt((self.robot.x - p_x)**2 + (self.robot.y - p_y)**2)
            closest_obstacle_dist = min(
                closest_obstacle_dist,
                d
            )
            obstacle_dist_sum += d
            obstacle_count += 1
        
        for wall in self.walls:
            p_x, p_y = get_nearest_point_from_rectangle(wall.x, wall.y, wall.length, wall.thickness, wall.orientation, self.robot.x, self.robot.y)
            d = np.sqrt((self.robot.x - p_x)**2 + (self.robot.y - p_y)**2)
            closest_obstacle_dist = min(
                closest_obstacle_dist,
                d
            )
            obstacle_dist_sum += d
            obstacle_count += 1

        info["MINIMUM_OBSTACLE_DISTANCE"] = closest_obstacle_dist
        info["AVERAGE_OBSTACLE_DISTANCE"] = None
        if obstacle_count > 0:
            info["AVERAGE_OBSTACLE_DISTANCE"] = obstacle_dist_sum / obstacle_count

        # information of the interacting entities within the environment
        info["interactions"] = {}
        info["interactions"]["human-human"] = []
        info["interactions"]["human-laptop"] = []

        curr_humans = len(self.static_humans + self.dynamic_humans)
        curr_laptops = len(self.laptops)

        for i, interaction in enumerate(self.moving_interactions + self.static_interactions):
            interaction_indices = []
            count_of_humans = 0
            
            for j, human in enumerate(interaction.humans):
                interaction_indices.append(curr_humans + j)
                count_of_humans += 1
            
            for p in range(len(interaction_indices)):
                for q in range(p+1, len(interaction_indices)):
                    info["interactions"]["human-human"].append((interaction_indices[p], interaction_indices[q]))
                    info["interactions"]["human-human"].append((interaction_indices[q], interaction_indices[p]))
            
            curr_humans += count_of_humans
        
        for i, interaction in enumerate(self.h_l_interactions):
            info["interactions"]["human-laptop"].append((curr_humans + i, curr_laptops + i))
            # assertion statement
            if(i == len(self.h_l_interactions)):
                assert(curr_humans + i == (self.total_humans - 1))
                assert(curr_laptops + i == len(self.laptops + self.h_l_interactions) - 1)

        return reward, info

    def check_robot_collision(self, robot):
        # check for object-robot and human-robot collisions
        collision_human = False
        for human in self.static_humans + self.dynamic_humans:
            if robot.collides(human):
                collision_human = True
                break
        
        # interaction-robot collision
        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            if collision_human:
                break
            if i.name == "human-human-interaction" and  i.collides(robot):
                collision_human = True
                break
            elif i.name == "human-laptop-interaction" and robot.collides(i.human):
                collision_human = True
                break

        collision_object = False
        collision_wall = False
        for object in self.plants + self.walls + self.tables + self.laptops:
            if(robot.collides(object)): 
                collision_object = True
                break
        for object in self.walls:
            if robot.collides(object):
                collision_wall = True
                break
        
        for i in self.h_l_interactions:
            if collision_object:
                break
            if robot.collides(i.laptop):
                collision_object = True
                break
        return collision_human, collision_object, collision_wall


    def check_timeout(self, start_time, period=30):
        if time.time()-start_time >= period:
            return True
        else:
            return False

    def _get_walls_for_L_shaped_room(self, location: int) -> List[Wall]:
        """Returns a list of walls according to the value of location

        Args:
            location (int): an integer between 0-3 (both inclusive)

        Returns:
            List[Wall]: A list of walls that together make the boundary of the L-shaped room
        """
        if location == 0:
            w_l8 = Wall(id=None, x=self.MAP_X/2 -self.L_X/2, y=self.MAP_Y/2 -self.L_Y, theta=np.pi, length=self.L_X, thickness=self.WALL_THICKNESS)
            w_l7 = Wall(id=None, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=-self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
            w_l6 = Wall(id=None, x=self.MAP_X/6, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l5 = Wall(id=None, x=-self.MAP_X/3, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l4 = Wall(id=None, x=-self.MAP_X/2 + (self.WALL_THICKNESS/2), y=-self.MAP_Y/6, theta=-np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l3 = Wall(id=None, x=-self.MAP_X/2 + (self.WALL_THICKNESS/2), y=self.MAP_Y/3, theta=-np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l2 = Wall(id=None, x=-self.L_X/2, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
            w_l1 = Wall(id=None, x=self.MAP_X/2 -self.L_X, y=self.MAP_Y/2 -self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=self.WALL_THICKNESS)
        elif location == 1:
            w_l8 = Wall(id=None, x=-self.MAP_X/2 + self.L_X, y=self.MAP_Y/2 -self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=self.WALL_THICKNESS)
            w_l7 = Wall(id=None, x=self.L_X/2, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
            w_l6 = Wall(id=None, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=self.MAP_Y/6, theta=np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l5 = Wall(id=None, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=-self.MAP_Y/3, theta=np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l4 = Wall(id=None, x=self.MAP_X/6, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l3 = Wall(id=None, x=-self.MAP_X/3, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l2 = Wall(id=None, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=-self.L_Y/2, theta=-np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
            w_l1 = Wall(id=None, x=-self.MAP_X/2 +self.L_X/2, y=self.MAP_Y/2 -self.L_Y, theta=np.pi, length=self.L_X, thickness=self.WALL_THICKNESS)
        elif location == 2:
            w_l8 = Wall(id=None, x=self.MAP_X/2 - self.L_X, y=-self.MAP_Y/2 + self.L_Y/2, theta=np.pi/2,length=self.L_Y, thickness=self.WALL_THICKNESS)
            w_l7 = Wall(id=None, x=-self.L_X/2, y=-self.MAP_Y/2+(self.WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
            w_l6 = Wall(id=None, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=-self.MAP_Y/6, theta=-np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l5 = Wall(id=None, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=self.MAP_Y/3, theta=-np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l4 = Wall(id=None, x=-self.MAP_X/6, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l3 = Wall(id=None, x=self.MAP_X/3, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l2 = Wall(id=None, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
            w_l1 = Wall(id=None, x=self.MAP_X/2 - self.L_X/2, y=-self.MAP_Y/2 +self.L_Y, theta=0, length=self.L_X, thickness=self.WALL_THICKNESS)
        elif location == 3:
            w_l8 = Wall(id=None, x=-self.MAP_X/2 + self.L_X/2, y=-self.MAP_Y/2 + self.L_Y, theta=0, length=self.L_X, thickness=self.WALL_THICKNESS)
            w_l7 = Wall(id=None, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=self.L_Y/2, theta=-np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
            w_l6 = Wall(id=None, x=-self.MAP_X/6, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l5 = Wall(id=None, x=self.MAP_X/3, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
            w_l4 = Wall(id=None, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=self.MAP_Y/6, theta=np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l3 = Wall(id=None, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=-self.MAP_Y/3, theta=np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
            w_l2 = Wall(id=None, x=self.L_X/2, y=-self.MAP_Y/2+(self.WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
            w_l1 = Wall(id=None, x= -self.MAP_X/2 +self.L_X, y= -self.MAP_Y/2 + self.L_Y/2, theta=-np.pi/2, length=self.L_Y, thickness=self.WALL_THICKNESS)
        else:
            raise AssertionError("location can take values between 0 and 3 only")

        return [w_l1, w_l2, w_l3, w_l4, w_l5, w_l6, w_l7, w_l8]
    
    def _add_corridors_to_L_shaped_room(self, location: int) -> None:
        """Adds corridors to an L-shaped room

        Args:
            location (int): an integer between 0-3 (both inclusive)
        """
        if location == 0:
            min_gap = max(self.ROBOT_RADIUS*2, self.HUMAN_DIAMETER) + 0.5
            gap1 = random.random() * min_gap + min_gap  # gap1 is sampled between min_gap and 2*min_gap
            gap2 = random.random() * min_gap + min_gap  # gap2 is sampled between min_gap and 2*min_gap
            gap1_center = random.random() * (self.MAP_X - gap1 - self.L_X) + (-self.MAP_X/2 + gap1/2)  # center of gap1 is sampled between (-X/2 + gap1/2, X/2 - LX - gap1/2)
            w1 = Wall(None, ((-self.MAP_X/2 + gap1_center-gap1/2)/2), self.MAP_Y/2 - self.L_Y, 0, (gap1_center-gap1/2 + self.MAP_X/2), self.WALL_THICKNESS)
            w2 = Wall(None, (gap1_center + gap1/2 + self.MAP_X/2 - self.L_X)/2, self.MAP_Y/2 - self.L_Y, 0, (self.MAP_X/2 - self.L_X - (gap1_center + gap1/2)), self.WALL_THICKNESS)
            self.walls.append(w1)
            self.objects.append(w1)
            self.walls.append(w2)
            self.objects.append(w2)
            self.objects.append(Wall(-1, gap1_center, -self.MAP_Y/2 - self.L_Y, 0, gap1, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
            gap2_center = random.random() * (self.MAP_Y - gap2 - self.L_Y) + (-self.MAP_Y/2 + gap2/2)  # center of gap2 is sampled between (-Y/2 + gap2/2, Y/2 - LY - gap2/2)
            w3 = Wall(None, self.MAP_X/2 - self.L_X, (-self.MAP_Y/2 + gap2_center-gap2/2)/2, np.pi/2, (gap2_center-gap2/2 + self.MAP_Y/2), self.WALL_THICKNESS)
            w4 = Wall(None, self.MAP_X/2 - self.L_X, (gap2_center + gap2/2 + self.MAP_Y/2 - self.L_Y)/2, np.pi/2, (self.MAP_Y/2 - self.L_Y - (gap2_center + gap2/2)), self.WALL_THICKNESS)
            self.walls.append(w3)
            self.objects.append(w3)
            self.walls.append(w4)
            self.objects.append(w4)
            self.objects.append(Wall(-1, self.MAP_X/2 - self.L_X, gap2_center, np.pi/2, gap2, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
        
        elif location == 1:
            min_gap = max(self.ROBOT_RADIUS*2, self.HUMAN_DIAMETER) + 0.5
            gap1 = random.random() * min_gap + min_gap  # gap1 is sampled between min_gap and 2*min_gap
            gap2 = random.random() * min_gap + min_gap  # gap2 is sampled between min_gap and 2*min_gap
            gap1_center = random.random() * (self.MAP_X - gap1 - self.L_X) + (-self.MAP_X/2 + self.L_X + gap1/2)  # center of gap1 is sampled between (-X/2 + LX + gap1/2, X/2 - gap1/2)
            w1 = Wall(None, ((-self.MAP_X/2 + self.L_X + gap1_center-gap1/2)/2), self.MAP_Y/2 - self.L_Y, 0, (gap1_center-gap1/2 + self.MAP_X/2 - self.L_X), self.WALL_THICKNESS)
            w2 = Wall(None, (gap1_center + gap1/2 + self.MAP_X/2)/2, self.MAP_Y/2 - self.L_Y, 0, (self.MAP_X/2 - (gap1_center + gap1/2)), self.WALL_THICKNESS)
            self.walls.append(w1)
            self.objects.append(w1)
            self.walls.append(w2)
            self.objects.append(w2)
            self.objects.append(Wall(-1, gap1_center, -self.MAP_Y/2 - self.L_Y, 0, gap1, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
            gap2_center = random.random() * (self.MAP_Y - gap2 - self.L_Y) + (-self.MAP_Y/2 + gap2/2)  # center of gap2 is sampled between (-Y/2 + gap2/2, Y/2 - LY - gap2/2)
            w3 = Wall(None, -self.MAP_X/2 + self.L_X, (-self.MAP_Y/2 + gap2_center-gap2/2)/2, np.pi/2, (gap2_center-gap2/2 + self.MAP_Y/2), self.WALL_THICKNESS)
            w4 = Wall(None, -self.MAP_X/2 + self.L_X, (gap2_center + gap2/2 + self.MAP_Y/2 - self.L_Y)/2, np.pi/2, (self.MAP_Y/2 - self.L_Y - (gap2_center + gap2/2)), self.WALL_THICKNESS)
            self.walls.append(w3)
            self.objects.append(w3)
            self.walls.append(w4)
            self.objects.append(w4)
            self.objects.append(Wall(-1, -self.MAP_X/2 + self.L_X, gap2_center, np.pi/2, gap2, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
        
        elif location == 2:
            min_gap = max(self.ROBOT_RADIUS*2, self.HUMAN_DIAMETER) + 0.5
            gap1 = random.random() * min_gap + min_gap  # gap1 is sampled between min_gap and 2*min_gap
            gap2 = random.random() * min_gap + min_gap  # gap2 is sampled between min_gap and 2*min_gap
            gap1_center = random.random() * (self.MAP_X - gap1 - self.L_X) + (-self.MAP_X/2 + gap1/2)  # center of gap1 is sampled between (-X/2 + gap1/2, X/2 - LX - gap1/2)
            w1 = Wall(None, ((-self.MAP_X/2 + gap1_center-gap1/2)/2), -self.MAP_Y/2 + self.L_Y, 0, (gap1_center-gap1/2 + self.MAP_X/2), self.WALL_THICKNESS)
            w2 = Wall(None, (gap1_center + gap1/2 + self.MAP_X/2 - self.L_X)/2, -self.MAP_Y/2 + self.L_Y, 0, (self.MAP_X/2 - self.L_X - (gap1_center + gap1/2)), self.WALL_THICKNESS)
            self.walls.append(w1)
            self.objects.append(w1)
            self.walls.append(w2)
            self.objects.append(w2)
            self.objects.append(Wall(-1, gap1_center, -self.MAP_Y/2 + self.L_Y, 0, gap1, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
            gap2_center = random.random() * (self.MAP_Y - gap2 - self.L_Y) + (-self.MAP_Y/2 + gap2/2 + self.L_Y)  # center of gap2 is sampled between (-Y/2 + gap2/2 + LY, Y/2 - gap2/2)
            w3 = Wall(None, self.MAP_X/2 - self.L_X, (-self.MAP_Y/2 + self.L_Y + gap2_center-gap2/2)/2, np.pi/2, (gap2_center-gap2/2 + self.MAP_Y/2 - self.L_Y), self.WALL_THICKNESS)
            w4 = Wall(None, self.MAP_X/2 - self.L_X, (gap2_center + gap2/2 + self.MAP_Y/2)/2, np.pi/2, (self.MAP_Y/2 - (gap2_center + gap2/2)), self.WALL_THICKNESS)
            self.walls.append(w3)
            self.objects.append(w3)
            self.walls.append(w4)
            self.objects.append(w4)
            self.objects.append(Wall(-1, self.MAP_X/2 - self.L_X, gap2_center, np.pi/2, gap2, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap

        elif location == 3:
            min_gap = max(self.ROBOT_RADIUS*2, self.HUMAN_DIAMETER) + 0.5
            gap1 = random.random() * min_gap + min_gap  # gap1 is sampled between min_gap and 2*min_gap
            gap2 = random.random() * min_gap + min_gap  # gap2 is sampled between min_gap and 2*min_gap
            gap1_center = random.random() * (self.MAP_X - gap1 - self.L_X) + (-self.MAP_X/2 + self.L_X + gap1/2)  # center of gap1 is sampled between (-X/2 + LX + gap1/2, X/2 - gap1/2)
            w1 = Wall(None, ((-self.MAP_X/2 + self.L_X + gap1_center-gap1/2)/2), -self.MAP_Y/2 + self.L_Y, 0, (gap1_center-gap1/2 + self.MAP_X/2 - self.L_X), self.WALL_THICKNESS)
            w2 = Wall(None, (gap1_center + gap1/2 + self.MAP_X/2)/2, -self.MAP_Y/2 + self.L_Y, 0, (self.MAP_X/2 - (gap1_center + gap1/2)), self.WALL_THICKNESS)
            self.walls.append(w1)
            self.objects.append(w1)
            self.walls.append(w2)
            self.objects.append(w2)
            self.objects.append(Wall(-1, gap1_center, -self.MAP_Y/2 + self.L_Y, 0, gap1, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
            gap2_center = random.random() * (self.MAP_Y - gap2 - self.L_Y) + (-self.MAP_Y/2 + gap2/2 + self.L_Y)  # center of gap2 is sampled between (-Y/2 + gap2/2 + LY, Y/2 - gap2/2)
            w3 = Wall(None, -self.MAP_X/2 + self.L_X, (-self.MAP_Y/2 + self.L_Y + gap2_center-gap2/2)/2, np.pi/2, (gap2_center-gap2/2 + self.MAP_Y/2 - self.L_Y), self.WALL_THICKNESS)
            w4 = Wall(None, -self.MAP_X/2 + self.L_X, (gap2_center + gap2/2 + self.MAP_Y/2)/2, np.pi/2, (self.MAP_Y/2 - (gap2_center + gap2/2)), self.WALL_THICKNESS)
            self.walls.append(w3)
            self.objects.append(w3)
            self.walls.append(w4)
            self.objects.append(w4)
            self.objects.append(Wall(-1, -self.MAP_X/2 + self.L_X, gap2_center, np.pi/2, gap2, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap

    def _add_corridors_to_rectangular_shaped_room(self):
        """Adds corridors to an rectangular/square shaped room
        """
        min_gap = max(self.ROBOT_RADIUS*2, self.HUMAN_DIAMETER) + 0.5
        gap1 = random.random() * min_gap + min_gap  # gap1 is sampled between min_gap and 2*min_gap
        gap2 = random.random() * min_gap + min_gap  # gap2 is sampled between min_gap and 2*min_gap
        gap1_center = random.random() * (self.MAP_X/2 - gap1/2)  # center of gap1 is sampled between (-X/2 + gap1/2, X/2 - gap1/2)
        w1 = Wall(None, ((-self.MAP_X/2 + gap1_center-gap1/2)/2), -self.MAP_Y/2 + self.MAP_Y/3, 0, (gap1_center-gap1/2 + self.MAP_X/2), self.WALL_THICKNESS)
        w2 = Wall(None, (gap1_center + gap1/2 + self.MAP_X/2)/2, -self.MAP_Y/2 + self.MAP_Y/3, 0, (self.MAP_X/2 - (gap1_center + gap1/2)), self.WALL_THICKNESS)
        self.walls.append(w1)
        self.objects.append(w1)
        self.walls.append(w2)
        self.objects.append(w2)
        self.objects.append(Wall(-1, gap1_center, -self.MAP_Y/2 + self.MAP_Y/3, 0, gap1, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap
        gap2_center = random.random() * (self.MAP_X/2 - gap2/2)  # center of gap2 is sampled between (-X/2 + gap2/2, X/2 - gap2/2)
        w3 = Wall(None, ((-self.MAP_X/2 + gap2_center-gap2/2)/2), -self.MAP_Y/2 + 2*self.MAP_Y/3, 0, (gap2_center-gap2/2 + self.MAP_X/2), self.WALL_THICKNESS)
        w4 = Wall(None, (gap2_center + gap2/2 + self.MAP_X/2)/2, -self.MAP_Y/2 + 2*self.MAP_Y/3, 0, (self.MAP_X/2 - (gap2_center + gap2/2)), self.WALL_THICKNESS)
        self.walls.append(w3)
        self.objects.append(w3)
        self.walls.append(w4)
        self.objects.append(w4)
        self.objects.append(Wall(-1, gap2_center, -self.MAP_Y/2 + 2*self.MAP_Y/3, 0, gap2, self.WALL_THICKNESS))  # adding this bit so that no obstacles are sampled in the gap


    def _add_walls(self):
        if self.shape == "L":
            # keep the direction of this as well
            self.location = np.random.randint(0,4)
            self.L_X = (2 if (self.location == 0 or self.location == 3) else 1) * self.MAP_X/3
            self.L_Y = (1 if (self.location == 0 or self.location == 3) else 2) * self.MAP_Y/3

            # placing a rectangular object (laptop) to avoid sampling new objects in the (rectangular) portion left outside the L-shaped room
            l = Laptop(
                id=None,
                x=(-1 if (self.location == 1 or self.location == 3) else 1)*self.MAP_X/2.0 + (-1 if (self.location == 0 or self.location == 2) else 1)*self.L_X/2.0,
                y=(-1 if (self.location == 2 or self.location == 3) else 1)*self.MAP_Y/2.0 + (-1 if (self.location == 0 or self.location == 1) else 1)*self.L_Y/2.0,
                width=self.L_Y,
                length=self.L_X,
                theta=0
            )

            walls = self._get_walls_for_L_shaped_room(self.location)
            if self.add_corridors:
                self._add_corridors_to_L_shaped_room(self.location)
            
            self.objects.append(l)
            for w in walls:
                self.walls.append(w)
                self.objects.append(w)

        # walls (hardcoded to be at the boundaries of the environment)
        elif self.shape != "no-walls":
            w1 = Wall(None, self.MAP_X/2-self.WALL_THICKNESS/2, 0, -np.pi/2, self.MAP_Y, self.WALL_THICKNESS)
            w2 = Wall(None, 0, -self.MAP_Y/2+self.WALL_THICKNESS/2, -np.pi, self.MAP_X, self.WALL_THICKNESS)
            w3 = Wall(None, -self.MAP_X/2+self.WALL_THICKNESS/2, 0, np.pi/2, self.MAP_Y, self.WALL_THICKNESS)
            w4 = Wall(None, 0, self.MAP_Y/2-self.WALL_THICKNESS/2, 0, self.MAP_X, self.WALL_THICKNESS)
            self.walls.append(w1)
            self.walls.append(w2)
            self.walls.append(w3)
            self.walls.append(w4)
            self.objects.append(w1)
            self.objects.append(w2)
            self.objects.append(w3)
            self.objects.append(w4)

        if self.add_corridors:  # corridors are hard coded to be at Y/3 and 2Y/3 where Y is the room's length along Y direction

            if self.shape == "L":
                pass
            
            else:
                self._add_corridors_to_rectangular_shaped_room()

    def _get_kwargs(self, object_type: SocNavGymObject, extra_info: dict = None):
        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
        arg_dict = {}
        if object_type == SocNavGymObject.ROBOT:
            arg_dict = {
                "id": 0,  # robot is assigned id 0
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "radius": self.ROBOT_RADIUS,
                "goal_x": None,
                "goal_y": None,
                "type": self.ROBOT_TYPE
            }
        elif object_type == SocNavGymObject.STATIC_HUMAN or object_type == SocNavGymObject.DYNAMIC_HUMAN:
            policy = self.HUMAN_POLICY
            if policy == "random": policy = random.choice(["sfm", "orca"])
            human_speed = 0
            human_type = "static"
            if object_type == SocNavGymObject.DYNAMIC_HUMAN:
                human_speed = random.uniform(0.0, self.MAX_ADVANCE_HUMAN)
                human_type = "dynamic"
            arg_dict = {
                "id": self.id,
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "width": self.HUMAN_DIAMETER,
                "speed": human_speed,
                "goal_radius": self.HUMAN_GOAL_RADIUS,
                "goal_x": None,
                "goal_y": None,
                "policy": policy,
                "fov": self.HUMAN_FOV,
                "prob_to_avoid_robot": self.PROB_TO_AVOID_ROBOT,
                "type": human_type
            }
        elif object_type == SocNavGymObject.PLANT:
            arg_dict = {
                "id": self.id,
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "radius": self.PLANT_RADIUS
            }
        elif object_type == SocNavGymObject.TABLE:
            arg_dict = {
                "id": self.id,
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "width": self.TABLE_WIDTH,
                "length": self.TABLE_LENGTH
            }
        elif object_type == SocNavGymObject.LAPTOP:
            # pick a random table
            i = random.randint(0, len(self.tables)-1)
            table = self.tables[i]
            
            # pick a random edge
            edge = np.random.randint(0, 4)
            if edge == 0:
                center = (
                    table.x + np.cos(table.orientation + np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2, 
                    table.y + np.sin(table.orientation + np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2
                )
                theta = table.orientation + np.pi
            
            elif edge == 1:
                center = (
                    table.x + np.cos(table.orientation + np.pi) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2, 
                    table.y + np.sin(table.orientation + np.pi) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2
                )
                theta = table.orientation - np.pi/2
            
            elif edge == 2:
                center = (
                    table.x + np.cos(table.orientation - np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2, 
                    table.y + np.sin(table.orientation - np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2
                )
                theta = table.orientation
            
            elif edge == 3:
                center = (
                    table.x + np.cos(table.orientation) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2, 
                    table.y + np.sin(table.orientation) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2
                )
                theta = table.orientation + np.pi/2
            
            arg_dict = {
                "id": self.id,
                "x": center[0],
                "y": center[1],
                "theta": theta,
                "width": self.LAPTOP_WIDTH,
                "length": self.LAPTOP_LENGTH
            }
        elif object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC \
            or object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING \
            or object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC \
            or object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING:
            assert extra_info != None
            if object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC:
                human_list = self.humans_in_h_h_dynamic_interactions
                interaction_type = "moving"
                can_disperse = True
            elif object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING:
                human_list = self.humans_in_h_h_dynamic_interactions_non_dispersing
                interaction_type = "moving"
                can_disperse = False
            elif object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC:
                human_list = self.humans_in_h_h_static_interactions
                interaction_type = "stationary"
                can_disperse = True
            else:
                human_list = self.humans_in_h_h_static_interactions_non_dispersing
                interaction_type = "stationary"
                can_disperse = False
            index = extra_info["index"]
            arg_dict = {
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X), 
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y), 
                "type": interaction_type, 
                "numOfHumans": human_list[index], 
                "radius": self.INTERACTION_RADIUS, 
                "human_width": self.HUMAN_DIAMETER, 
                "MAX_HUMAN_SPEED": self.MAX_ADVANCE_HUMAN, 
                "goal_radius": self.INTERACTION_GOAL_RADIUS, 
                "noise": self.INTERACTION_NOISE_VARIANCE, 
                "can_disperse": can_disperse
            }
        elif object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION \
            or object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION_NON_DISPERSING:
            assert extra_info != None
            arg_dict = {
                "laptop": extra_info["laptop"],
                "distance": self.LAPTOP_WIDTH+self.HUMAN_LAPTOP_DISTANCE,
                "width": self.HUMAN_DIAMETER,
                "can_disperse": object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION
            }

        return arg_dict

    def _sample_object(self, start_time: float, object_type: SocNavGymObject, extra_info=None):
        object_class = object_type.get_class()
        while True:
            if self.check_timeout(start_time):
                print(f"timed out spawning object of type {object_type.name}, starting again")
                return None
            
            kwargs = self._get_kwargs(object_type, extra_info)
            new_obj = object_class(**kwargs) 

            collides = False
            # check collision with other laptops in case of laptop. Otherwise, check collision with all other objects.
            objects_to_check_collision = None
            if object_type == SocNavGymObject.LAPTOP:
                objects_to_check_collision = copy.deepcopy(self.laptops)
                for i in self.h_l_interactions:
                    objects_to_check_collision.append(i.laptop)
            else: objects_to_check_collision = self.objects
            for obj in objects_to_check_collision: # check if spawned object collides with any of the exisiting objects
                if(new_obj.collides(obj)):
                    collides = True
                    break
            
            if collides:
                del new_obj
            else:
                break
        return new_obj
    
    def _sample_human_laptop_interaction(self, start_time: float, object_type: SocNavGymObject):
        while True:
            if self.check_timeout(start_time):
                print(f"timed out spawning object of type {object_type.name}, starting again")
                return None, None
            laptop = self._sample_object(start_time, SocNavGymObject.LAPTOP)
            if laptop == None:
                return None, None
            can_disperse = object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION
            i = Human_Laptop_Interaction(laptop, self.LAPTOP_WIDTH+self.HUMAN_LAPTOP_DISTANCE, self.HUMAN_DIAMETER, can_disperse=can_disperse)
            c = False
            for o in self.objects:
                if i.collides(o, human_only=True):
                    c = True
                    break
            if not c:
                return laptop, i
            else:
                del i
                del laptop

    def reset(self, seed=None, options=None) :
        success = False
        while not success:
            success, obs, info = self.try_reset(seed, options)
        return obs, info

    def try_reset(self, seed=None, options=None) :
        """
        Resets the environment
        """
        super().reset(seed=seed)
        start_time = time.time()
        if not self.has_configured:
            raise Exception("Please pass in the keyword argument config=\"path to config\" while calling gym.make")
        self.cumulative_reward = 0

        # setting seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # randomly initialize the parameters 
        self.randomize_params()
        self.id = 1

        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
        
        # keeping track of the scenarios for sngnn reward
        self.sn_sequence = []

        # to keep track of the current objects
        self.objects = []
        self.laptops = []
        self.walls = []
        self.static_humans = []
        self.dynamic_humans = []
        self.plants = []
        self.tables = []
        self.goals:Dict[int, Plant] = {}  # dictionary to store all the goals. The key would be the id of the entity. The goal would be a Plant object so that collision checks can be done.
        self.moving_interactions = []  # a list to keep track of moving interactions
        self.static_interactions = []
        self.h_l_interactions = []

        # clearing img_list
        if self.img_list is not None: 
            del self.img_list
            self.img_list = None

        # variable that shows whether a crowd is being formed currently or not
        self.crowd_forming = False

        # variable that shows whether a human-laptop-interaction is being formed or not
        self.h_l_forming = False

        # adding walls to the environment
        self._add_walls()

        # robot
        robot = self._sample_object(start_time, SocNavGymObject.ROBOT)
        if robot == None:
            return False, None, None
        self.robot = robot
        self.objects.append(self.robot)

        # making a copy of the robot for calculating time taken by a robot that has orca policy
        self.robot_orca = copy.deepcopy(self.robot)
        # defining a few parameters for the orca robot
        self.has_orca_robot_reached_goal = False
        self.has_orca_robot_collided = False
        self.orca_robot_reach_time = None
        self.orca_robot_path_length = 0

        # dynamic humans
        for _ in range(self.NUMBER_OF_DYNAMIC_HUMANS): # spawn specified number of humans
            human = self._sample_object(start_time, SocNavGymObject.DYNAMIC_HUMAN)
            if human == None:
                return False, None, None
            self.dynamic_humans.append(human)
            self.objects.append(human)
            self.id += 1

        # static humans
        for _ in range(self.NUMBER_OF_STATIC_HUMANS): # spawn specified number of humans
            human = self._sample_object(start_time, SocNavGymObject.STATIC_HUMAN)
            if human == None:
                return False, None, None
            self.static_humans.append(human)
            self.objects.append(human)
            self.id += 1
        
        # plants
        for _ in range(self.NUMBER_OF_PLANTS): # spawn specified number of plants
            plant = self._sample_object(start_time, SocNavGymObject.PLANT)
            if plant == None:
                return False, None, None
            self.plants.append(plant)
            self.objects.append(plant)
            self.id += 1

        # tables
        for _ in range(self.NUMBER_OF_TABLES): # spawn specified number of tables
            table = self._sample_object(start_time, SocNavGymObject.TABLE)  
            if table == None:
                return False, None, None
            self.tables.append(table)
            self.objects.append(table)
            self.id += 1

        # laptops
        if(len(self.tables) == 0):
            pass
        elif self.NUMBER_OF_LAPTOPS + self.NUMBER_OF_H_L_INTERACTIONS + self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING > 4 * self.NUMBER_OF_TABLES:
            raise AssertionError("Number of laptops exceeds the number of edges available on tables")
        else:
            for _ in range(self.NUMBER_OF_LAPTOPS): # placing laptops on tables
                laptop = self._sample_object(start_time, SocNavGymObject.LAPTOP)
                if laptop == None:
                    return False, None, None
                self.laptops.append(laptop)
                self.objects.append(laptop)
                self.id += 1

        # interactions        
        for ind in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC, extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.moving_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for ind in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING, extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.moving_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for ind in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC, extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.static_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for ind in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING, extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.static_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1
        
        for _ in range(self.NUMBER_OF_H_L_INTERACTIONS):
            # sampling a laptop
            laptop, interaction = self._sample_human_laptop_interaction(start_time, SocNavGymObject.HUMAN_LAPTOP_INTERACTION)
            if laptop == None or interaction == None:
                return False, None, None
            self.h_l_interactions.append(interaction)
            self.objects.append(interaction)
            self.id+=1
            interaction.human.id = self.id
            self.id += 1

        for _ in range(self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING):
            # sampling a laptop
            laptop, interaction = self._sample_human_laptop_interaction(start_time, SocNavGymObject.HUMAN_LAPTOP_INTERACTION_NON_DISPERSING)
            if laptop == None or interaction == None:
                return False, None, None
            self.h_l_interactions.append(interaction)
            self.objects.append(interaction)
            self.id+=1
            interaction.human.id = self.id
            self.id += 1

        # assigning ids to walls
        for wall in self.walls:
            wall.id = self.id
            self.id += 1

        # adding goals
        for human in self.dynamic_humans:   
            o = self.sample_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                return False, None, None
            self.goals[human.id] = o
            human.set_goal(o.x, o.y)

        for human in self.static_humans:   
            self.goals[human.id] = Plant(id=None, x=human.x, y=human.y, radius=self.HUMAN_GOAL_RADIUS)
            human.set_goal(human.x, human.y)  # setting goal of static humans to where they are spawned

        robot_goal = self.sample_goal(self.GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
        if robot_goal is None:
            return False, None, None
        self.goals[self.robot.id] = robot_goal
        self.robot.goal_x = robot_goal.x
        self.robot.goal_y = robot_goal.y
        self.robot_orca.goal_x = robot_goal.x
        self.robot_orca.goal_y = robot_goal.y

        for i in self.moving_interactions:
            o = self.sample_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                return False, None, None
            for human in i.humans:
                self.goals[human.id] = o
            i.set_goal(o.x, o.y)

        self._is_terminated = False
        self._is_truncated = False
        self._collision = False
        self.ticks = 0
        self.compliant_count = 0  # keeps track of how many times the agent is outside the personal space of humans
        self.prev_goal_distance = np.sqrt((self.robot.x - self.robot.goal_x)**2 + (self.robot.y - self.robot.goal_y)**2)
        self.robot_path_length = 0
        self.stalled_time = 0
        self.failure_to_progress = 0
        self.v_min = float("inf")
        self.v_max = 0.0
        self.v_avg = 0.0
        self.prev_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.a_min = float("inf")
        self.a_max = 0.0
        self.a_avg = 0.0
        self.prev_a = np.array([0.0, 0.0], dtype=np.float32)
        self.jerk_min = float("inf")
        self.jerk_max = 0.0
        self.jerk_avg = 0.0

        # all entities in the environment
        self.count = 0

        # a dictionary indexed by the id of the entity that stores the previous state observations for all the entities (except walls)
        self._prev_observations:Dict[int, EntityObs] = {}
        self._current_observations:Dict[int, EntityObs] = {}
        self.populate_prev_obs()

        obs = self._get_obs()

        self.reward_calculator.re_init(self)
        if self.reward_calculator.use_sngnn:
            self.reward_calculator.sngnn = SocNavAPI(device=('cuda'+str(self.cuda_device) if torch.cuda.is_available() else 'cpu'), params_dir=(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "sngnnv2", "example_model")))

        return True, obs, {}

    def render_without_showing(self, mode="human", draw_human_gaze=False, draw_human_goal=True):
        """
        Visualizing the environment
        """

        self.world_image = (np.ones((int(self.RESOLUTION_Y),int(self.RESOLUTION_X),3))*255).astype(np.uint8)

        # can be used for debugging. 
        if draw_human_gaze:
            for human in self.static_humans + self.dynamic_humans:
                human.draw_gaze_range(self.world_image, self.HUMAN_GAZE_ANGLE, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        for wall in self.walls:
            wall.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for table in self.tables:
            table.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for laptop in self.laptops:
            laptop.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        for plant in self.plants:
            plant.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        cv2.circle(self.world_image, (w2px(self.robot.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(self.robot.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(self.robot.x + self.GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(self.robot.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 255, 0), 2)
        
        if draw_human_goal:
            for human in self.dynamic_humans:  # only draw goals for the dynamic humans
                cv2.circle(self.world_image, (w2px(human.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(human.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(human.x + self.HUMAN_GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(human.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (120, 0, 0), 2)
            
            for i in self.moving_interactions:
                cv2.circle(self.world_image, (w2px(i.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(i.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(i.x + i.goal_radius, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(i.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 0, 255), 2)
        
        for human in self.static_humans + self.dynamic_humans:
            human.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        self.robot.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            i.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        ## uncomment to save the images 
        # cv2.imwrite("img"+str(self.count)+".jpg", self.world_image)
        # self.count+=1

        return self.world_image

    def render(self, mode="human", draw_human_gaze=False, draw_human_goal=True):
        if not self.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(self.RESOLUTION_VIEW), int(self.RESOLUTION_VIEW))
            self.window_initialised = True
        self.world_image = self.render_without_showing(mode, draw_human_gaze, draw_human_goal)
        cv2.imshow("world", self.world_image)
        k = cv2.waitKey(self.MILLISECONDS)
        if k%255 == 27:
            sys.exit(0)

    def record(self, path:str):
        """To record the episode 

        Args:
            path (str): Path to the video file (with .mp4 extension) 
        """
        if self.img_list is None:
            self.img_list = []
        
        img = (np.ones((int(self.RESOLUTION_Y),int(self.RESOLUTION_X),3))*255).astype(np.uint8)

        for wall in self.walls:
            wall.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for table in self.tables:
            table.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for laptop in self.laptops:
            laptop.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        for plant in self.plants:
            plant.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        cv2.circle(img, (w2px(self.robot.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(self.robot.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(self.robot.x + self.GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(self.robot.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 255, 0), 2)
        
        for human in self.dynamic_humans:  # only draw goals for the dynamic humans
            cv2.circle(img, (w2px(human.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(human.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(human.x + self.HUMAN_GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(human.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (120, 0, 0), 2)
        
        for i in self.moving_interactions:
            cv2.circle(img, (w2px(i.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(i.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(i.x + i.goal_radius, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(i.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 0, 255), 2)
        
        for human in self.static_humans + self.dynamic_humans:
            human.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        self.robot.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            i.draw(img, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        self.img_list.append(img)
        height, width, _ = img.shape

        if self._is_terminated or self._is_truncated:
            output = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (width, height))
            for i in range(len(self.img_list)):
                output.write(self.img_list[i])
            output.release()

    def close(self):
        pass
