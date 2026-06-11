import sys

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate
from collections import namedtuple
import numpy as np
import math
import json
import os
import uuid
import numpy as np

import pickle

import socnavgym
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2, EntityObs
from socnavgym.envs.utils.utils import point_to_segment_dist
from socnavgym.envs.rewards.reward_api import RewardAPI

from socnavgym.envs.rewards.RNNBaseline.metrics import *
from socnavgym.envs.rewards.RNNBaseline.rnn  import RNNModel # collate_fn, FRAME_THRESHOLD
from socnavgym.envs.rewards.RNNBaseline.dataset_rnn import TrajectoryDataset, collate_fn

from datetime import datetime

class Reward(RewardAPI):
    def __init__(self, env: SocNavEnv_v2) -> None:
        super().__init__(env)
        self.checkpoint_directory = os.path.join(os.path.dirname(__file__), "RNNBaseline", "checkpoints") 
        checkpoint_path = os.path.join(self.checkpoint_directory, "baseline.pytorch")
        self.current_working_directory = os.getcwd()
        self.json_directory = os.path.join(self.current_working_directory, "experimental_JSONs")
        if not os.path.isdir(self.json_directory):
            os.mkdir(self.json_directory)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            os.makedirs(self.checkpoint_directory, exist_ok=True)
            import urllib.request
            URL = "https://www.dropbox.com/scl/fo/5mdx98kxux31tpz17t737/AAHIzVc82m32fPYvAjOyooU/models/baseline.pytorch?rlkey=70f89t67bg4zoa6g6lw5dcflg&st=sabxyxe3&dl=1"
            urllib.request.urlretrieve(URL, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.checkpoint = checkpoint


        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']

        if 'use_new_context' in checkpoint.keys():
            USE_NEW_CONTEXT = checkpoint['use_new_context']
        else:
            USE_NEW_CONTEXT = False
        if 'rnn_type' in checkpoint.keys():
            RNN_TYPE = checkpoint['rnn_type']
        else:
            RNN_TYPE = 'GRU'
        if 'frame_threshold' in checkpoint.keys():
            self.FRAME_THRESHOLD = checkpoint['frame_threshold']
        else:
            self.FRAME_THRESHOLD = 0.1

        if 'linear_layers' in checkpoint.keys():
            LINEAR_LAYERS = checkpoint['linear_layers']
        else:
            LINEAR_LAYERS = []

        if 'activation' in checkpoint.keys():
            ACTIVATION = checkpoint['activation']
        else:
            ACTIVATION = 'linear'

        if 'context_features' in checkpoint.keys():
            CONTEXT_FEATURES = checkpoint['context_features']
        else:
            CONTEXT_FEATURES = 0

        model = RNNModel(input_size, hidden_size, num_layers, rnn_type=RNN_TYPE, linear_layers=LINEAR_LAYERS, activation=ACTIVATION, context_vars=CONTEXT_FEATURES).to("cpu")
        model.to(self.device)
        self.model = model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.reset()
        self.new_trajectory()

    def reset(self):
        self.reach_reward = 1.0
        self.out_of_map_reward = -1.0
        self.max_steps_reward = -1.0
        self.alive_reward = -1e-6
        self.collision_reward = -1.0
        self.distance_reward_scaler = 0.01
        self.discomfort_distance = 0.6
        self.discomfort_penalty_factor = 0.5
        self.prev_distance = None
        self.prev_angular_distance = None
        self.total_distance_reward = 0.0
        self.min_dist = float('inf')

    def remove_JSON(self):
        os.remove(self.new_filepath)

    def create_JSON(self):
        try:
            datafile = open(self.new_filepath, "w")
            json.dump(self.data, datafile, indent=4)
            datafile.close()
        except TypeError as e:
            with open('serialized_datafile_{time.time()}.pkl', 'wb') as file:
                pickle.dump(self.data, file)
                raise e

    def new_trajectory(self):
        filestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_filename = f"Trajectory-{filestamp}-{uuid.uuid4().hex}.json"
        self.new_filepath = os.path.join(self.json_directory, new_filename)
        self.data = {

        }
        self.data["grid"] = {}
        self.data["sequence"] = []
        walls = []
        for wall in self.env.walls:
            x1 = wall.x - np.cos(wall.orientation)*wall.length/2
            x2 = wall.x + np.cos(wall.orientation)*wall.length/2
            y1 = wall.y - np.sin(wall.orientation)*wall.length/2
            y2 = wall.y + np.sin(wall.orientation)*wall.length/2
            walls.append([x1, y1, x2, y2])
        self.data["walls"] = walls

    def add_to_trajectory(self):
        timestep = {
            "timestamp": float(self.env.ticks * self.env.TIMESTEP),
            "robot": {
                "x": float(self.env.robot.x),
                "y": float(self.env.robot.y),
                "angle": float(self.env.robot.orientation),
                "speed_x": float(self.env.robot.vel_x),
                "speed_y": float(self.env.robot.vel_y),
                "speed_a": float(self.env.robot.vel_a),
                "shape": {
                    "type": "Circle",
                    "width": float(self.env.robot.radius*2),
                    "length": float(self.env.robot.radius*2)
                }
            },
            "people": [],
            "objects": [],
            "goal": {
                "type": "go-to",
                "human": None,
                "x": self.env.robot.goal_x,
                "y": self.env.robot.goal_y,
                "angle": self.env.robot.goal_a,
                "pos_threshold": self.env.GOAL_RADIUS,
                "angle_threshold": self.env.GOAL_ORIENTATION_THRESHOLD
            }
        }
        for human in self.env.dynamic_humans + self.env.static_humans:
            humanInfo = {
                "id": human.id,
                "x": human.x,
                "y": human.y,
                "angle": human.orientation
            }
            timestep["people"].append(humanInfo)
        for object in self.env.tables + self.env.plants + self.env.chairs + self.env.laptops:
            objectInfo = {
                "id": object.id,
                "x": object.x,
                "y": object.y,
                "angle": object.orientation
            }
            timestep["people"].append(objectInfo)
        self.data["sequence"].append(timestep)


    def compute_dmin(self, action):
        dmin = float('inf')

        all_humans = []
        for human in self.env.static_humans + self.env.dynamic_humans: all_humans.append(human)

        for i in self.env.static_interactions + self.env.moving_interactions:
            for h in i.humans: all_humans.append(h)

        for i in self.env.h_l_interactions:
            all_humans.append(i.human)

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


    def compute_dgoal(self):
        distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x)**2 + (self.env.robot.goal_y - self.env.robot.y)**2) #dist to goal
        if self.prev_distance is None: # in the case there is no prev distance set yet
            self.prev_distance = distance_to_goal
            distance_change = 0 # on the first step distance change would be zero

        else: # in the case there is a prev distance
            distance_change = self.prev_distance - distance_to_goal # we calculate the distance changes
            self.prev_distance = distance_to_goal # then update the previous (so the current distance is the prev distance for the next timestep)

        dis_reward = distance_change / MAX_ENV_PATH / MAX_EP_LENGTH # then we get the 'reward' for this distance change and return
        return dis_reward

    def predict(self, model):
        self.create_JSON()
        if len(self.data.get("sequence", [])) == 0:
            return self.collision_reward
        try:
            dataset = TrajectoryDataset(self.new_filepath, os.path.dirname(__file__)+"/RNNBaseline/anthropic_claude_context.csv", label_exists=False, frame_threshold=self.FRAME_THRESHOLD, overwrite_context="A robot is performing routine tasks in a home.")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error occured in creating dataset for trajectory: {self.new_filepath}, returing collision reward. Error: {e}")
            return self.collision_reward
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        prediction = []
        with torch.no_grad():
            for trajectories, _, slengths in data_loader:
                # Pass the whole batched graph sequence to the model at once
                preds = model(trajectories.to(self.device), slengths.to(self.device))
                prediction += preds.tolist() 
        if len(prediction)==0:
            print(f"Prediction error for dataset formed by trajectory: {self.new_filepath}, returning collision reward")
        key = prediction[0][0]
        return key

    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs, terminated, truncated):
        if terminated or truncated:
            try:
                predicted_reward = self.predict(self.model)
                self.remove_JSON()
                ret = predicted_reward
            except IndexError as e:
                # An exception can be raised if the episode is exceedingly short, as some of
                # the metrics used as an input for the learned metric cannot be computed.
                if self.check_reached_goal():
                    ret = 0.9  # Close to 1, but the robot didn't do anything
                else:
                    ret = 0
        else:
            self.add_to_trajectory()
            ret = 0

        ret = (ret-0.5) * 2

