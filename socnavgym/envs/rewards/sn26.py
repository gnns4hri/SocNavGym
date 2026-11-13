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
import time
import numpy as np

import pickle

import socnavgym
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2, EntityObs
from socnavgym.envs.utils.utils import point_to_segment_dist
from socnavgym.envs.rewards.reward_api import RewardAPI

from socnavgym.envs.rewards.RNNBaseline.metrics import *
from socnavgym.envs.rewards.RNNBaseline.rnn  import RNNModel # collate_fn, FRAME_THRESHOLD
from socnavgym.envs.rewards.RNNBaseline.dataset_rnn import TrajectoryDataset, collate_fn
import socnavgym.envs.rewards.RNNBaseline.dataset_rnn as _ds

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

        # CPU on purpose: one tiny GRU pass per episode, and on GPUs cuDNN may use TF32, which shifts rewards by ~1e-4
        # and makes them depend on the machine (found 2026-11-08)
        self.device = torch.device('cpu')

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


    def create_JSON(self):
        try:
            datafile = open(self.new_filepath, "w")
            json.dump(self.data, datafile, indent=4)
            datafile.close()
        except TypeError as e:
            with open(f"serialized_datafile_{time.time()}.pkl", "wb") as file:
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
                "x": float(self.env.robot.goal_x),
                "y": float(self.env.robot.goal_y),
                "angle": float(self.env.robot.goal_a),
                "pos_threshold": float(self.env.GOAL_RADIUS),
                "angle_threshold": float(self.env.GOAL_ORIENTATION_THRESHOLD)
            }
        }
        for human in self.all_humans():
            humanInfo = {
                "id": int(human.id),
                "x": float(human.x),
                "y": float(human.y),
                "angle": float(human.orientation)
            }
            timestep["people"].append(humanInfo)

        for object in self.env.tables + self.env.plants + self.env.chairs + self.env.laptops:
            if object.name == "plant":   # metrics.get_dist_from_obj: circle 'length' is the diameter
                shape = {"type": "circle", "width": float(2 * object.radius), "length": float(2 * object.radius)}
            else:                        # rectangle: 'length' along 'angle', 'width' across (same as the env's collision shape)
                shape = {"type": "rectangle", "width": float(object.width), "length": float(object.length)}
            objectInfo = {
                "id": int(object.id),
                "type": object.name,
                "x": float(object.x),
                "y": float(object.y),
                "angle": float(object.orientation),
                "shape": shape
            }
            timestep["objects"].append(objectInfo)
        self.data["sequence"].append(timestep)

    def all_humans(self):
        """Every human in the scene: individuals, members of human-human interactions, and humans at laptops."""
        humans = list(self.env.static_humans) + list(self.env.dynamic_humans)
        for i in self.env.static_interactions + self.env.moving_interactions:
            humans += list(i.humans)
        for i in self.env.h_l_interactions:
            humans.append(i.human)
        return humans


    def compute_dmin(self, action):
        dmin = float('inf')
        for human in self.all_humans():
            px = human.x - self.env.robot.x
            py = human.y - self.env.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        return dmin


    CONTEXT = "A robot is performing routine tasks in a home."
    _tools = {}              # (context file, frame threshold) -> (context features, max metric values, feature names); read once

    def _scoring_tools(self):
        context_file = os.path.dirname(__file__)+"/RNNBaseline/anthropic_claude_context.csv"
        key = (context_file, self.FRAME_THRESHOLD)
        if key not in Reward._tools:
            if not os.path.exists(context_file):
                raise FileNotFoundError(f"sn26 context file missing: {context_file} (install the package with its data files)")
            ds = TrajectoryDataset([], context_file, label_exists=False, frame_threshold=self.FRAME_THRESHOLD, overwrite_context=self.CONTEXT)
            Reward._tools[key] = (ds.context_df.loc[self.CONTEXT.rstrip()].to_dict(), ds.max_metric_values, ds.all_features)
        return Reward._tools[key]

    def predict(self, model):
        """The network's score of the recorded trajectory. Same computation as before (TrajectoryDataset + DataLoader on a
        JSON file written for the purpose), without the file: the trajectory holds only Python floats, ints, strings and
        None, which a JSON round trip returns unchanged, so the features and the score are identical."""
        if len(self.data.get("sequence", [])) == 0:
            return 0.0
        context, max_values, features = self._scoring_tools()
        # sequence_to_tensor adds keys to every frame: work on copies of the frames so that self.data stays as recorded
        t_data = dict(self.data); t_data["sequence"] = [dict(f) for f in self.data["sequence"]]
        tensors = _ds.tensor_transform_to_goal_fr(_ds.sequence_to_tensor(t_data, self.FRAME_THRESHOLD, dict(context)))
        traj = _ds.metrics.normalize_and_cat_features(_ds.metrics.compute_metrics(tensors), max_values, features).to(torch.float32)
        with torch.no_grad():
            preds = model(traj[None].to(self.device), torch.tensor([traj.shape[0]-1], dtype=torch.long).to(self.device))
        prediction = preds.tolist()
        if len(prediction)==0:
            return 0.0
        return  float(prediction[0][0])


    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs, terminated, truncated):
        if terminated or truncated:
            self.add_to_trajectory()   # the terminal state (goal reached / collision) must be part of the scored trajectory
            try:
                predicted_reward = self.predict(self.model)
                ret = predicted_reward
            except IndexError as e:
                # An exception can be raised if the episode is exceedingly short, as some of
                # the metrics used as an input for the learned metric cannot be computed.
                if self.check_reached_goal():
                    ret = 0.9  # Close to 1, but the robot didn't do anything
                else:
                    ret = 0
            ret = (ret-0.5) * 2
        else:
            self.add_to_trajectory()
            ret = 0

        return ret
