import torch
import socnavgym
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.utils.sngnnv2.socnav import SocNavDataset
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import Human as otherHuman
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import Object as otherObject
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import SNScenario, SocNavAPI
from socnavgym.envs.utils.utils import point_to_segment_dist
from socnavgym.envs.utils import Human, Human_Human_Interaction, Human_Laptop_Interaction, Plant, Laptop, Wall
from socnavgym.envs.socnavenv_v1 import EntityObs 
from typing import Dict
import numpy as np
import os

class RewardAPI:
    def __init__(self, env:SocNavEnv_v1) -> None:
        self.env = env
        self.use_sngnn = False  # set this to True if SNGNN is being used in the reward function 
        self.sngnn = SocNavAPI(device= ('cuda' if torch.cuda.is_available() else 'cpu'), params_dir=(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "sngnnv2", "example_model")))
        self.sn_sequence = []
        self.info = {}  # record any information that should be returned in the info dict (in step function of the environment) here

        ## default values that should be calculated in your reward function to be returned
        self.info["DISCOMFORT_SNGNN"] = 0
        self.info["DISCOMFORT_DSRNN"] = 0
        self.info["distance_reward"] = 0
        self.info["alive_reward"] = 0
        self.info["sngnn_reward"] = 0

    def re_init(self, env:SocNavEnv_v1):
        self.__init__(env)

    def check_collision(self):
        # check for object-robot collisions
        collision = False

        for object in self.env.static_humans + self.env.dynamic_humans + self.env.plants + self.env.walls + self.env.tables + self.env.laptops:
            if(self.env.robot.collides(object)): 
                collision = True
       
        # interaction-robot collision
        for i in (self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions):
            if i.collides(self.env.robot):
                collision = True
                break
        
        return collision

    def check_out_of_map(self):
        return (self.env.MAP_X/2 < self.env.robot.x) or (self.env.robot.x < -self.env.MAP_X/2) or (self.env.MAP_Y/2 < self.env.robot.y) or (self.env.robot.y < -self.env.MAP_Y/2)

    def check_reached_goal(self):
        distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x)**2 + (self.env.robot.goal_y - self.env.robot.y)**2)
        return distance_to_goal < self.env.GOAL_THRESHOLD

    def check_timeout(self):
        return self.env.ticks > self.env.EPISODE_LENGTH
    
    def update_env(self, new_env:SocNavEnv_v1):
        self.env = new_env

    def compute_sngnn_reward(self, action, prev_obs:Dict[int, EntityObs], curr_obs:Dict[int, EntityObs]):
        assert(self.use_sngnn), "Please set the value of self.use_sngnn to True. This would help in choosing the GPU device according to the environment config"
        with torch.no_grad():
            sn = SNScenario((self.env.ticks * self.env.TIMESTEP))
            robot_goal = self.env.get_robot_frame_coordinates(np.array([[self.env.robot.goal_x, self.env.robot.goal_y]])).flatten()
            sn.add_goal(-robot_goal[1], robot_goal[0])
            if (10.32*float(action[2])) >= 4: rot = 4
            elif (10.32*float(action[2])) <= -4: rot = -4
            else: rot = (10.32*float(action[2]))
            if self.env.robot.type == "diff-drive":
                sn.add_command([min(9.4*float(action[0]), 3.5), 0.0, rot])
            else:
                # WARNING, SNGNN HAS NOT BEEN TRAINED ON HOLONOMIC ROBOTS
                sn.add_command([min(9.4*float(action[0]), 3.5), min(9.4*float(action[1]), 3.5), rot])  # these factors are multiplied to convert our units to Coppeliasim units
            # print(f"Action linear: {float(action[0])}  Action angular: {action[1]}")

            # Note that the following adjustments have been done so that the SNGNN network gets data from the distribution that it was trained on
            for laptop in self.env.laptops:
                obs = curr_obs[laptop.id]
                sn.add_object(
                    otherObject(
                        laptop.id, 
                        -obs.y,  # (-y, x) since the axes were inverted
                        obs.x, 
                        -(np.pi/2 + np.arctan2(obs.sin_theta, obs.cos_theta)),  # axis was opposite in SNGNN API
                        (prev_obs[laptop.id].x - obs.x) / (self.env.TIMESTEP/0.2),  # dx instead of vx 
                        (prev_obs[laptop.id].y - obs.y) / (self.env.TIMESTEP/0.2),  # dy instead of vy
                        (prev_obs[laptop.id].theta - np.arctan2(obs.sin_theta, obs.cos_theta))/(self.env.TIMESTEP/0.2),  # d_theta instead v_theta
                        laptop.length, 
                        laptop.width
                    )
                )

            for human in self.env.static_humans + self.env.dynamic_humans:
                human_obs = curr_obs[human.id]
                sn.add_human(
                    otherHuman(
                        human.id, 
                        -human_obs.y, 
                        human_obs.x, 
                        -(np.pi/2 + np.arctan2(human_obs.sin_theta, human_obs.cos_theta)), 
                        (prev_obs[human.id].x - human_obs.x)/(self.env.TIMESTEP/0.2),
                        (prev_obs[human.id].y - human_obs.y)/(self.env.TIMESTEP/0.2), 
                        (prev_obs[human.id].theta - np.arctan2(human_obs.sin_theta, human_obs.cos_theta))/(self.env.TIMESTEP/0.2)
                    )
                )
            
            for interaction in self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions:
                if interaction.name == "human-human-interaction":
                    for human in interaction.humans:
                        obs = curr_obs[human.id]
                        sn.add_human(
                            otherHuman(
                                human.id, 
                                -obs.y, 
                                obs.x, 
                                -(np.pi/2 + np.arctan2(obs.sin_theta, obs.cos_theta)), 
                                (prev_obs[human.id].x - obs.x)/(self.env.TIMESTEP/0.2),
                                (prev_obs[human.id].y - obs.y)/(self.env.TIMESTEP/0.2), 
                                (prev_obs[human.id].theta - np.arctan2(obs.sin_theta, obs.cos_theta))/(self.env.TIMESTEP/0.2)
                            )
                        )
                    for i in range(len(interaction.humans)):
                        for j in range(i+1, len(interaction.humans)):
                            sn.add_interaction([interaction.humans[i].id, interaction.humans[j].id])
                            sn.add_interaction([interaction.humans[j].id, interaction.humans[i].id])
                
                if interaction.name == "human-laptop-interaction":
                    human = interaction.human
                    laptop = interaction.laptop
                    obs = curr_obs[interaction.human.id]
                    sn.add_human(
                        otherHuman(
                            human.id, 
                            -obs.y, 
                            obs.x, 
                            -(np.pi/2 + np.arctan2(obs.sin_theta, obs.cos_theta)), 
                            (prev_obs[human.id].x - obs.x)/(self.env.TIMESTEP/0.2),
                            (prev_obs[human.id].y - obs.y)/(self.env.TIMESTEP/0.2), 
                            (prev_obs[human.id].theta - np.arctan2(obs.sin_theta, obs.cos_theta))/(self.env.TIMESTEP/0.2)
                        )
                    )
                    obs = curr_obs[interaction.laptop.id]
                    sn.add_object(
                        otherObject(
                            laptop.id, 
                            -obs.y, 
                            obs.x, 
                            -(np.pi/2 + np.arctan2(obs.sin_theta, obs.cos_theta)), 
                            (prev_obs[laptop.id].x - obs.x) / (self.env.TIMESTEP/0.2),
                            (prev_obs[laptop.id].y - obs.y) / (self.env.TIMESTEP/0.2),
                            (prev_obs[laptop.id].theta - np.arctan2(obs.sin_theta, obs.cos_theta))/(self.env.TIMESTEP/0.2),
                            interaction.laptop.length, 
                            interaction.laptop.width
                        )
                    )
                    sn.add_interaction([human.id, laptop.id])
            
            for plant in self.env.plants:
                obs = curr_obs[plant.id]
                sn.add_object(
                    otherObject(
                        plant.id, 
                        -obs.y, 
                        obs.x, 
                        -(np.pi/2 + np.arctan2(obs.sin_theta, obs.cos_theta)), 
                        (prev_obs[plant.id].x - obs.x) / (self.env.TIMESTEP/0.2),
                        (prev_obs[plant.id].y - obs.y) / (self.env.TIMESTEP/0.2),
                        (prev_obs[plant.id].theta - np.arctan2(obs.sin_theta, obs.cos_theta))/(self.env.TIMESTEP/0.2),
                        plant.radius*2, 
                        plant.radius*2
                    )
                )
            
            for table in self.env.tables:
                obs = curr_obs[table.id]
                sn.add_object(
                    otherObject(
                        table.id, 
                        -obs.y, 
                        obs.x, 
                        -(np.pi/2 + np.arctan2(obs.sin_theta, obs.cos_theta)), 
                        (prev_obs[table.id].x - obs.x) / (self.env.TIMESTEP/0.2),
                        (prev_obs[table.id].y - obs.y) / (self.env.TIMESTEP/0.2),
                        (prev_obs[table.id].theta - np.arctan2(obs.sin_theta, obs.cos_theta))/(self.env.TIMESTEP/0.2),
                        table.length, 
                        table.width
                    )
                )

            wall_list = []
            for wall in self.env.walls:
                x1 = wall.x - np.cos(wall.orientation)*wall.length/2
                x2 = wall.x + np.cos(wall.orientation)*wall.length/2
                y1 = wall.y - np.sin(wall.orientation)*wall.length/2
                y2 = wall.y + np.sin(wall.orientation)*wall.length/2
                a1 = self.env.get_robot_frame_coordinates(np.array([[x1, y1]])).flatten()
                a2 = self.env.get_robot_frame_coordinates(np.array([[x2, y2]])).flatten()
                wall_list.append({'x1': -a1[1], 'x2': -a2[1], 'y1': a1[0], 'y2': a2[0]})

            sn.add_room(wall_list)
            self.sn_sequence.insert(0, sn.to_json())
            ## Uncomment to write in json file
        
            # import json
            # with open("sample1.json", "w") as f:
            #     f.write("[")
            #     for i, d in enumerate(self.sn_sequence):
            #         json.dump(d, f, indent=4)
            #         if i != len(self.sn_sequence)-1:
            #             f.write(",\n")
            #     f.write("]")

            #     f.close()
            graph = SocNavDataset(self.sn_sequence, "1", "test", verbose=False)
            ret_gnn = self.sngnn.predictOneGraph(graph)[0]
            sngnn_value = float(ret_gnn[0].item())
            if sngnn_value < 0.:
                sngnn_value = 0.
            elif sngnn_value > 1.:
                sngnn_value = 1.
            
            return sngnn_value

    def compute_reward(self, action, prev_obs:EntityObs, curr_obs:EntityObs):
        raise NotImplementedError
