import os
import sys
import time
import json
import random

import numpy as np

import cv2

import socnavgym
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2

import gymnasium as gym

from stable_baselines3 import PPO, SAC, TD3, DDPG, TRPO
from stable_baselines3.common.callbacks import CheckpointCallback

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList

CONFIG_FILE = "omni_config2.yaml"
GYM_NAME = "SocNavGym-v2"
SB3_POLICY = "MlpPolicy"
PROJECT_NAME = "RL"
DO_RENDERING = True
#
# These three constants here should probably be randomised.
#
GRID_WIDTH = 250 # size in cells
GRID_HEIGHT = 250 # size in cells
GRID_CELL_SIZE = 0.05 # size in meters. Square cells are assumed

NUMBER_OF_ENVS_IN_PARALLEL=12

OBJECTS_IN_GRID = False
UPDATE_PERIOD = 0.1
TOTAL_TIMESTEPS= 3_000_000

KEYS = sorted(["robot", "humans", "laptops", "plants", "tables", "walls"])
ALGOS = ["TD3", "PPO", "SAC", "DDPG", "TRPO"]
random.shuffle(ALGOS)


class TimeLimitCallback(BaseCallback):
    def __init__(self, time_limit_hours):
        super(TimeLimitCallback, self).__init__()
        self.time_limit_seconds = time_limit_hours * 3600
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit_seconds:
            print(f"Time limit of {self.time_limit_seconds/3600:.2f} hours reached. Stopping training.")
            return False
        return True




os.environ["SOCNAV_CONFIG_FILE"] = CONFIG_FILE

if __name__ == '__main__':


    class SocNavGymObservationWrapper(gym.ObservationWrapper):
        def __init__(self, render_mode=None):
            self.original_env = SocNavEnv_v2(render_mode=render_mode)
            super().__init__(self.original_env)
            self.keys = KEYS
            lows = []
            highs = []
            shape = 0
            for k in self.keys:
                lows.append(self.original_env.observation_space[k].low)
                highs.append(self.original_env.observation_space[k].high)
                shape += self.original_env.observation_space[k].shape[0]
                # print(f"KEY {k}: {self.original_env.observation_space[k].low.shape}, TOTAL {shape}")
            self.observation_space = gym.spaces.Box(low=np.concatenate(lows), high=np.concatenate(highs), shape=(shape,), dtype=np.float32)

        def observation(self, observation):
            obs = []
            for k in self.keys:
                obs.append(observation[k])
            return np.concatenate(obs)
        
    class ConcatObservationWrapper(gym.ObservationWrapper):
        def __init__(self, render_mode=None):
            self.original_env = SocNavEnv_v2(render_mode=render_mode)
            super().__init__(self.original_env)
            self.keys = KEYS
            lows = []
            highs = []
            shape = 0
            for k in self.keys:
                lows.append(self.original_env.observation_space[k].low)
                highs.append(self.original_env.observation_space[k].high)
                shape += self.original_env.observation_space[k].shape[0]
            self.observation_space = gym.spaces.Box(low=np.concatenate(lows), high=np.concatenate(highs), shape=(shape,), dtype=np.float32)

        def observation(self, observation):
            obs = []
            for k in self.keys:
                obs.append(observation[k])
            return np.concatenate(obs)
        


    config = {
            "policy_type": SB3_POLICY,
            "total_timesteps": TOTAL_TIMESTEPS,
            "env_name": GYM_NAME
    }

    def constructor():
            # return gym.make(GYM_NAME, config=CONFIG_FILE)
            # from socnavgym.wrappers.world_frame_observations import WorldFrameObservations
            return SocNavGymObservationWrapper(gym.make(GYM_NAME, config=CONFIG_FILE))


    class RenderEpisodeCallback(BaseCallback):
        def __init__(self, render_freq: int, identifier=None, verbose: int = 0):
            super().__init__(verbose)
            self.render_freq = render_freq
            self.episode_count = 0
            self.env = constructor()
            self.original_env = self.env.unwrapped
            self.save_dir = "SocNav3_RL_jsons/"
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)
            self.data_file_index = 0
            self.identifier = str(identifier)

        def _on_step(self) -> bool:
            finish = self.locals["dones"].nonzero()[0]
            if finish.size > 0:
                self.episode_count += finish.size
                if self.episode_count >= self.render_freq:
                    self.episode_count = 0
                    obs, _ = self.env.reset()
                    done = False
                    terminated = False
                    reward_acc = 0
                    n_steps = 0
                    last_save_simulation_time = -1
                    sequence = []
                    while not done and not terminated:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, terminated, _ = self.env.step(action)
                        reward_acc += reward
                        if DO_RENDERING is True:
                            self.env.render()

                        # SAVE THE DATA
                        people, objects, walls, robot, goal = self.get_data()

                        if n_steps==0:
                            grid = self.generate_grid(objects, walls)
                            walls = walls

                        n_steps += 1
                        simulation_time = n_steps*self.original_env.TIMESTEP

                        observation = {}
                        observation["timestamp"] = simulation_time
                        observation["robot"] = robot
                        observation["people"] = people
                        observation["objects"] = objects
                        observation["goal"] = goal

                        if simulation_time-last_save_simulation_time >= UPDATE_PERIOD:
                            sequence.append(observation)            
                            last_save_simulation_time = simulation_time
                    
                    self.save_data(grid, walls, sequence, reward_acc)

                    print(f"PLAY EPISODE {reward_acc=}")
            return True

        def get_data(self):
            people = []
            for human in self.original_env.static_humans + self.original_env.dynamic_humans:
                person = {}
                person["id"] = human.id
                person["x"] = human.x
                person["y"] = human.y
                person["angle"] = human.orientation
                people.append(person)

            objects = []
            for o in self.original_env.laptops + self.original_env.tables + self.original_env.chairs:
                obj = {}
                obj["id"] = o.id
                obj["x"] = o.x
                obj["y"] = o.y
                obj["angle"] = o.orientation
                obj["shape"] = {}
                obj["shape"]["type"] = "rectangle"
                obj["shape"]["width"] = o.width
                obj["shape"]["length"] = o.length
                if o in self.original_env.laptops:
                    obj["type"] = "laptop"
                elif o in self.original_env.tables:
                    obj["type"] = "table"
                else:
                    obj["type"] = "chair"
                objects.append(obj)
            for o in self.original_env.plants:
                obj = {}
                obj["id"] = o.id
                obj["x"] = o.x
                obj["y"] = o.y
                obj["angle"] = o.orientation
                obj["shape"] = {}
                obj["shape"]["type"] = "circle"
                obj["shape"]["width"] = o.radius*2
                obj["shape"]["length"] = o.radius*2
                obj["type"] = "plant"
                objects.append(obj)

            walls = []
            for wall in self.original_env.walls:
                x1 = wall.x - np.cos(wall.orientation)*wall.length/2
                x2 = wall.x + np.cos(wall.orientation)*wall.length/2
                y1 = wall.y - np.sin(wall.orientation)*wall.length/2
                y2 = wall.y + np.sin(wall.orientation)*wall.length/2
                walls.append([x1, y1, x2, y2])

            for interaction in self.original_env.moving_interactions + self.original_env.static_interactions + self.original_env.h_l_interactions:
                if interaction.name == "human-human-interaction":
                    for human in interaction.humans:
                        person = {}
                        person["id"] = human.id
                        person["x"] = human.x
                        person["y"] = human.y
                        person["angle"] = human.orientation
                        people.append(person)

                    for i in range(len(interaction.humans)):
                        for j in range(i+1, len(interaction.humans)):
                            inter = {}
                            inter["idSrc"] = interaction.humans[i].id
                            inter["idDst"] = interaction.humans[j].id
                            inter["type"] = "human-human-interaction"

                
                if interaction.name == "human-laptop-interaction":
                    human = interaction.human
                    laptop = interaction.laptop

                    person = {}
                    person["id"] = human.id
                    person["x"] = human.x
                    person["y"] = human.y
                    person["angle"] = human.orientation
                    people.append(person)

                    obj = {}
                    obj["id"] = laptop.id
                    obj["x"] = laptop.x
                    obj["y"] = laptop.y
                    obj["angle"] = laptop.orientation
                    obj["shape"]["type"] = "rectangle"
                    obj["shape"]["width"] = laptop.width
                    obj["shape"]["length"] = laptop.length
                    obj["type"] = "laptop"
                    objects.append(obj)

            
            robot = {}
            robot["x"] = self.original_env.robot.x
            robot["y"] = self.original_env.robot.y
            robot["angle"] = self.original_env.robot.orientation
            robot["speed_x"] = float(self.original_env.robot.vel_x)
            robot["speed_y"] = float(self.original_env.robot.vel_y)
            robot["speed_a"] = float(self.original_env.robot.vel_a)
            robot["shape"] = {}
            robot["shape"]["type"] = "circle"
            robot["shape"]["width"] = self.original_env.ROBOT_RADIUS*2
            robot["shape"]["length"] = self.original_env.ROBOT_RADIUS*2

            goal = {}
            goal["x"] = self.original_env.robot.goal_x
            goal["y"] = self.original_env.robot.goal_y
            goal["angle"] = self.original_env.robot.goal_a
            goal["pos_threshold"] = self.original_env.GOAL_RADIUS
            goal["angle_threshold"] = self.original_env.GOAL_ORIENTATION_THRESHOLD
            goal["type"] = "go-to"
            goal["human"] = None

            return people, objects, walls, robot, goal

        def world_to_grid(self, pW):
            pGx = pW[0]/GRID_CELL_SIZE + GRID_WIDTH/2
            pGy = pW[1]/GRID_CELL_SIZE + GRID_HEIGHT/2
            return (int(pGx), int(pGy))

        def rotate_points(self, points, center, angle):
            r_points = []
            for p in points:        
                p_x = center[0] - np.sin(angle) * (p[0] - center[0]) + np.cos(angle) * (p[1] - center[1])
                p_y = center[1] + np.cos(angle) * (p[0] - center[0]) + np.sin(angle) * (p[1] - center[1])

                r_points.append((p_x, p_y))
            return r_points

        def generate_grid(self, objects, walls):
            grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.int8)
            grid.fill(-1)
            room = []
            for w in walls:
                p1 = self.world_to_grid((w[0], w[1]))
                p2 = self.world_to_grid((w[2], w[3]))
                room.append(p1)
                room.append(p2)

            cv2.fillPoly(grid, [np.array(room, np.int32)], 0)
            cv2.polylines(grid, [np.array(room, np.int32)], True, 1)
            if OBJECTS_IN_GRID:
                for o in objects:
                    if o['type'] == "plant":
                        c = self.world_to_grid((o['x'], o['y']))
                        r = int(o['size'][0]/(2*GRID_CELL_SIZE))
                        cv2.circle(grid, c, r, 1, -1)
                    else:
                        points = []
                        points.append((o['x']-o['size'][0]/2, o['y']-o['size'][1]/2))
                        points.append((o['x']+o['size'][0]/2, o['y']-o['size'][1]/2))
                        points.append((o['x']+o['size'][0]/2, o['y']+o['size'][1]/2))
                        points.append((o['x']-o['size'][0]/2, o['y']+o['size'][1]/2))
                        r_points = self.rotate_points(points, (o['x'], o['y']), o['angle'])
                        g_points = []
                        for p in r_points:
                            w_p = self.world_to_grid(p)
                            g_points.append([int(w_p[0]), int(w_p[1])])
                        cv2.fillPoly(grid, [np.array(g_points, np.int32)], 1)

            return grid

        def save_data(self, grid, walls, sequence, reward):
            file_name = "SocNav3_RL_" + str(self.identifier) + '_{0:06d}'.format(self.data_file_index)
            print("saving", file_name)
            final_data = {}
            grid_data = {}
            grid_data["width"] = GRID_WIDTH
            grid_data["height"] = GRID_HEIGHT
            grid_data["cell_size"] = GRID_CELL_SIZE
            grid_data["x_orig"] = -GRID_CELL_SIZE*GRID_WIDTH/2
            grid_data["y_orig"] = -GRID_CELL_SIZE*GRID_HEIGHT/2
            grid_data["angle_orig"] = 0
            grid_data["data"] = grid.tolist()
            final_data["grid"] = grid_data
            final_data["walls"] = walls
            final_data["sequence"] = sequence
            final_data["reward"] = reward
            try:
                with open(self.save_dir+ file_name +'.json', 'w') as f:
                    f.write(json.dumps(final_data))
                    f.close()
                self.data_file_index += 1
            except Exception as e:
                print("Problem saving the json file")
                print(e)
                return
            



    for rl_model in ALGOS:
        match rl_model:
            case "SAC":
                env = make_vec_env(constructor, n_envs=NUMBER_OF_ENVS_IN_PARALLEL, vec_env_cls=SubprocVecEnv) # SubprocVecEnv  DummyVecEnv
                model = SAC(SB3_POLICY, env, verbose=1, tensorboard_log=f"./logs")
            case "PPO":
                env = make_vec_env(constructor, n_envs=NUMBER_OF_ENVS_IN_PARALLEL, vec_env_cls=SubprocVecEnv) # SubprocVecEnv  DummyVecEnv
                model = PPO(SB3_POLICY, env, verbose=1, tensorboard_log=f"./logs")
            case "TD3":
                env = constructor()
                model = TD3(SB3_POLICY, env, verbose=1, tensorboard_log=f"./logs")
            case "DDPG":
                env = make_vec_env(constructor, n_envs=NUMBER_OF_ENVS_IN_PARALLEL, vec_env_cls=SubprocVecEnv) # SubprocVecEnv  DummyVecEnv
                model = DDPG(SB3_POLICY, env, verbose=1, tensorboard_log=f"./logs")
            case "TRPO":
                env = make_vec_env(constructor, n_envs=NUMBER_OF_ENVS_IN_PARALLEL, vec_env_cls=SubprocVecEnv) # SubprocVecEnv  DummyVecEnv
                model = TRPO(SB3_POLICY, env, verbose=1, tensorboard_log=f"./logs")


        if "train" in sys.argv:
            run = wandb.init(
                project=PROJECT_NAME,
                id=rl_model+f"_{time.time()}",
                config=config,
                sync_tensorboard=True,
                monitor_gym=False,
                save_code=True,
            )
            checkpoint_cb = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix=f"{rl_model}_{time.time()}")
            # render_cb = RenderEpisodeCallback(render_freq=50, identifier=rl_model)
            wandb_cb = WandbCallback(gradient_save_freq=10, model_save_freq=100, model_save_path=f"models/{run.id}", verbose=2, log="all")
            tlimit_cb = TimeLimitCallback(time_limit_hours=12)  # Set your desired time limit

            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([checkpoint_cb, wandb_cb, tlimit_cb]))  #  render_cb,
            
            run.finish()

        # if "test" in sys.argv:
        #     env = constructor()
        #     # env.render_mode = "human"
        #     match rl_model:
        #         case "SAC":
        #             model = SAC(SB3_POLICY, env, verbose=1)
        #         case "PPO":
        #             model = PPO(SB3_POLICY, env, verbose=1)
        #         case "TD3":
        #             model = TD3(SB3_POLICY, env, verbose=1)
        #     model.load("this.zip", print_system_info=True)
        #     for _ in range(1000):
        #         obs, _ = env.reset()
        #         print("New episode")
        #         new_episode = False
        #         reward_acc = 0
        #         while new_episode is False:
        #             action, _states = model.predict(obs, deterministic=True)
        #             # action *= 2
        #             obs, reward, terminated, truncated, info = env.step(action)
        #             reward_acc += reward
        #             if DO_RENDERING is True:
        #                 env.render()
        #             if terminated or truncated:
        #                 new_episode = True
        #                 print(f"{reward_acc}")
        #     env.close()



