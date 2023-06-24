import gym
import socnavgym
import torch
from socnavgym.wrappers import DiscreteActions
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import argparse
from tqdm import tqdm
from stable_baselines3.common.monitor import Monitor
import sys


def eval(model, num_episodes, env):
    # intialising metrics
    discomfort_sngnn = 0
    discomfort_crowdnav = 0
    timeout = 0
    success_rate = 0
    time_taken = 0
    closest_human_dist = 0
    closest_obstacle_dist = 0
    collision_rate = 0
    collision_rate_human = 0
    collision_rate_object = 0
    total_psc = 0
    total_stl = 0

    
    total_reward = 0
    print(f"Evaluating model for {num_episodes} episodes")

    for i in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        has_reached_goal = 0
        has_collided = 0
        has_collided_human = 0
        has_collided_object = 0
        has_timed_out = 0
        steps = 0
        episode_discomfort_sngnn = 0
        episode_discomfort_crowdnav = 0
        psc = 0
        stl = 0
        min_human_dist = float('inf')
        min_obstacle_dist = float('inf')

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # env.render()
            steps += 1

            # storing the rewards
            episode_reward += reward

            # storing discomforts
            episode_discomfort_sngnn += info["sngnn_reward"]
            episode_discomfort_crowdnav += info["DISCOMFORT_CROWDNAV"]

            # storing whether the agent reached the goal
            if info["REACHED_GOAL"]:
                has_reached_goal = 1
                stl = info["success_weighted_by_time_length"]
            
            if info["COLLISION"]:
                has_collided = 1
                
                if info["COLLISION_HUMAN"]:
                    has_collided_human = 1
                if info["COLLISION_OBJECT"]:
                    has_collided_object = 1

                steps = env.EPISODE_LENGTH
            
            if info["MAX_STEPS"]:
                has_timed_out = 1

            min_human_dist = min(min_human_dist, info["closest_human_dist"])
            min_obstacle_dist = min(min_obstacle_dist, info["closest_obstacle_dist"])

            episode_reward += reward
            
            obs = new_state
            
            if done:
                psc = info["personal_space_compliance"]

        discomfort_sngnn += episode_discomfort_sngnn
        discomfort_crowdnav += episode_discomfort_crowdnav
        timeout += has_timed_out
        success_rate += has_reached_goal
        time_taken += steps
        closest_human_dist += min_human_dist
        closest_obstacle_dist += min_obstacle_dist
        collision_rate += has_collided
        collision_rate_human += has_collided_human
        collision_rate_object += has_collided_object
        total_psc += psc
        total_stl += stl

    print(f"Average discomfort_sngnn: {discomfort_sngnn/num_episodes}") 
    print(f"Average discomfort_crowdnav: {discomfort_crowdnav/num_episodes}") 
    print(f"Average timeout: {timeout/num_episodes}") 
    print(f"Average success_rate: {success_rate/num_episodes}") 
    print(f"Average time_taken: {time_taken/num_episodes}") 
    print(f"Average closest_human_dist: {closest_human_dist/num_episodes}") 
    print(f"Average closest_obstacle_dist: {closest_obstacle_dist/num_episodes}") 
    print(f"Average collision_rate: {collision_rate/num_episodes}")
    print(f"Average human_collision_rate: {collision_rate_human/num_episodes}")
    print(f"Average object_collision_rate: {collision_rate_object/num_episodes}")
    print(f"Average psc: {total_psc/num_episodes}")
    print(f"Average stl: {total_stl/num_episodes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_episodes", type=int, required=True, help="number of episodes")
    ap.add_argument("-w", "--weight_path", type=str, required=True, help="path to weight file")
    ap.add_argument("-c", "--config", type=str, required=True, help="path to config file")
    args = vars(ap.parse_args())
    env = gym.make("SocNavGym-v1", config=args["config"])
    env = DiscreteActions(env)

    try:
        model = DQN.load(args["weight_path"])
    except Exception as e:
        print(e)
        sys.exit(0)
        
    print("Successfully loaded")
    eval(model, args["num_episodes"], env)