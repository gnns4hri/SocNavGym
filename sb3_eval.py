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
    discomfort_dsrnn = 0
    timeout = 0
    success_rate = 0
    time_taken = 0
    closest_human_dist = 0
    closest_obstacle_dist = 0
    collision_rate = 0
    collision_rate_human = 0
    collision_rate_object = 0
    collision_rate_wall = 0
    total_psc = 0
    total_stl = 0
    total_spl = 0
    total_failure_to_progress = 0
    total_stalled_time = 0
    total_path_length = 0
    total_vel_min = 0
    total_vel_max = 0
    total_vel_avg = 0
    total_a_min = 0
    total_a_max = 0
    total_a_avg = 0
    total_jerk_min = 0
    total_jerk_max = 0
    total_jerk_avg = 0
    total_avg_obstacle_distance = 0
    total_minimum_time_to_collision = 0
    total_time_to_reach_goal = 0
    
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
        has_collided_wall = 0
        has_timed_out = 0
        steps = 0
        count = 0
        episode_discomfort_sngnn = 0
        episode_discomfort_dsrnn = 0
        psc = 0
        stl = 0
        spl = 0
        failure_to_progress = 0
        stalled_time = 0
        time_to_reach_goal = env.EPISODE_LENGTH
        path_length = 0
        vel_min = 0
        vel_max = 0
        vel_avg = 0
        a_min = 0
        a_max = 0
        a_avg = 0
        jerk_min = 0
        jerk_max = 0
        jerk_avg = 0
        min_human_dist = float('inf')
        min_obstacle_dist = float('inf')
        avg_obstacle_dist = 0
        avg_minimum_time_to_collision = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # env.render()
            steps += 1
            count += 1

            # storing the rewards
            episode_reward += reward

            # storing discomforts
            episode_discomfort_sngnn += info["sngnn_reward"]
            episode_discomfort_dsrnn += info["DISCOMFORT_DSRNN"]

            # storing whether the agent reached the goal
            if info["SUCCESS"]:
                has_reached_goal = 1
                stl = info["STL"]
                spl = info["SPL"]
                time_to_reach_goal = info["TIME_TO_REACH_GOAL"]
            
            if info["COLLISION"]:
                has_collided = 1
                if info["COLLISION_HUMAN"]:
                    has_collided_human = 1
                if info["COLLISION_OBJECT"]:
                    has_collided_object = 1
                if info["COLLISION_WALL"]:
                    has_collided_wall = 1

                steps = env.EPISODE_LENGTH
            
            if info["TIMEOUT"]:
                has_timed_out = 1

            min_human_dist = min(min_human_dist, info["MINIMUM_DISTANCE_TO_HUMAN"])
            min_obstacle_dist = min(min_obstacle_dist, info["MINIMUM_OBSTACLE_DISTANCE"])
            avg_obstacle_dist += info["AVERAGE_OBSTACLE_DISTANCE"]
            if info["TIME_TO_COLLISION"] != -1: avg_minimum_time_to_collision += info["TIME_TO_COLLISION"]
            else: avg_minimum_time_to_collision += env.EPISODE_LENGTH
            episode_reward += reward
            
            obs = new_state
            
            if done:
                psc = info["PERSONAL_SPACE_COMPLIANCE"]
                failure_to_progress = info["FAILURE_TO_PROGRESS"]
                stalled_time = info["STALLED_TIME"]
                path_length = info["PATH_LENGTH"]
                vel_min = info["V_MIN"]
                vel_avg = info["V_AVG"]
                vel_max = info["V_MAX"]
                a_min = info["A_MIN"]
                a_avg = info["A_AVG"]
                a_max = info["A_MAX"]
                jerk_min = info["JERK_MIN"]
                jerk_avg = info["JERK_AVG"]
                jerk_max = info["JERK_MAX"]

        discomfort_sngnn += episode_discomfort_sngnn
        discomfort_dsrnn += episode_discomfort_dsrnn
        timeout += has_timed_out
        success_rate += has_reached_goal
        time_taken += steps
        closest_human_dist += min_human_dist
        closest_obstacle_dist += min_obstacle_dist
        collision_rate += has_collided
        collision_rate_human += has_collided_human
        collision_rate_object += has_collided_object
        collision_rate_wall += has_collided_wall
        total_psc += psc
        total_stl += stl
        total_spl += spl
        total_failure_to_progress += failure_to_progress
        total_stalled_time += stalled_time
        total_path_length += path_length
        total_vel_min += vel_min 
        total_vel_max += vel_max 
        total_vel_avg += vel_avg 
        total_a_min += a_min 
        total_a_max += a_max
        total_a_avg += a_avg 
        total_jerk_min += jerk_min 
        total_jerk_max += jerk_max 
        total_jerk_avg += jerk_avg
        total_avg_obstacle_distance += (avg_obstacle_dist / count)
        total_minimum_time_to_collision += (avg_minimum_time_to_collision / count)
        total_time_to_reach_goal += time_to_reach_goal

    print(f"Average discomfort_sngnn: {discomfort_sngnn/num_episodes}") 
    print(f"Average discomfort_dsrnn: {discomfort_dsrnn/num_episodes}") 
    
    print(f"Average success_rate: {success_rate/num_episodes}") 
    print(f"Average collision_rate: {collision_rate/num_episodes}")
    print(f"Average wall_collision_rate: {collision_rate_wall/num_episodes}")
    print(f"Average object_collision_rate: {collision_rate_object/num_episodes}")
    print(f"Average human_collision_rate: {collision_rate_human/num_episodes}")
    print(f"Average timeout: {timeout/num_episodes}") 
    print(f"Average time_taken: {time_taken/num_episodes}") 
    print(f"Average failure_to_progress: {total_failure_to_progress/num_episodes}")
    print(f"Average stalled_time: {total_stalled_time/num_episodes}")
    print(f"Average time_to_reach_goal: {total_time_to_reach_goal/num_episodes}")
    print(f"Average path_length: {total_path_length/num_episodes}")
    print(f"Average stl: {total_stl/num_episodes}")
    print(f"Average spl: {total_spl/num_episodes}")
    
    print(f"Average vel_min: {total_vel_min/num_episodes}")
    print(f"Average vel_avg: {total_vel_avg/num_episodes}")
    print(f"Average vel_max: {total_vel_max/num_episodes}")
    print(f"Average a_min: {total_a_min/num_episodes}")
    print(f"Average a_avg: {total_a_avg/num_episodes}")
    print(f"Average a_max: {total_a_max/num_episodes}")
    print(f"Average jerk_min: {total_jerk_min/num_episodes}")
    print(f"Average jerk_avg: {total_jerk_avg/num_episodes}")
    print(f"Average jerk_max: {total_jerk_max/num_episodes}")
    print(f"Average closest_obstacle_dist: {closest_obstacle_dist/num_episodes}") 
    print(f"Average average_obstacle distance: {total_avg_obstacle_distance/num_episodes}")
    print(f"Average psc: {total_psc/num_episodes}")
    print(f"Average closest_human_dist: {closest_human_dist/num_episodes}") 
    print(f"Average minimum_time_to_collision: {total_minimum_time_to_collision/num_episodes}")


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