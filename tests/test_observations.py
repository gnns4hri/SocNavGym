import pytest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gym
import socnavgym
import yaml
import numpy as np

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_observations():
    robot = [[5, 0, 0], [1, 1, 0], [0, 0, -np.pi/6], [-3, 4, np.pi/4]]
    laptop = [[8, 0, 0], [5, 5, 0], [-np.sqrt(3), 1, 0], [-7, 0, 0]]
    answer = [[3, 0, 0, 1], [4, 4, 0, 1], [-2, 0, 0.5, np.sqrt(3)/2], [-4*np.sqrt(2), 0, -1/np.sqrt(2), 1/np.sqrt(2)]]

    env = gym.make("SocNavGym-v1", config=os.path.dirname(os.path.abspath(__file__)) + "/../environment_configs/test2.yaml")
    env.set_padded_observations(True)
    obs, _ = env.reset()
    for i in range(len(robot)):
        env.robot.x = robot[i][0]
        env.robot.y = robot[i][1]
        env.robot.orientation = robot[i][2]

        env.laptops[0].x = laptop[i][0]
        env.laptops[0].y = laptop[i][1]
        env.laptops[0].orientation = laptop[i][2]
        
        obs, _, terminated, truncated, _ = env.step([0,0,0])
        obs = obs["laptops"]
        assert (np.abs(obs[6]-answer[i][0])<=1e-6) and (np.abs(obs[7]-answer[i][1])<=1e-6) and (np.abs(obs[8]-answer[i][2])<=1e-6) and (np.abs(obs[9]-answer[i][3])<=1e-6)

        if terminated or truncated:
            env.reset()
    
    env.set_padded_observations(False)
    for i in range(len(robot)):
        env.robot.x = robot[i][0]
        env.robot.y = robot[i][1]
        env.robot.orientation = robot[i][2]

        env.laptops[0].x = laptop[i][0]
        env.laptops[0].y = laptop[i][1]
        env.laptops[0].orientation = laptop[i][2]
        
        obs, _, terminated, truncated, _= env.step([0,0,0])
        obs = obs["laptops"]
        assert (np.abs(obs[6]-answer[i][0])<=1e-6) and (np.abs(obs[7]-answer[i][1])<=1e-6) and (np.abs(obs[8]-answer[i][2])<=1e-6) and (np.abs(obs[9]-answer[i][3])<=1e-6)

        if terminated or truncated:
            env.reset()