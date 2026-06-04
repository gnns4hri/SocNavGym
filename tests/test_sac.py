import os
import time

import yaml

import gymnasium as gym
from stable_baselines3 import SAC

import socnavgym

from dict_subset_wrapper import DictToFlatWrapper



def load_train_config(config_path="train_config.yaml"):
    """Load configuration from YAML file."""
    # Look for config file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



# Load configuration
config = load_train_config("train_config.yaml")


# Create environment
base_env = gym.make("SocNavGym-v2", config=config["env_config"])
# Convert the environment to dictionary
env = DictToFlatWrapper(base_env, keys=config["keys"])

while not hasattr(base_env, "MILLISECONDS"):
    base_env = base_env.env


# Load the trained model
model = SAC.load("model.zip")

# Run the agent in the environment in test mode
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
    env.render()
    time.sleep(0.001*base_env.MILLISECONDS)
env.close()
