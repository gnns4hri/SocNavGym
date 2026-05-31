import os
import sys
import time
from datetime import datetime
from collections import namedtuple

import yaml
import numpy as np

import socnavgym
import gymnasium as gym


from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


try:
    WANDB_MODE = os.environ["WANDB_MODE"]
except:
    WANDB_MODE = "online"

if WANDB_MODE != "offline":
    from wandb.integration.sb3 import WandbCallback
    import wandb

from dict_subset_wrapper import DictToFlatWrapper


def load_config(config_path="train_config.yaml"):
    """Load configuration from YAML file."""
    # Look for config file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories."""
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["best_model_dir"], exist_ok=True)
    os.makedirs("models", exist_ok=True)


# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

config = load_config()

# Generate a run name if not provided
if config["wandb"]["name"] is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["wandb"]["name"] = f"sac_{timestamp}"

# ---------------------------------------------------------------------------
# Initialise wandb run
# ---------------------------------------------------------------------------
if WANDB_MODE != "offline":
    run = wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
else:
    run = namedtuple('rubbish', ["id", "run", "name"])("offline", "offline", config["wandb"]["name"])

# Update config with wandb run ID for model saving
config["wandb"]["run_id"] = run.id

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env():
    """Instantiate, wrap, and monitor a single environment."""
    config_path = os.path.join(os.path.dirname(__file__), config["env_config"])
    env = gym.make("SocNavGym-v2", config=config_path)
    
    # Apply HER wrapper if enabled
    if config.get("her", {}).get("enabled", False):
        from her_wrapper import HERGoalEnvWrapper
        env = HERGoalEnvWrapper(env, config["her"])
        
    env = DictToFlatWrapper(env, keys=config["keys"])
    env = Monitor(env, filename=None)
    return env


# ---------------------------------------------------------------------------
# Build vectorised envs
# ---------------------------------------------------------------------------

train_env = make_vec_env(make_env, n_envs=config["n_envs"])
eval_env  = make_vec_env(make_env, n_envs=1)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

setup_directories(config)

checkpoint_callback = CheckpointCallback(
    save_freq=config["eval_freq"],
    save_path=config["checkpoint_dir"],
    name_prefix=f"sac_model_{run.name}",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=config["best_model_dir"],
    log_path=config["best_model_dir"],
    eval_freq=config["eval_freq"],
    n_eval_episodes=config["eval_episodes"],
    deterministic=True,
    render=False,
)

if WANDB_MODE != "offline":
    wandb_callback = WandbCallback(gradient_save_freq=1000, model_save_path=f"models/{run.id}", model_save_freq=config["eval_freq"], verbose=2)
    callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
else:
    callbacks = None

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Extract SAC-specific hyperparameters from config
sac_params = {k: v for k, v in config["sac_hyperparams"].items() if v is not None}

model = SAC(
    policy="MlpPolicy",
    env=train_env,
    verbose=config["verbose"],
    tensorboard_log=f"runs/{run.id}",
    **sac_params
)

# Apply HER replay buffer wrapper if enabled
if config.get("her", {}).get("enabled", False):
    from her_wrapper import HERReplayBufferWrapper
    
    # Get the replay buffer from the model
    # Unwrap the environment to get the HER wrapper
    base_env = model.env.envs[0]
    while hasattr(base_env, 'env') and not hasattr(base_env, 'sample_her_transitions'):
        base_env = base_env.env
    
    her_buffer = HERReplayBufferWrapper(
        model.replay_buffer,
        base_env,  # Get the HER wrapper
        config["her"]
    )
    model.replay_buffer = her_buffer
    print(f"✓ HER enabled with strategies: {config['her']['strategies']}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if True:
# try:
    model.learn(
        total_timesteps=config["total_steps"],
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model with run name
    final_model_path = f"{run.name}_final"
    model.save(final_model_path)
    
    # Upload final model to wandb
    if WANDB_MODE != "offline":
        artifact = wandb.Artifact(f"model_{run.name}", type="model")
        artifact.add_file(final_model_path + ".zip")
        run.log_artifact(artifact)
        print(f"WANDB run name: {run.name}")
    
    print(f"Training complete. Final model saved to {final_model_path}.zip")
    print(f"You can recover this run using: wandb artifact get {run.path}/model_{run.name}:latest")
    
# except Exception as e:
#     print(f"Training failed: {e}")
#     if WANDB_MODE != "offline":
#         run.finish()
#     raise

# finally:
#     if WANDB_MODE != "offline":
#         run.finish()


# ---------------------------------------------------------------------------
# (Optional) Resume from checkpoint
# ---------------------------------------------------------------------------

# model = SAC.load("checkpoints/sac_model_200000_steps", env=train_env)
# model.load_replay_buffer("checkpoints/sac_model_replay_buffer_200000_steps")
# model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, reset_num_timesteps=False)
