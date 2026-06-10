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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback



try:
    HER_THRESHOLD = float(os.getenv("HER_THRESHOLD"))
except:
    HER_THRESHOLD = float("nan")

try:
    WANDB_MODE = os.getenv("WANDB_MODE", "online").lower()
except:
    WANDB_MODE = "online"

if WANDB_MODE != "offline":
    from wandb.integration.sb3 import WandbCallback
    import wandb


from dict_subset_wrapper import DictToFlatWrapper
from her_wrapper import HERGoalEnvWrapper

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
    do_her = config.get("her", {}).get("enabled", False)
    if do_her:
        if np.isnan(HER_THRESHOLD):
            HER_THRESHOLD = config["her"]["threshold"]
        her_value = f"_HER({HER_THRESHOLD})"
    else:
        her_value = ""

    config["wandb"]["name"] = f"sac_{timestamp}{her_value}"

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
    # Update config with wandb run ID for model saving
    config["wandb"]["run_id"] = run.id
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_upload = [
        ('train_config', 'train_config.yaml', 'config'),
        ('test_env_config', 'test_env.yaml', 'config'),
        ('her_wrapper', 'her_wrapper.py', 'code'),
        ('train_script', 'train.py', 'code')
    ]
    for artifact_name, filename, artifact_type in files_to_upload:
        filepath = os.path.join(script_dir, filename)
        try:
            if os.path.exists(filepath):
                artifact = wandb.Artifact(artifact_name, type=artifact_type)
                artifact.add_file(filepath)
                run.log_artifact(artifact)
            else:
                print(f"Warning: File {filename} not found at {filepath}")
        except Exception as e:
            print(f"Failed to upload {filename} as artifact: {e}")
else:
    run = namedtuple('rubbish', ["id", "run", "name"])("offline", "offline", config["wandb"]["name"])
    config["wandb"]["run_id"] = run.id
    


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env():
    """Instantiate, wrap, and monitor a single environment."""
    config_path = os.path.join(os.path.dirname(__file__), config["env_config"])
    env = gym.make("SocNavGym-v2", config=config_path)

    # Convert the environment to dictionary
    env = DictToFlatWrapper(env, keys=config["keys"])

    # Apply HER wrapper (even if disabled, for debugging purposes)
    if config.get("her", {}).get("enabled", False):
        env = HERGoalEnvWrapper(env, config["her"])

    env = Monitor(env, filename=None)
    return env


# ---------------------------------------------------------------------------
# Build vectorised envs
# ---------------------------------------------------------------------------

train_env = make_vec_env(make_env, n_envs=config["n_envs"])
eval_env  = make_vec_env(make_env, n_envs=1)
print("EVAL ENV", eval_env)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

setup_directories(config)


class HERStatusCallback(WandbCallback):
    """Callback to log HER status to wandb during evaluation."""

    def __init__(self, verbose=0):
        super(HERStatusCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log HER status during evaluation
        wandb.log({"her_active": float(HERGoalEnvWrapper.active)}, commit=False)
        return True





class RewardThresholdCallback(EvalCallback):
    def __init__(self, threshold: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.threshold_reached = False

    def _on_step(self) -> bool:
        # Run the parent eval logic (handles periodic evaluation internally)
        continue_training = super()._on_step()

        # last_mean_reward is set by EvalCallback after each evaluation
        if self.last_mean_reward >= self.threshold and not self.threshold_reached:
            self.threshold_reached = True
            self._on_threshold_reached(self.last_mean_reward)
        return continue_training

    def _on_threshold_reached(self, reward: float):
        print("Mean reward threshold reached!")
        HERGoalEnvWrapper.active = False




if WANDB_MODE != "offline":
    checkpoint_callback = CheckpointCallback(
        save_freq=config["eval_freq"], save_path=config["checkpoint_dir"], name_prefix=f"sac_model_{run.name}",
        save_replay_buffer=False, save_vecnormalize=True)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=config["best_model_dir"], log_path=config["best_model_dir"],
        eval_freq=config["eval_freq"], n_eval_episodes=config["eval_episodes"],
        deterministic=True, render=False)

    wandb_callback = WandbCallback(gradient_save_freq=config["eval_freq"], model_save_path=f"models/{run.id}", model_save_freq=config["eval_freq"], verbose=2)
    her_status_callback = HERStatusCallback()
    reward_threshold_callback = RewardThresholdCallback(threshold=config["her"]["threshold"], eval_env=eval_env, eval_freq=config["eval_freq"],
        n_eval_episodes=30, deterministic=True, verbose=1)
    callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback, reward_threshold_callback, her_status_callback])
else:
    callbacks = None

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Extract SAC-specific hyperparameters from config
sac_params = {k: v for k, v in config["sac_hyperparams"].items() if v is not None}

model = SAC(policy="MlpPolicy", env=train_env, verbose=config["verbose"], tensorboard_log=f"runs/{run.id}", **sac_params)


# Get HER wrapper environment
for her_env in model.env.envs:
    # Unwrap the environment to get the HER wrapper
    while hasattr(her_env, 'env') and not hasattr(her_env, 'THIS_IS_THE_HER_WRAPPER'):
        print("her_env", type(her_env), her_env, "going down")
        her_env = her_env.env
    print("USING", type(her_env), her_env, "going down")
HERGoalEnvWrapper.replay_buffer = model.replay_buffer
print(f"✓ HER enabled")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
try:
    model.learn(total_timesteps=config["total_steps"], callback=callbacks, progress_bar=(WANDB_MODE != "offline"))
    
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
    
except Exception as e:
    print(f"Training failed: {e}")
    if WANDB_MODE != "offline":
        run.finish()
    raise

finally:
    if WANDB_MODE != "offline":
        run.finish()

