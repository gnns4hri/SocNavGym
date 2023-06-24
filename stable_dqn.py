import gym
import socnavgym
import torch
import torch.nn as nn
import torch.nn.functional as F
from socnavgym.wrappers import DiscreteActions
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from math import sqrt
import argparse
from comet_ml import Experiment
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


class CometMLCallback(CheckpointCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, run_name:str, save_path:str, project_name:str, api_key:str, verbose=0):
        # super(CometMLCallback, self).__init__(verbose)
        super(CometMLCallback, self).__init__(save_freq=25000, save_path=save_path, verbose=verbose)
        print("Logging using comet_ml")
        self.run_name = run_name
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            parse_args=False   
        )
        self.experiment.set_name(self.run_name)

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        metrics = {
            "rollout/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.locals['self'].ep_info_buffer]),
            "rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.locals['self'].ep_info_buffer])
        }
        if len(self.locals['self'].ep_success_buffer) > 0:
            metrics["rollout/success_rate"] = safe_mean(self.locals['self'].ep_success_buffer)

        l = [
            "train/loss",
            "train/n_updates",
        ]

        for val in l:
            if val in self.logger.name_to_value.keys():
                metrics[val] = self.logger.name_to_value[val]

        step = self.locals['self'].num_timesteps

        self.experiment.log_metrics(metrics, step=step)
        

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-r", "--run_name", help="name of comet_ml run", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-p", "--project_name", help="project name in comet ml", required=True)
ap.add_argument("-a", "--api_key", help="api key to your comet ml profile", required=True)
ap.add_argument("-d", "--use_deep_net", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-g", "--gpu", help="gpu id to use", required=False, default="0")
args = vars(ap.parse_args())

env = gym.make("SocNavGym-v1", config=args["env_config"])
env = DiscreteActions(env)

net_arch = {}

if not args["use_deep_net"]:
    net_arch = [512, 256, 128, 64]

else:
    net_arch = [512, 256, 256, 256, 128, 128, 64]

policy_kwargs = {"net_arch" : net_arch}

device = 'cuda:'+str(args["gpu"]) if torch.cuda.is_available() else 'cpu'
model = DQN("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)
callback = CometMLCallback(args["run_name"], args["save_path"], args["project_name"], args["api_key"])
model.learn(total_timesteps=50000*200, callback=callback)
model.save(args["save_path"])