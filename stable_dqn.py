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


class MLP(nn.Module):
    """
    Class for a Multi Layered Perceptron. LeakyReLU activations would be applied between each layer.

    Args:
    input_layer_size (int): The size of the input layer
    hidden_layers (list): A list containing the sizes of the hidden layers
    last_relu (bool): If True, then a LeakyReLU would be applied after the last hidden layer
    """
    def __init__(self, input_layer_size:int, hidden_layers:list, last_relu=False) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_layer_size, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
        for i in range(len(hidden_layers)-1):
            if i != (len(hidden_layers)-2):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
                nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
            elif last_relu:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
                nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
            else:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.network(x)
        return x
    
class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            )
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        # gain_last_layer = nn.init.calculate_gain('tanh', 0.01)
        nn.init.xavier_uniform_(self.linear[0].weight, gain=gain)
    
    def forward(self, x):
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list) -> None:
        super().__init__()
        self.input_emb1 = input_emb1
        self.input_emb2 = input_emb2
        self.d_model = d_model
        self.d_k = d_k
        self.mlp_hidden_layers = mlp_hidden_layers
        self.embedding1 = Embedding(input_dim=input_emb1, output_dim=d_model)
        self.embedding2 = Embedding(input_dim=input_emb2, output_dim=d_model)
        self.key_net = nn.Sequential(
            nn.Linear(d_model, d_k),
            nn.LeakyReLU()
            )
        self.query_net = nn.Sequential(
            nn.Linear(d_model, d_k),
            nn.LeakyReLU()
            )

        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_k),
            nn.LeakyReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP(2*d_model, mlp_hidden_layers) if mlp_hidden_layers is not None else None
        self.layer_norm = nn.LayerNorm([1, d_model])

        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        # gain_last_layer = nn.init.calculate_gain('leaky_relu', 0.01)
        nn.init.xavier_uniform_(self.key_net[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.query_net[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.value_net[0].weight, gain=gain)


    def forward(self, inp1, inp2):
        # passing through the embedding layers

        # inp1.shape = (b, 1, input_emb1)
        # inp2.shape = (b, n-1, input_emb2)

        embedding1 = self.embedding1(inp1)  # embedding1.shape = (b, 1, d_model)
        embedding2 = self.embedding2(inp2)  # embedding2.shape = (b, n-1, d_model)

        # query net
        q = self.query_net(embedding1)  # q.shape = (b, 1, d_k)
        # key net
        k = self.key_net(embedding2)  # k.shape = (b, n-1, d_k)
        # value net
        v = self.value_net(embedding2)  # v.shape = (b, n-1, d_k)
        # scaled dot product attention
        attention_matrix = self.softmax(torch.matmul(q, k.transpose(1,2))/sqrt(self.d_k))  # attention_matrix.shape = (b, 1, n-1)
        attention_value = torch.matmul(attention_matrix, v)  # attention_value.shape = (b, 1, d_k)
        # add and norm is applied only when d_model == d_k
        if self.d_k == self.d_model:
            embedding2_mean = torch.mean(embedding2, dim=1, keepdim=True)  # embedding2_mean.shape = (b, 1, d_k)
            assert(attention_value.shape == embedding2_mean.shape == embedding1.shape), "something wrong in the shapes of tensors"
            x = attention_value + embedding2_mean + embedding1
            x = self.layer_norm(x)
        else:
            x = attention_value
        
        # x.shape = (b, 1, d_k)

        # feed forward network
        if self.mlp is not None:
            q = self.mlp(x)
            return q
        else:
            return x


class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        self.transformer = Transformer(8, 14, 512, 512, None)

        # Update the features dim manually
        self._features_dim = 512

        print("Using transformer for feature extraction")

    def preprocess_observation(self, obs):
        """
        To convert dict observation to numpy observation
        """
        assert(type(obs) == dict)
        observation = torch.tensor([], device=obs["goal"].device).float()
        if "goal" in obs.keys() : observation = torch.cat((observation, obs["goal"]) , dim=1)
        if "humans" in obs.keys() : observation = torch.cat((observation, obs["humans"]) , dim=1)
        if "laptops" in obs.keys() : observation = torch.cat((observation, obs["laptops"]) , dim=1)
        if "tables" in obs.keys() : observation = torch.cat((observation, obs["tables"]) , dim=1)
        if "plants" in obs.keys() : observation = torch.cat((observation, obs["plants"]) , dim=1)
        if "walls" in obs.keys():
            observation = torch.cat((observation, obs["walls"]), dim=1)
        return observation

    
    def postprocess_observation(self, obs):
        """
        To convert a one-vector observation into two inputs that can be given to the transformer
        """
        if(len(obs.shape) == 1):
            obs = obs.reshape(1, -1)
        
        robot_state = obs[:, 0:8].reshape(obs.shape[0], -1, 8)
        entity_state = obs[:, 8:].reshape(obs.shape[0], -1, 14)
        
        return robot_state, entity_state

    def forward(self, observations):
        pre = self.preprocess_observation(observations)
        r, e = self.postprocess_observation(pre)
        out = self.transformer(r, e)
        out = out.squeeze(1)
        return out

class CometMLCallback(CheckpointCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, run_name:str, save_path:str, project_name:str, api_key:str, verbose=0):
        # super(CometMLCallback, self).__init__(verbose)
        super(CometMLCallback, self).__init__(save_freq=25000, save_path=save_path, verbose=verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
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
ap.add_argument("-u", "--use_transformer", help="True or False, based on whether you want a transformer based feature extractor", required=True, default=False)
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

if args["use_transformer"]:
    policy_kwargs = {"net_arch" : net_arch, "features_extractor_class": TransformerExtractor}
else:
    policy_kwargs = {"net_arch" : net_arch}

device = 'cuda:'+str(args["gpu"]) if torch.cuda.is_available() else 'cpu'
model = DQN("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)
callback = CometMLCallback(args["run_name"], args["save_path"], args["project_name"], args["api_key"])
model.learn(total_timesteps=50000*200, callback=callback)
model.save(args["save_path"])