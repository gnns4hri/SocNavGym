"""sn26gnn: the SN26 learned metric with a graph network (reward_file: "sn26gnn").

The metric of https://github.com/SocNavData/SN26_metric (code copied to rewards/SN26GNN/): every frame of the trajectory is
a graph (robot, goal, people, objects, walls; all connected to the robot) that a GAT turns into a vector; a GRU reads the
sequence of these vectors together with the hand-crafted per-frame metrics and the context features, and a small MLP gives
a score p in [0, 1]. As with "sn26" (the GRU-only metric, rewards/sn26.py) the reward is 0 on every step and 2p - 1 on the
last one, and the trajectory is recorded in the same way (this class reuses sn26's recording).

Weights: rewards/SN26GNN/checkpoints/gnn_gru_model.pytorch, or the file named by the environment variable
SN26GNN_CHECKPOINT. They are not in the repository of the metric (no automatic download yet) and not in git here.
Needs torch_geometric."""
import os
import torch

from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2
from socnavgym.envs.rewards.reward_api import RewardAPI
from socnavgym.envs.rewards.sn26 import Reward as SN26Reward
from socnavgym.envs.rewards.SN26GNN.datasetHomo import SocNavHomoDataset, collate
from socnavgym.envs.rewards.SN26GNN.modelHomo import HybridModel

HERE = os.path.join(os.path.dirname(__file__), "SN26GNN")


CONTEXT = SN26Reward.CONTEXT
DEVICE = torch.device("cpu")   # CPU on purpose, as for sn26: one pass per episode, and the score must not depend on the machine's GPU
_loaded = {}                   # checkpoint path -> (model, frame threshold): read once per process (the env re-creates the reward at every reset)


def checkpoint_path():
    return os.environ.get("SN26GNN_CHECKPOINT") or os.path.join(HERE, "checkpoints", "gnn_gru_model.pytorch")


def available():
    """Are the weights there?"""
    return os.path.exists(checkpoint_path())


def load(path=None):
    """-> (model, frame threshold) of the checkpoint."""
    path = path or checkpoint_path()
    if path not in _loaded:
        if not os.path.exists(path):
            raise FileNotFoundError(f"sn26gnn: no weights at {path}. Copy gnn_gru_model.pytorch there or set SN26GNN_CHECKPOINT "
                                    "(the weights are not in the SN26_metric repository).")
        c = torch.load(path, map_location=DEVICE, weights_only=True)
        model = HybridModel(c["num_layers"], gnn_input=c["gnn_input_size"], gnn_output=c["gnn_output"], rnn_hidden_channels=c["rnn_hidden_size"],
                            gnn_hidden_channels=c["gnn_hidden_size"], rnn_type=c["rnn_type"], gnn_heads=c["gnn_heads"], gnn_concat=c["gnn_concat"],
                            linear_layers=c["linear_layers"], rnn_activation=c["activation"], context_vars=c["context_features"],
                            metrics_vars=c["metric_features"], only_gnn=False, only_metrics=False).to(DEVICE)
        # weights saved by the earlier model code call the GRU "rnn_layer"; the repository's HybridModel calls it "seq_block.rnn_block"
        sd = {("seq_block.rnn_block." + k[len("rnn_layer."):] if k.startswith("rnn_layer.") else k): v for k, v in c["model_state_dict"].items()}
        model.load_state_dict(sd); model.eval()
        _loaded[path] = (model, c.get("frame_threshold", 0.1))
    return _loaded[path]


def score(data, context=CONTEXT, path=None):
    """p in [0, 1]: the network's score of a recorded trajectory (the dictionary that sn26's Reward records: walls, sequence),
    as SN26_metric's sn26gnn.Reward.compute_reward computes it. Any reward that records the trajectory in this format can
    be scored with it afterwards (the benchmark scores the same episode with sn26 and with sn26gnn)."""
    model, thr = load(path)
    # the dataset adds keys to every frame: it works on copies so that `data` stays as recorded
    t_data = dict(data); t_data["sequence"] = [dict(f) for f in data["sequence"]]
    ds = SocNavHomoDataset(t_data, data_path=None, context_path=os.path.join(HERE, "anthropic_claude_context.csv"),
                           overwrite_contexts=context, timestamp_threshold=thr)
    graphs, metrics, _, slengths, _ = collate([ds.get(0)])
    with torch.no_grad():
        out = model(graphs.to(DEVICE), metrics.to(DEVICE), slengths.to(DEVICE))
    return float(out[0][0])


class Reward(SN26Reward):
    def __init__(self, env: SocNavEnv_v2) -> None:
        RewardAPI.__init__(self, env)
        self.json_directory = os.path.join(os.getcwd(), "experimental_JSONs")   # only names a file that is never written
        self.device = DEVICE
        self.model, self.FRAME_THRESHOLD = load()
        self.reset()
        self.new_trajectory()

    def predict(self, model):
        if len(self.data.get("sequence", [])) == 0:
            return 0.0
        return score(self.data, self.CONTEXT)
