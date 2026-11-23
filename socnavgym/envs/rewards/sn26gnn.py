"""sn26gnn: the SN26 learned metric with a graph network (reward_file: "sn26gnn").

The metric of https://github.com/SocNavData/SN26_metric (code copied to rewards/SN26GNN/): every frame of the trajectory is
a graph (robot, goal, people, objects, walls; all connected to the robot) that a GAT turns into a vector; a GRU reads the
sequence of these vectors together with the hand-crafted per-frame metrics and the context features, and a small MLP gives
a score p in [0, 1]. As with "sn26" (the GRU-only metric, rewards/sn26.py) the reward is 0 on every step and 2p - 1 on the
last one, and the trajectory is recorded in the same way (this class reuses sn26's recording).

Weights: rewards/SN26GNN/checkpoints/gnn_gru_model.pytorch, or the file named by the environment variable
SN26GNN_CHECKPOINT. They are not in the repository of the metric (no automatic download yet) and not in git here.
Needs torch_geometric."""
import math
import os
import warnings
import torch
from torch_geometric.data import Batch

from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2
from socnavgym.envs.rewards.reward_api import RewardAPI
from socnavgym.envs.rewards.sn26 import Reward as SN26Reward
from socnavgym.envs.rewards.SN26GNN.datasetHomo import SocNavHomoDataset, collate
from socnavgym.envs.rewards.SN26GNN.modelHomo import HybridModel
from socnavgym.envs.rewards.SN26GNN.data_conversions import sequence_to_tensor
from socnavgym.envs.rewards.SN26GNN.data_normalization import tensor_transform_to_goal_fr
from socnavgym.envs.rewards.SN26GNN import metrics as _metrics

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


def _inputs_ref(data, context, thr):
    """The network's inputs for a recorded trajectory, as SN26_metric builds them: one torch_geometric graph per frame, one
    node at a time (SocNavHomoDataset._structure_to_graph), then joined into a batch (collate). -> (dataset, graphs, metrics, lengths)"""
    # the dataset adds keys to every frame: it works on copies so that `data` stays as recorded
    t_data = dict(data); t_data["sequence"] = [dict(f) for f in data["sequence"]]
    ds = SocNavHomoDataset(t_data, data_path=None, context_path=os.path.join(HERE, "anthropic_claude_context.csv"),
                           overwrite_contexts=context, timestamp_threshold=thr)
    graphs, metrics, _, slengths, _ = collate([ds.get(0)])
    return ds, graphs, metrics, slengths


def frames_batch(ds, nd, T):
    """The graphs of ALL the frames of a trajectory at once, as one batch, and the per-frame metrics: the same tensors as
    SocNavHomoDataset._structure_to_graph for every frame + collate (same node order, same edge order, same rounding), without
    a Python loop over frames and nodes and without one Data object per frame (user, 2026-11-16: "precomputing distances, sin
    and cos", "avoiding loops in the loading of the graphs", "natively generating batches of graphs").
    ds: a SocNavHomoDataset (for its feature names and its walls); nd: the normalised dictionary of the trajectory; T: frames.
    The sines and cosines are taken with math.sin / math.cos, as there (torch.sin may differ in the last bit)."""
    F = ds.all_features; ix = F.index; f64 = torch.float64
    def sc(t):
        v = t.reshape(-1).tolist()
        return (torch.tensor([math.sin(a) for a in v], dtype=f64).reshape(t.shape), torch.tensor([math.cos(a) for a in v], dtype=f64).reshape(t.shape))
    rob, goal, ppl, obj, met, cm = nd["robot"], nd["goal"], nd["people"], nd["objects"], nd["metrics"], nd["computed_metrics"]
    Nh = ppl["x"].shape[1]; No = obj["x"].shape[1]
    pts, ids, wa, wl = ds._get_walls(nd["walls"]) if nd["walls"] is not None else ([], [], [], [])   # once, not at every frame
    Nw = len(pts); n = 2 + Nh + No + Nw; X = torch.zeros(T, n, len(F), dtype=f64)
    rx, ry = rob["x"], rob["y"]; s, c = sc(rob["a"])
    for k, v in (("robot", 1.0), ("r_x", rx), ("r_y", ry), ("r_sin_a", s), ("r_cos_a", c), ("r_w", rob["w"]), ("r_l", rob["l"]),
                 ("dlin_gr", cm["dist_to_goal_pos"]), ("dang_gr", cm["dist_to_goal_angle"])): X[:, 0, ix(k)] = v
    s, c = sc(goal["a"])
    for k, v in (("goal", 1.0), ("g_x", goal["x"]), ("g_y", goal["y"]), ("g_sin_a", s), ("g_cos_a", c), ("th_pos", goal["th_p"]), ("th_angle", goal["th_a"])):
        X[:, 1, ix(k)] = v
    if Nh:
        px, py = ppl["x"], ppl["y"]; s, c = sc(ppl["a"]); h = slice(2, 2 + Nh)
        for k, v in (("human", 1.0), ("h_x", px), ("h_y", py), ("h_sin_a", s), ("h_cos_a", c), ("dist_hr", met["dist_human"]),
                     ("dcenter_hr", torch.sqrt((px - rx[:, None]) ** 2 + (py - ry[:, None]) ** 2))): X[:, h, ix(k)] = v
    if No:
        ox, oy = obj["x"], obj["y"]; s, c = sc(obj["a"]); o = slice(2 + Nh, 2 + Nh + No)
        for k, v in (("object", 1.0), ("o_x", ox), ("o_y", oy), ("o_sin_a", s), ("o_cos_a", c), ("dist_or", met["dist_object"]), ("o_w", obj["w"]),
                     ("o_l", obj["l"]), ("dcenter_or", torch.sqrt((ox - rx[:, None]) ** 2 + (oy - ry[:, None]) ** 2))): X[:, o, ix(k)] = v
    if Nw:
        w = slice(2 + Nh + No, n); row = lambda v: torch.tensor([float(a) for a in v], dtype=f64)
        for k, v in (("wall", 1.0), ("w_x", row([p[0] for p in pts])), ("w_y", row([p[1] for p in pts])), ("w_sin_a", row([math.sin(a) for a in wa])),
                     ("w_cos_a", row([math.cos(a) for a in wa])), ("dist_wr", met["dist_walls"][:, ids]), ("w_w", row(wl)), ("w_l", 0.01)): X[:, w, ix(k)] = v
    there = torch.cat([torch.ones(T, 2, dtype=torch.bool), ppl["exists"].bool().reshape(T, Nh), obj["exists"].bool().reshape(T, No),
                       torch.ones(T, Nw, dtype=torch.bool)], 1)                      # the nodes of every frame: robot, goal, the people and objects in it, walls
    x = X[there].float(); cnt = there.sum(1); ptr = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)])
    # every node of a frame <-> its robot (the first node), per frame: [others -> robot], then [robot -> others]
    m = cnt - 1; e = torch.arange(int(2 * m.sum())); mm = torch.repeat_interleave(m, 2 * m); first = torch.repeat_interleave(ptr[:-1], 2 * m)
    pos = e - torch.repeat_interleave(2 * m.cumsum(0) - 2 * m, 2 * m); back = pos >= mm; other = first + 1 + pos - mm * back
    ei = torch.stack([torch.where(back, first, other), torch.where(back, other, first)])
    mets = torch.cat([torch.stack([cm[k].to(torch.float32).reshape(T) for k in cm], 1),
                      torch.stack([nd["context"][k].to(torch.float32).reshape(T) for k in nd["context"]], 1)], 1)
    return Batch(x=x, edge_index=ei, batch=torch.repeat_interleave(torch.arange(T), cnt), ptr=ptr), mets


FAST = [os.environ.get("SN26GNN_FAST", "1") != "0"]   # frames_batch instead of SN26_metric's frame-by-frame loading (the same tensors)
_ref = {}                                             # context -> (a SocNavHomoDataset of this process, its context row)


def inputs(data, context=CONTEXT, thr=0.1):
    """-> (graphs, metrics, lengths) of a recorded trajectory, for the network. The first trajectory of a process (per context)
    is loaded both ways, SN26_metric's and frames_batch: they must give the same tensors, or the fast way is switched off for
    the process (with a warning). From then on only the fast way is used."""
    if not FAST[0] or context not in _ref:
        ds, graphs, mets, slengths = _inputs_ref(data, context, thr)
        if not FAST[0]: return graphs, mets, slengths
        _ref[context] = (ds, ds.context_df.loc[context.rstrip()].to_dict())
    else: graphs = None
    ds, ctx = _ref[context]
    try:
        t_data = dict(data); t_data["sequence"] = [dict(f) for f in data["sequence"]]
        td, T = sequence_to_tensor(t_data, thr, ctx)
        nd = _metrics.normalize_features(_metrics.compute_metrics(tensor_transform_to_goal_fr(td)), ds.max_values)
        g2, m2 = frames_batch(ds, nd, T)
        ok = graphs is None or (torch.equal(g2.x, graphs.x) and torch.equal(g2.edge_index, graphs.edge_index) and torch.equal(g2.ptr, graphs.ptr)
                                and torch.equal(m2, mets) and int(slengths[0]) == T)
        why = "does not give the tensors of SN26_metric's loader"
    except Exception as e:                                                          # never the reason for a job to die: the loader is still there
        ok = False; why = f"failed ({type(e).__name__}: {e})"
    if not ok:
        if graphs is None: _, graphs, mets, slengths = _inputs_ref(data, context, thr)   # (raises what it always raised if the trajectory itself is the problem, e.g. a single frame)
        warnings.warn(f"sn26gnn: frames_batch {why}; using the loader (slow) in this process")
        FAST[0] = False
        return graphs, mets, slengths
    return g2, m2, torch.tensor([T], dtype=torch.long)


def score(data, context=CONTEXT, path=None):
    """p in [0, 1]: the network's score of a recorded trajectory (the dictionary that sn26's Reward records: walls, sequence),
    as SN26_metric's sn26gnn.Reward.compute_reward computes it. Any reward that records the trajectory in this format can
    be scored with it afterwards (the benchmark scores the same episode with sn26 and with sn26gnn)."""
    model, thr = load(path)
    graphs, metrics, slengths = inputs(data, context, thr)
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
        return score(self.data, self.context())          # the episode's social context (the default unless its scenario names another)
