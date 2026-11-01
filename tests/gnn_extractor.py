import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

# Ordered list of entity keys — must match the order in SocNavEnv_v2.observation_space
# `robot`` too, but not explicitly in the list
ENTITY_KEYS = ["humans", "laptops", "tables", "plants", "chairs", "walls" ]
# One-hot type indices: robot=0, then ENTITY_KEYS in order
N_TYPES = 1 + len(ENTITY_KEYS)  # 7

ENTITY_OBS_DIM = 8   # all keys (robot included) produce 8-dim observations
RADIUS_IDX     = 4   # index of the radius field inside each 8-dim slot (0 only for padding)


class FilterZeroObsWrapper(gym.ObservationWrapper):
    """
    Removes Dict observation keys whose space has shape (0,) from both the
    observation space and the observations themselves.

    SB3's obs_to_tensor does obs_.reshape((-1, *space.shape)), which fails in
    NumPy ≥2.0 when space.shape == (0,) because the -1 dimension is
    indeterminate (0/0 is undefined).  Stripping these empty keys before SB3
    ever sees them avoids the reshape entirely.

    Just to make it clear, as it could potentially be misunderstood... This filters out
    keys that have no entities at all (empty tensor). It does not filter padding. Padding
    is filtered when we build the graph.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "FilterZeroObsWrapper requires a Dict observation space"
        )
        self.valid_keys = [
            k for k, sp in env.observation_space.spaces.items()
            if np.prod(sp.shape) > 0
        ]
        self.observation_space = spaces.Dict(
            {k: env.observation_space.spaces[k] for k in self.valid_keys}
        )

    def observation(self, obs: dict) -> dict:
        return {k: obs[k] for k in self.valid_keys}


class GATv2Extractor(BaseFeaturesExtractor):
    """
    GATv2-based feature extractor for SocNavGym-v2 Dict observations.

    Each entity (robot, humans, laptops, tables, plants, chairs, wall segments)
    becomes a graph node.  All entities share the same 8-dim observation format.
    A 7-dim one-hot entity-type vector is appended to each observation, giving
    15-dim node features.  Padding slots (radius field == 0) are stripped before
    graph construction.

    Keys absent from the observation (e.g. filtered by FilterZeroObsWrapper
    because MAX_* == 0) are silently skipped.

    All real nodes are connected in a fully-connected directed graph (including
    self-loops).  After GATv2Conv message passing, global mean pooling over the
    real nodes produces the fixed-size feature vector expected by SB3 policies.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        in_dim = ENTITY_OBS_DIM + N_TYPES  # 8 + 7 = 15

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim,
                hidden_dim // n_heads,
                heads=n_heads,
                concat=True,
                add_self_loops=True,
            )
            for _ in range(n_layers)
        ])
        # LayerNorm applied before each GATv2 layer (pre-norm residual style)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.act = nn.ELU()
        # Readout takes robot node embedding || global mean — 2× hidden_dim input
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, features_dim),
            nn.ReLU(),
        )

    def _build_graph(self, obs: dict, b: int) -> Data:
        device = obs["robot"].device
        chunks = []  # list of (N_k, 15) tensors to be concatenated

        # Robot node — type index 0
        oh = torch.zeros(N_TYPES, device=device)
        oh[0] = 1.0
        chunks.append(torch.cat([obs["robot"][b], oh]).unsqueeze(0))  # (1, 15)

        # Entity nodes — type indices 1..6; skip keys absent from the obs dict
        for type_idx, key in enumerate(ENTITY_KEYS, start=1):
            if key not in obs:
                continue
            slots = obs[key][b].reshape(-1, ENTITY_OBS_DIM)
            real = slots[slots[:, RADIUS_IDX] != 0]  # drop zero-padded slots
            if real.shape[0] == 0:
                continue
            oh = torch.zeros(real.shape[0], N_TYPES, device=device)
            oh[:, type_idx] = 1.0
            chunks.append(torch.cat([real, oh], dim=1))  # (N_k, 15)

        x = torch.cat(chunks, dim=0)  # (N_real, 15)
        N = x.shape[0]
        idx = torch.arange(N, device=device)
        edge_index = torch.stack([idx.repeat_interleave(N), idx.repeat(N)])  # fully connected + self-loops

        return Data(x=x, edge_index=edge_index)

    def forward(self, obs: dict) -> torch.Tensor:
        B = obs["robot"].shape[0]
        graphs = [self._build_graph(obs, b) for b in range(B)]
        batch = Batch.from_data_list(graphs)

        h = self.act(self.input_proj(batch.x))
        for layer, norm in zip(self.gat_layers, self.norms):
            h = h + self.act(layer(norm(h), batch.edge_index))  # pre-norm residual

        # Robot is always node 0 within each graph; batch.ptr[i] is its index in the batch
        robot_h = h[batch.ptr[:-1]]                # (B, hidden_dim)
        pooled  = global_mean_pool(h, batch.batch) # (B, hidden_dim)
        combined = torch.cat([robot_h, pooled], dim=-1)  # (B, 2*hidden_dim)
        return self.readout(combined)               # (B, features_dim)


# --- Smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    import socnavgym
    import os

    config = os.path.join(os.path.dirname(__file__), "test_env.yaml")
    env = gym.make("SocNavGym-v2", config=config)
    env = FilterZeroObsWrapper(env)
    obs, _ = env.reset()

    print("Observation keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}")

    # Wrap numpy obs in a batch-of-1 tensor dict
    obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}

    extractor = GATv2Extractor(
        observation_space=env.observation_space,
        hidden_dim=128, n_heads=4, n_layers=2, features_dim=256
    )
    extractor.eval()
    with torch.no_grad():
        out = extractor(obs_t)

    print(f"\nExtractor output shape: {out.shape}  (expected: (1, 256))")
    assert out.shape == (1, 256), f"Unexpected shape: {out.shape}"
    print("Smoke test passed ✓")
