"""Smoke tests for human_policy "learned". Run from a scratch directory: python /path/to/tests/test_learned_humans.py"""
import os, tempfile
import numpy as np
import gymnasium as gym
import yaml
import socnavgym  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))


def make():
    c = yaml.safe_load(open(os.path.join(HERE, "test_env.yaml"))); c["human"]["human_policy"] = "learned"
    fd, path = tempfile.mkstemp(suffix=".yaml"); os.close(fd); yaml.safe_dump(c, open(path, "w"))
    return gym.make("SocNavGym-v2", config=path).unwrapped


def rollout(env, seed, steps=60):
    env.reset(seed=seed); traj = []
    for _ in range(steps):
        _, _, te, tr, _ = env.step(np.zeros(3, np.float32))
        traj.append([(h.x, h.y) for h in env.dynamic_humans])
        if te or tr: break
    return np.array(traj)


def test_learned_humans_walk_at_human_speeds():
    env = make()
    assert all(h.policy == "learned" for h in env.dynamic_humans)
    T = rollout(env, 3)
    v = np.linalg.norm(np.diff(T, axis=0), axis=-1).ravel() / env.TIMESTEP
    moving = v[v > 0.3]
    assert len(moving) > 0.3 * len(v), "learned humans should mostly be walking"
    assert 0.5 < np.median(moving) < 1.6, np.median(moving)


def test_learned_humans_deterministic_per_seed():
    env = make()
    a, b = rollout(env, 5), rollout(env, 5)
    assert a.shape == b.shape and np.allclose(a, b)


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp(prefix="socnavgym_tests_"))
    import warnings; warnings.filterwarnings("ignore")
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_")):
        try: fn(); print("PASS", name, flush=True)
        except Exception as e: fails += 1; print("FAIL", name, type(e).__name__, e, flush=True)   # noqa: BLE001
    raise SystemExit(1 if fails else 0)
