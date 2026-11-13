"""Regression tests for the SocNavGym / sn26 fixes (bug numbers refer to the bug report).
Run from a scratch directory (sn26 writes JSON files into the CWD):  python /path/to/tests/test_bugfixes.py
"""
import os, copy, tempfile
import numpy as np
import gymnasium as gym
import socnavgym  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "test_env.yaml")


def make(cfg=CFG, **edits):
    """Env from cfg with optional {"section.key": value} edits."""
    if edits:
        import yaml
        c = yaml.safe_load(open(cfg))
        for k, v in edits.items():
            sec, key = k.split("."); c[sec][key] = v
        fd, path = tempfile.mkstemp(suffix=".yaml"); os.close(fd); yaml.safe_dump(c, open(path, "w")); cfg = path
    return gym.make("SocNavGym-v2", config=cfg).unwrapped


def test_1_reward_in_range_on_first_step_success():
    env = make()
    env.reset(seed=11)
    env.robot.x, env.robot.y, env.robot.orientation = env.robot.goal_x, env.robot.goal_y, env.robot.goal_a
    _, r, term, trunc, info = env.step(np.zeros(3))
    assert info["SUCCESS"] and term and not trunc
    assert -1.0 <= r <= 1.0, r


def test_2_terminal_frame_is_scored():
    env = make()
    env.reset(seed=12)
    gx, gy, ga = env.robot.goal_x, env.robot.goal_y, env.robot.goal_a
    env.robot.x, env.robot.y, env.robot.orientation = gx - 0.6 * np.cos(ga), gy - 0.6 * np.sin(ga), ga
    env.step(np.zeros(3))
    seq = None
    orig = type(env.reward_calculator).predict
    def spy(self, model):
        nonlocal seq; seq = copy.deepcopy(self.data["sequence"]); return orig(self, model)
    type(env.reward_calculator).predict = spy
    try:
        while True:
            _, r, term, trunc, info = env.step(np.array([1.0, 0, 0]))
            if term or trunc: break
    finally:
        type(env.reward_calculator).predict = orig
    assert info["SUCCESS"]
    f = seq[-1]; d = np.hypot(f["robot"]["x"] - f["goal"]["x"], f["robot"]["y"] - f["goal"]["y"])
    assert d < f["goal"]["pos_threshold"], "the scored trajectory must end in the goal"


def test_3_metric_sees_all_humans_and_objects():
    env = make(**{"env.min_tables": 1, "env.max_tables": 1, "env.min_plants": 2, "env.max_plants": 2})
    env.reset(seed=3)
    rc = env.reward_calculator
    rc.add_to_trajectory(); f = rc.data["sequence"].pop()
    assert len(f["people"]) == len(env.all_humans)
    assert {p["id"] for p in f["people"]} == {h.id for h in env.all_humans}
    assert len(f["objects"]) == len(env.tables) + len(env.plants) + len(env.chairs) + len(env.laptops)
    assert all(o["shape"]["type"] in ("circle", "rectangle") for o in f["objects"])


def test_4_goal_angle_is_wrapped():
    env = make()
    env.reset(seed=7)
    env.robot.x, env.robot.y = env.robot.goal_x, env.robot.goal_y
    env.robot.goal_a = np.pi - 0.05; env.robot.orientation = -np.pi + 0.05      # true gap 0.1 rad < any threshold
    _, _, term, _, info = env.step(np.zeros(3))
    assert info["SUCCESS"] and term


def test_5_9_observations_within_bounds():
    env = make()
    space = env.observation_space
    bad = {}
    for s in range(5):
        obs, _ = env.reset(seed=100 + s)
        for t in range(150):
            obs, r, te, tr, _ = env.step(np.array([0.3, 0.3, 1.0]))     # spinning robot: the old wrap spikes appeared here
            for k in ("robot", "humans"):
                if not space[k].contains(obs[k]): bad[k] = bad.get(k, 0) + 1
            h = obs["humans"].reshape(-1, 8)
            assert np.abs(h[h[:, 4] != 0, 6]).max(initial=0) <= np.pi / env.TIMESTEP + 1e-3
            if te or tr: break
    assert not bad, bad


def test_6_human_ellipse_rotates_with_heading():
    env = make()
    env.reset(seed=3)
    h = env.static_humans[0]
    def coll_dist(dx, dy):
        lo, hi = 0.0, 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            env.robot.x, env.robot.y = h.x + dx * mid, h.y + dy * mid
            lo, hi = (mid, hi) if env.robot.collides(h) else (lo, mid)
        return lo
    h.orientation = 0.0;       front0, side0 = coll_dist(1, 0), coll_dist(0, 1)
    h.orientation = np.pi / 2; front1, side1 = coll_dist(0, 1), coll_dist(1, 0)
    assert abs(front0 - front1) < 1e-3 and abs(side0 - side1) < 1e-3 and side0 > front0 + 0.1


def test_10_no_wall_clock_in_humans_or_scene_generation():
    import inspect
    from socnavgym.envs.utils import human as H
    from socnavgym.envs import socnavenv_v2 as E
    import re
    assert "pytime.time()-" not in inspect.getsource(H)
    assert not re.search(r"^\s*start_time = time\.time\(\)", inspect.getsource(E), re.M)   # code lines only, not docstrings


def test_14_L_shaped_rooms():
    env = make(**{"env.set_shape": "L"})
    for s in range(8):
        env.reset(seed=s)
        assert len(env.walls) == 8
        for t in range(20):
            _, _, te, tr, _ = env.step(env.action_space.sample())
            if te or tr: break


def test_15_unbuildable_seed_does_not_hang():
    # dense furniture: some seeds cannot place every goal on the first attempt
    env = make(**{"env.set_shape": "rectangle", "env.min_tables": 1, "env.max_tables": 2, "env.min_plants": 1, "env.max_plants": 3,
                  "env.min_chairs": 1, "env.max_chairs": 3, "env.max_laptops": 1})
    for s in (7, 16, 30):
        env.reset(seed=s)                       # used to loop forever


def test_sn26_terminated_xor_truncated():
    env = make()
    env.reset(seed=1)
    for t in range(600):
        _, r, te, tr, info = env.step(np.zeros(3))
        if te or tr: break
    assert bool(te) != bool(tr)


def test_wrappers_import_and_step():
    from socnavgym.wrappers import DiscreteActions, NoisyObservations, PartialObservations, WorldFrameObservations
    for W, kw in ((DiscreteActions, {}), (NoisyObservations, dict(mean=0, std_dev=0.1)), (PartialObservations, dict(fov_angle=2 * np.pi / 3, range=5.0)), (WorldFrameObservations, {})):
        env = W(gym.make("SocNavGym-v2", config=CFG), **kw)
        env.reset(seed=0)
        env.step(env.action_space.sample())


def test_21_sfm_humans_walk():
    env = make(**{"human.human_policy": "sfm"})
    env.reset(seed=1); pos = {h.id: (h.x, h.y) for h in env.dynamic_humans}; sp = []
    for t in range(100):
        _, _, te, tr, _ = env.step(np.zeros(3))
        for h in env.dynamic_humans:
            sp.append(np.hypot(h.x - pos[h.id][0], h.y - pos[h.id][1]) / env.TIMESTEP); pos[h.id] = (h.x, h.y)
        if te or tr: break
    assert np.mean(np.array(sp) > 0.1) > 0.5, "SFM humans used to stand still forever"


def test_speed_range_and_policy_mix():
    import yaml
    c = yaml.safe_load(open(CFG)); c["human"]["human_policy"] = {"orca": 0.5, "sfm": 0.5}; c["human"]["speed_range"] = [0.3, 1.1]
    fd, path = tempfile.mkstemp(suffix=".yaml"); os.close(fd); yaml.safe_dump(c, open(path, "w"))
    env = gym.make("SocNavGym-v2", config=path).unwrapped
    prefs, pols = [], set()
    for s in range(10):
        env.reset(seed=s)
        for h in env.dynamic_humans: prefs.append(env.human_speed(h)); pols.add(h.policy)
    assert pols == {"orca", "sfm"} and 0.3 <= min(prefs) and max(prefs) <= 1.1 and np.std(prefs) > 0.1


def test_23_seeded_scene_does_not_depend_on_previous_episode():
    env = make(**{"env.set_shape": "random"})
    def scene(seed):
        env.reset(seed=seed); return (env.shape, round(env.MAP_X, 6), round(env.MAP_Y, 6), round(env.robot.x, 6))
    fresh = [scene(s) for s in range(6)]
    again = [scene(s) for s in reversed(range(6))][::-1]          # same seeds, different predecessors
    assert fresh == again


def test_max_linear_speed_caps_the_norm():
    """max_linear_speed limits |(vx, vy)| (m/s) keeping the direction; rotation stays limited by max_rotation (rad/s)."""
    for cap, want in ((None, np.hypot(1.0, 1.0)), (1.0, 1.0)):
        env = make(**({"robot.max_linear_speed": cap} if cap else {}))
        env.reset(seed=0); x0, y0 = env.robot.x, env.robot.y
        env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v = np.hypot(env.robot.vel_x, env.robot.vel_y)
        assert abs(v - want) < 1e-5 and abs(env.robot.vel_x - env.robot.vel_y) < 1e-6, (cap, v)
        assert abs(np.hypot(env.robot.x - x0, env.robot.y - y0) - want * env.TIMESTEP) < 1e-4
    env = make(**{"robot.max_linear_speed": 1.0}); env.reset(seed=0)
    env.step(np.array([0.6, 0.0, 1.0], dtype=np.float32))                       # under the cap: unchanged
    assert abs(env.robot.vel_x - 0.6) < 1e-6 and abs(env.robot.vel_a - env.MAX_ROTATION) < 1e-6


def test_max_lateral_speed():
    env = make(**{"robot.max_lateral_speed": 0.2, "robot.max_linear_speed": 1.0}); env.reset(seed=0)
    env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    assert abs(env.robot.vel_y + 0.2 / np.hypot(1.0, 0.2)) < 1e-6 and abs(np.hypot(env.robot.vel_x, env.robot.vel_y) - 1.0) < 1e-5   # lateral cap, then norm cap
    env.step(np.array([0.0, 0.1, 0.0], dtype=np.float32))                       # under the cap: unchanged
    assert abs(env.robot.vel_y - 0.1) < 1e-6


def test_25_robot_spawns_clear_of_humans():
    """No human may start within reach of the robot (ellipse major semi-axis + spawn clearance), and no episode may
    start with a collision caused by a human turning in place."""
    env = make()
    for s in range(30):
        env.reset(seed=s)
        for h in env.reward_calculator.all_humans():
            assert np.hypot(h.x - env.robot.x, h.y - env.robot.y) >= env.ROBOT_RADIUS + env.HUMAN_DIAMETER / 2 + env.SPAWN_CLEARANCE - 1e-9


def test_26_no_walls_means_no_map_limit():
    """Without walls the robot may leave the spawn rectangle: no termination, no collision, OUT_OF_MAP stays False.
    With env.bounded_without_walls the old rule applies; walled scenes are unaffected by the key."""
    def drive_out(**env_over):
        env = make(**{"env.set_shape": "no-walls", **{"env." + k: v for k, v in env_over.items()}})
        env.reset(seed=0)
        env.robot.x = env.MAP_X / 2 + 1.0                       # outside the spawn rectangle
        _, _, te, tr, info = env.step(np.zeros(3, np.float32))
        return te, tr, info
    te, tr, info = drive_out()
    assert not te and not tr and not info["OUT_OF_MAP"] and not info["COLLISION"]
    te, tr, info = drive_out(bounded_without_walls=True)
    assert te and info["OUT_OF_MAP"]


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp(prefix="socnavgym_tests_"))
    import warnings; warnings.filterwarnings("ignore")
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_")):
        try:
            fn(); print("PASS", name, flush=True)
        except Exception as e:                    # noqa: BLE001
            fails += 1; print("FAIL", name, type(e).__name__, e, flush=True)
    raise SystemExit(1 if fails else 0)
