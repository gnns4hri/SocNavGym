"""Tests of hand-made scenarios (socnavgym/envs/utils/scenario.py)."""
import math, os, tempfile
import numpy as np
import pytest
import yaml

from test_bugfixes import make
from socnavgym.envs.utils.scenario import Scenario, ScenarioError, export, dump

STILL = np.zeros(3, np.float32)
FORWARD = np.array([1.0, 0.0, 0.0], np.float32)


def corridor(**people):
    return {"name": "t", "map": {"size": [9, 6], "walls": [[-4.5, -1, 4.5, -1], [-4.5, 1, 0.5, 1], [1.5, 1, 4.5, 1]]},
            "robot": {"x": -3.8, "y": 0, "theta": 0, "goal": {"x": 3.8, "y": 0, "theta": 0}}, "people": list(people.values())}


def test_scene_is_built_as_written():
    env = make(**{"env.max_tables": 1, "env.max_plants": 1})
    spec = corridor(a={"name": "a", "x": 1.0, "y": 1.6, "theta": -1.0, "behaviour": "static"},
                    b={"name": "b", "x": 3.0, "y": 0.0, "behaviour": "goals", "policy": "orca", "goals": [[-3.0, 0.0]]})
    spec["map"]["walls"] = spec["map"]["walls"][:1]
    spec["furniture"] = [{"type": "table", "x": 0.0, "y": -2.0, "theta": 0.3}, {"type": "plant", "x": -2.0, "y": 2.0, "radius": 0.25}]
    env.reset(seed=3, options={"scenario": spec})
    assert (env.MAP_X, env.MAP_Y) == (9, 6) and len(env.walls) == 5 and env.shape == "custom"
    assert (env.robot.x, env.robot.y, env.robot.orientation, env.robot.goal_x, env.robot.goal_y, env.robot.goal_a) == (-3.8, 0, 0, 3.8, 0, 0)
    assert [(h.x, h.y, h.orientation) for h in env.static_humans] == [(1.0, 1.6, -1.0)]
    b = env.dynamic_humans[0]
    assert (b.x, b.y, b.goal_x, b.goal_y, b.policy) == (3.0, 0.0, -3.0, 0.0, "orca")
    assert abs(b.orientation - math.pi) < 1e-9                        # faces its first goal when theta is not given
    t = env.tables[0]; assert (t.x, t.y, t.orientation, t.length, t.width) == (0.0, -2.0, 0.3, env.TABLE_LENGTH, env.TABLE_WIDTH)
    assert env.plants[0].radius == 0.25
    assert len(env.static_interactions + env.moving_interactions + env.h_l_interactions) == 0


def test_open_scene_has_no_walls_and_random_scenes_come_back():
    env = make()
    env.reset(seed=1, options={"scenario": {"map": {"size": [9, 9], "boundary": False},
                                            "robot": {"x": 0, "y": 0, "goal": {"x": 3, "y": 0}}}})
    assert env.shape == "no-walls" and env.walls == [] and env._scenario is not None
    o1, _ = env.reset(seed=5); o2, _ = make().reset(seed=5)            # without a scenario: the usual random scene
    assert env._scenario is None and all(np.array_equal(o1[k], o2[k]) for k in o1)


def test_trigger_waits_then_script_is_followed_exactly():
    env = make()
    spec = corridor(p={"name": "p", "x": 1.0, "y": 1.6, "behaviour": "scripted",
                       "path": [[0, 1.0, 1.6], [1.0, 1.0, 0.6], [2.0, 2.0, 0.6]], "start": {"time": 1.0}})
    env.reset(seed=0, options={"scenario": spec}); h = env.dynamic_humans[0]
    seen = []
    for k in range(40):
        _, _, te, tr, info = env.step(STILL); seen += info["scenario_events"]
        t = (k + 1) * env.TIMESTEP
        if t < 1.0 - 1e-9: assert (h.x, h.y) == (1.0, 1.6), (t, h.x, h.y)   # waiting: exactly still (no noise)
        else:
            u = t - 1.0; ex = (1.0, 1.6 - u) if u <= 1.0 else (1.0 + (u - 1.0), 0.6) if u <= 2.0 else (2.0, 0.6)
            assert abs(h.x - ex[0]) < 1e-6 and abs(h.y - ex[1]) < 1e-6, (t, h.x, h.y, ex)
    assert seen[0][1] == "p" and abs(seen[0][0] - 1.0) < 1e-9


def test_robot_within_delay_and_after_chain():
    env = make()
    spec = corridor(a={"name": "a", "x": 1.0, "y": 1.6, "behaviour": "scripted", "path": [[0, 1.0, 1.6], [1, 1.0, 2.2]],
                       "start": {"robot_within": {"distance": 2.0, "of": [1.0, 0.0]}, "delay": 0.5}},
                    b={"name": "b", "x": 2.0, "y": 2.2, "behaviour": "scripted", "path": [[0, 2.0, 2.2], [1, 2.6, 2.2]],
                       "start": {"after": {"person": "a", "delay": 0.3}}})
    env.reset(seed=0, options={"scenario": spec})
    ev = []
    for _ in range(60):
        _, _, te, tr, info = env.step(FORWARD); ev += info["scenario_events"]
        if te or tr: break
    starts = {n: t for t, n, what in ev if what.startswith("starts")}
    met = [t for t, n, what in ev if n == "a" and what.startswith("trigger")][0]
    assert abs(starts["a"] - met - 0.5) < 1e-9 and abs(starts["b"] - starts["a"] - 0.3) < 1e-9


def test_robot_crosses_line():
    env = make()
    spec = corridor(a={"name": "a", "x": 1.0, "y": 1.6, "behaviour": "scripted", "path": [[0, 1.0, 1.6], [1, 1.0, 2.2]],
                       "start": {"robot_crosses": [[-2.0, -1.0], [-2.0, 1.0]]}})
    env.reset(seed=0, options={"scenario": spec})
    for k in range(40):
        x0 = env.robot.x; _, _, _, _, info = env.step(FORWARD)
        if info["scenario_events"]:
            assert x0 < -2.0 <= env.robot.x; return
    raise AssertionError("never fired")


def test_goals_then_stop_and_loop():
    env = make(**{"human.human_policy": "orca"})
    spec = {"map": {"size": [9, 9], "boundary": False}, "robot": {"x": -4, "y": -4, "goal": {"x": -4, "y": 4}},
            "people": [{"name": "s", "x": 0, "y": 0, "behaviour": "goals", "policy": "orca", "speed": 1.0, "avoids_robot": False,
                        "goals": [[2, 0]], "then": "stop"},
                       {"name": "l", "x": 0, "y": 3, "behaviour": "goals", "policy": "orca", "speed": 1.0, "avoids_robot": False,
                        "goals": [[2, 3], [0, 3]], "then": "loop"}]}
    env.reset(seed=0, options={"scenario": spec}); s, l = env.dynamic_humans
    assert env.human_speed(s) == 1.0
    ev = []
    for _ in range(120):
        _, _, te, tr, info = env.step(STILL); ev += info["scenario_events"]
        if te or tr: break
    assert any(n == "s" and "stops" in w for _, n, w in ev) and s.policy == "wait"
    x = s.x; env.step(STILL); assert s.x == x                            # stopped people stay put
    assert l.policy == "orca" and l.goal_x in (0, 2)


def test_learned_people_and_determinism():
    env = make(**{"human.human_policy": "orca"})                         # the controller is added for the scenario
    spec = {"map": {"size": [9, 9], "boundary": False}, "robot": {"x": -4, "y": -4, "goal": {"x": -4, "y": 4}},
            "people": [{"name": "w", "x": 0, "y": -2, "behaviour": "goals", "policy": "learned", "speed": 0.9, "goals": [[0, 3]]},
                       {"name": "t", "x": 2, "y": -2, "behaviour": "goals", "policy": "learned", "goals": [[2, 3]], "start": {"time": 1.0}}]}
    runs = []
    for _ in range(2):
        env.reset(seed=7, options={"scenario": spec}); P = []
        for _ in range(40): env.step(STILL); P.append([(h.x, h.y) for h in env.dynamic_humans])
        runs.append(np.array(P))
    assert env.LEARNED_HUMANS is not None and np.array_equal(runs[0], runs[1])
    assert runs[0][-1, 0, 1] > -1.0 and runs[0][-1, 1, 1] > -2.0      # both walked towards their goals


def test_errors():
    env = make()
    bad = [({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "people": [{"x": 0.3, "y": 0, "behaviour": "static"}]}, "within"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "people": [{"x": 3, "y": 0, "goals": [[1, 1]], "colour": "red"}]}, "unknown"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "people": [{"x": 3 - 0.8 * i, "y": 3, "behaviour": "static"} for i in range(9)]}, "too many"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "furniture": [{"type": "table", "x": 3, "y": 3}]}, "too many"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "map": {"walls": [[-1, 0, 1, 0]]}}, "overlaps"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "people": [{"x": 3, "y": 0, "path": [[0, 2, 0]]}]}, "start where"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "people": [{"x": 3, "y": 0, "goals": [[1, 1]], "start": {"after": "nobody"}}]}, "unknown person"),
           ({"robot": {"x": 0, "y": 0, "goal": {"x": 1, "y": 1}}, "people": [{"x": 3, "y": 0, "goals": [[1, 1]], "start": {"soon": 1}}]}, "unknown field")]
    for spec, msg in bad:
        with pytest.raises(ScenarioError, match=msg): env.reset(seed=0, options={"scenario": spec})


def test_export_round_trip_and_set_scenario():
    env = make()
    env.reset(seed=21)
    d = export(env, name="r", seed=21)
    fd, path = tempfile.mkstemp(suffix=".yaml"); os.close(fd); dump(d, path)
    a = [(round(h.x, 3), round(h.y, 3)) for h in env.all_humans]
    env.set_scenario(path)
    for s in (0, 1):
        env.reset(seed=s)
        b = [(h.x, h.y) for h in env.all_humans]
        assert sorted(a) == sorted(b) and (env.robot.goal_x, env.robot.goal_y) == (round(d["robot"]["goal"]["x"], 3), round(d["robot"]["goal"]["y"], 3))
    d2 = export(env, name="r2"); assert d2["people"] == yaml.safe_load(open(path))["people"]   # a scenario exports as itself
    env.set_scenario(None); env.reset(seed=21)
    assert env._scenario is None


def test_collision_with_scripted_person_is_reported():
    env = make()
    spec = {"map": {"size": [9, 9], "boundary": False}, "robot": {"x": -3, "y": 0, "theta": 0, "goal": {"x": 3, "y": 0}},
            "people": [{"name": "rammer", "x": -1.5, "y": 0, "behaviour": "scripted", "path": [[0, -1.5, 0], [1.0, -3.0, 0]]}]}
    env.reset(seed=0, options={"scenario": spec})
    for _ in range(30):
        _, _, te, tr, info = env.step(STILL)
        if te: break
    assert info["COLLISION_HUMAN"] and info["scenario_collision"] == [("rammer", "scripted")]
