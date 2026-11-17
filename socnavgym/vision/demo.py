"""Runs an environment that has a `vision:` section and shows (or saves) what the robot's sensors and the 3D view see.

    python -m socnavgym.vision.demo                          the example scene, windows (needs a screen)
    python -m socnavgym.vision.demo --no-window --save out   no screen: 100 steps, pictures saved in out/

Windows: the simulator's 2D window is always there; its "Windows" menu shows and hides the rendered ones (the 3D view and the
         sensors' readings; all open at the start). A window's close button hides it too; closing the 2D window quits.
         Without the renderer (see README.md) only the 2D window appears and the simulation runs as usual.
3D view: starts from above, framed on the whole scene. Left drag turns, right drag (or shift + left drag) moves, wheel zooms,
         arrows / + / - too, F follows the robot, R goes back to the starting view. Its window can be resized.
Robot:   W / S forward / back, A / D turn, Z / C sideways (holonomic robots); J / L and I / K move the first two joints
         (the default robot's head: pan and tilt); without keys the robot drives itself towards its goal (--drive).
Other:   space pauses the simulation (the view can still be moved), N starts a new episode, Q or Esc quits.
"""
import os, argparse, time
import numpy as np


def main():
    here = os.path.dirname(os.path.abspath(__file__)); ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "..", "..", "environment_configs", "vision_demo.yaml"))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=None, help="stop after this many steps (default: 100 without windows, endless with)")
    ap.add_argument("--drive", choices=["goal", "still"], default="goal", help="what the robot does when no key is held")
    ap.add_argument("--no-window", action="store_true", help="no windows (computers without a screen)"); ap.add_argument("--save", default=None, help="folder for the pictures of every step")
    a = ap.parse_args()
    import gymnasium as gym, socnavgym  # noqa: F401
    env = gym.make("SocNavGym-v2", config=a.config).unwrapped; vis = env.vision       # vis is None when the renderer is not installed and nothing of the `vision:` section works without it
    if vis is not None and not a.no_window:
        for k in vis.shown: vis.shown[k] = True                  # every window open at the start; the menu on the 2D window hides them
    if a.save: os.makedirs(a.save, exist_ok=True); import cv2
    obs, _ = env.reset(seed=a.seed); seed = a.seed; n = 0; paused = False; steps = a.steps if a.steps is not None else (100 if a.no_window else None)
    nj = vis.nj if vis is not None else 0; u = np.zeros(nj, np.float32); was = set()
    if nj: u = 2 * (vis.q0 - vis.lower) / np.maximum(vis.upper - vis.lower, 1e-9) - 1             # the joints' targets, in the action's units
    if vis is None: print("no sensors and no 3D view (see the warning above): the 2D simulation alone")
    else: print(f"sensors: {list(vis.sensors)}; joints: {[j.name for j in vis.joints]}; renderer: {vis.client.info['device'] if vis.client else 'none'}")
    while steps is None or n < steps:
        act = np.zeros(3, np.float32)
        if a.drive == "goal":                                    # towards the goal (obs["robot"][:2]: the goal in the robot's frame), slowly
            g = np.asarray(obs["robot"][:2], float); d = float(np.hypot(*g)); ang = float(np.arctan2(g[1], g[0]))
            if d > 0.3: act[:] = (0.5 * np.cos(ang), 0.5 * np.sin(ang) if env.robot.type == "holonomic" else 0.0, np.clip(ang, -0.5, 0.5))
        if not a.no_window:
            import pygame as pg
            if pg.get_init() and pg.event.peek(pg.QUIT): break   # the 2D window was closed (without vision nobody else looks at the events)
            env.render(); k = pg.key.get_pressed()               # the 2D window with its menu, and the windows switched on in it
            if (vis is not None and vis.windows is not None and not vis.windows.alive) or k[pg.K_q] or k[pg.K_ESCAPE]: break
            now = {c for c in (pg.K_SPACE, pg.K_n) if k[c]}; hit = now - was; was = now       # keys that act once per press
            if pg.K_SPACE in hit: paused = not paused
            if pg.K_n in hit: seed += 1; obs, _ = env.reset(seed=seed); continue
            drive = np.array([int(k[pg.K_w]) - int(k[pg.K_s]), int(k[pg.K_z]) - int(k[pg.K_c]), int(k[pg.K_a]) - int(k[pg.K_d])], np.float32)
            if drive.any(): act = drive * (0.7, 0.7, 0.5)
            if nj >= 1: u[0] = np.clip(u[0] + 0.06 * (int(k[pg.K_j]) - int(k[pg.K_l])), -1, 1)
            if nj >= 2: u[1] = np.clip(u[1] + 0.06 * (int(k[pg.K_k]) - int(k[pg.K_i])), -1, 1)
            if paused: time.sleep(0.02); continue
        obs, rew, term, trunc, info = env.step(np.concatenate([act, u])); n += 1
        if a.save and vis is not None:
            for name, im in vis.images().items(): cv2.imwrite(os.path.join(a.save, f"{n:04d}_{name.replace(' ', '_')}.png"), im[..., ::-1])
            if vis.viewer is not None: cv2.imwrite(os.path.join(a.save, f"{n:04d}_view.png"), vis.view()[..., ::-1])
        if term or trunc: seed += 1; obs, _ = env.reset(seed=seed)
    env.close()


if __name__ == "__main__": main()
