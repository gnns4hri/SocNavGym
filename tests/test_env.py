import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "SocNavGym-main"))

import socnavgym
import inspect
from socnavgym.envs.utils.human import Human
print("socnavgym imported from:", socnavgym.__file__)
print("Human class loaded from:", inspect.getfile(Human))

import gymnasium as gym


plt.ion()
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('SocNavGym Wall Segments')


env = gym.make("SocNavGym-v2", config=os.path.join(current_dir, "test_env.yaml"))
obs, _ = env.reset()


for i in range(10000):
    # Step the environment
    obs, reward, terminated, truncated, info = env.step([0.1, 0, 0.01])
    env.render()
    if terminated or truncated:
        for i in range(100):
            env.render()
            time.sleep(0.1)
        obs, _ = env.reset()

    # Clear the previous frame
    ax.clear()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True)
    ax.set_title('SocNavGym Wall Segments')

    # Draw axis
    ax.plot([-20, 20], [  0,  0], 'k-', linewidth=1)
    ax.plot([  0,  0], [-20, 20], 'k-', linewidth=1)


    if info["relative_frame"] == "GOAL_FR":
        # Draw robot
        robot = obs["robot"]
        robot_x = robot[0]
        robot_y = robot[1]
        robot_x2 = robot_x + 0.2*robot[4]
        robot_y2 = robot_y + 0.2*robot[3]
        ax.plot([robot_x, robot_x2], [robot_y, robot_y2], 'r-', linewidth=4)
        ax.add_patch(Circle((robot_x, robot_y), robot[2], fill=False, color='red', linewidth=2))
        # Draw goal
        ax.add_patch(Circle((0, 0), robot[2], fill=False, color='red', linewidth=2))
    if info["relative_frame"] == "ROBOT_FR":
        # Draw goal
        robot = obs["robot"]
        robot_x = robot[0]
        robot_y = robot[1]
        robot_x2 = robot_x + 0.2*robot[4]
        robot_y2 = robot_y + 0.2*robot[3]
        ax.plot([robot_x, robot_x2], [robot_y, robot_y2], 'r-', linewidth=4)
        ax.add_patch(Circle((robot_x, robot_y), robot[2], fill=False, color='red', linewidth=2))
        # Draw robot
        ax.add_patch(Circle((0, 0), robot[2], fill=False, color='red', linewidth=2))
    else:
        print(info)
        sys.exit(1)


    # Draw walls
    walls = obs["walls"].reshape(-1, 8)
    walls = walls[~np.all(walls == 0, axis=1)]
    for wall in walls:
        # Calculate wall endpoints
        half_len = wall[4]
        left_x = wall[0] - half_len * wall[3]
        left_y = wall[1] - half_len * wall[2]
        right_x = wall[0] + half_len * wall[3]
        right_y = wall[1] + half_len * wall[2]
        # Plot the wall as a line segment
        ax.plot([left_x, right_x], [left_y, right_y], 'b-', linewidth=2)


    # Draw humans
    humans = obs["humans"].reshape(-1, 8)
    humans = humans[~np.all(humans == 0, axis=1)]
    for human in humans:
        # Calculate wall endpoints
        h_x = human[0]
        h_y = human[1]
        # Plot the wall as a line segment
        ax.add_patch(Circle((h_x, h_y), 0.2, fill=False, color='orange', linewidth=2))
        ax.plot([left_x, right_x], [left_y, right_y], 'b-', linewidth=2)


    # Update the plot
    plt.draw()
    plt.pause(0.5)  # Non-blocking pause

    time.sleep(0.5)



