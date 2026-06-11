import sys
import os
import time
import numpy as np
import cv2
import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
pygame.init()

import socnavgym

import gymnasium as gym

GRAPH_WIDTH=800



env_orig = gym.make("SocNavGym-v2", config="./test_env.yaml")
env = env_orig.env.env
obs, _ = env.reset()
env.world_image = env.render_without_showing("human", draw_human_gaze=False, draw_human_goal=False)
plt_array = np.ones((env.world_image.shape[0], GRAPH_WIDTH, 3), dtype=np.uint8)*100


plt.ion()
dpi = 100
fig, axs = plt.subplots(1, 1, figsize=(GRAPH_WIDTH/dpi, env.world_image.shape[0]/dpi), dpi=dpi)
rotation = transforms.Affine2D().rotate_deg(90)
axs.set_xlim(-7, 7)
axs.set_ylim(-7, 7)
axs.set_aspect('equal')
axs.grid(True)
axs.transData = rotation + axs.transData
plt.tight_layout()




vx = 0.3
vy = 0
va = 0

done = False
episode = -1
while True:
    if done:
        episode += 1
        print(f"Generating next expisode...")
        obs, _ = env.reset()
        done = False
        print(f"Episode {episode} started.")
    else:
        obs, reward, terminated, truncated, info = env.step([vx, vy, va])
        done = terminated or truncated
    
    # Clear the previous frame
    axs.clear()
    axs.set_xlim(-7, 7)
    axs.set_ylim(-7, 7)
    axs.grid(True)

    # Draw axis
    axs.plot([-10, 10], [  0,  0], 'k-', linewidth=1)
    axs.plot([  0,  0], [-10, 10], 'k-', linewidth=1)


    if info["relative_frame"] == "ROBOT_FR":
        # Draw goal
        robot = obs["robot"]
        robot_x = robot[0]
        robot_y = robot[1]
        robot_x2 = robot_x + robot[2]*robot[4]
        robot_y2 = robot_y + robot[2]*robot[3]
        axs.plot([robot_x, robot_x2], [robot_y, robot_y2], 'g-', linewidth=4)
        axs.add_patch(Circle((robot_x, robot_y), robot[2], fill=False, color='green', linewidth=2))
        # Draw robot
        axs.add_patch(Circle((0, 0), robot[6], fill=False, color='red', linewidth=2))
        axs.plot([0, robot[6]*1.5], [0, 0], 'r-', linewidth=4)
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
        axs.plot([left_x, right_x], [left_y, right_y], 'b-', linewidth=2)


    # Draw humans
    humans = obs["humans"].reshape(-1, 8)
    humans = humans[~np.all(humans == 0, axis=1)]
    for human in humans:
        # Calculate human's coordinates
        h_x = human[0]
        h_y = human[1]
        h_x2 = h_x + 0.35*human[3]
        h_y2 = h_y + 0.35*human[2]
        axs.plot([h_x, h_x2], [h_y, h_y2], '-', color="orange", linewidth=4)
        # Plot the human as a circle and a segment
        axs.add_patch(Ellipse((h_x, h_y), width=0.3, height=0.7, angle=np.atan2(human[2], human[3])*180/np.pi, fill=False, color='orange', linewidth=2))

    #print(rewards)
    # Update the plot
    # plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()



