import sys
import os
import time
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse



plt.ion()
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('SocNavGym Wall Segments')
plt.tight_layout()

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "SocNavGym-main"))
import pygame
pygame.init()

import socnavgym
import inspect
from socnavgym.envs.utils.human import Human
print("socnavgym imported from:", socnavgym.__file__)
print("Human class loaded from:", inspect.getfile(Human))

import gymnasium as gym





env = gym.make("SocNavGym-v2", config=os.path.join(current_dir, "test_env.yaml"))
obs, _ = env.reset()


pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected!")
    sys.exit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Initialized joystick: {joystick.get_name()}")


def get_joy_values():
    if joystick_count > 0:
        vx = -joystick.get_axis(1)
        vy = -joystick.get_axis(0)*0.1
        va = -joystick.get_axis(2)*0.1
        return np.array([vx, vy, va])
    else:
        print("no joy")
        return np.array([0, 0, 0])


stale_joystick = get_joy_values()
print(f"{stale_joystick=}")


pause = False
i = 0
while i < 10_000:
    i += 1

    pygame.event.pump()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("escape was pressed, exiting")
                pygame.quit()
                sys.exit(0)
            if event.key == pygame.K_p:
                pause = not pause
                if pause is True:
                    print("paused")
                else:
                    print("unpaused")
        elif event.type == pygame.JOYAXISMOTION:
            pass
            # print(f"Axis {event.axis}: {event.value:.2f}")
    if joystick_count > 0:
        joystic_data = get_joy_values() - stale_joystick
        # print(f"{joystic_data=} === {get_joy_values()=} --- {stale_joystick=}")
        vx = joystic_data[0]
        vy = joystic_data[1]
        va = joystic_data[2]
    else:
        vx = 0.1
        vy = 0
        va = 0

    # Step the environment with joystick input
    if pause is False:
        obs, reward, terminated, truncated, info = env.step([vx, vy, va])


    env.render()
    if terminated or truncated:
        for i in range(10):
            env.render()
            time.sleep(0.1)
        obs, _ = env.reset()

    # Clear the previous frame
    ax.clear()
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.grid(True)
    ax.set_title('SocNavGym Wall Segments')

    # Draw axis
    ax.plot([-11, 11], [  0,  0], 'k-', linewidth=1)
    ax.plot([  0,  0], [-11, 11], 'k-', linewidth=1)


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
        ax.add_patch(Circle((0, 0), robot[2], fill=False, color='green', linewidth=2))
    if info["relative_frame"] == "ROBOT_FR":
        # Draw goal
        robot = obs["robot"]
        robot_x = robot[0]
        robot_y = robot[1]
        robot_x2 = robot_x + robot[2]*robot[4]
        robot_y2 = robot_y + robot[2]*robot[3]
        ax.plot([robot_x, robot_x2], [robot_y, robot_y2], 'g-', linewidth=4)
        ax.add_patch(Circle((robot_x, robot_y), robot[2], fill=False, color='green', linewidth=2))
        # Draw robot
        ax.add_patch(Circle((0, 0), robot[6], fill=False, color='red', linewidth=2))
        ax.plot([0, robot[6]*1.5], [0, 0], 'r-', linewidth=4)
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
        # Calculate human's coordinates
        h_x = human[0]
        h_y = human[1]
        h_x2 = h_x + 0.35*human[3]
        h_y2 = h_y + 0.35*human[2]
        ax.plot([h_x, h_x2], [h_y, h_y2], '-', color="orange", linewidth=4)
        # Plot the human as a circle and a segment
        ax.add_patch(Ellipse((h_x, h_y), width=0.3, height=0.7, angle=np.atan2(human[2], human[3])*180/np.pi, fill=False, color='orange', linewidth=2))
        ax.plot([left_x, right_x], [left_y, right_y], 'b-', linewidth=2)


    # Update the plot
    plt.draw()
    fig.canvas.flush_events()  # <-- This is critical



