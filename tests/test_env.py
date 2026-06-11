import sys
import os
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
pygame.init()

import socnavgym

import gymnasium as gym

GRAPH_WIDTH=800


done = False
MAX_EPISODES = 25
MAX_PATIENCE = 100
patience = MAX_PATIENCE
env_orig = gym.make("SocNavGym-v2", config="./test_env.yaml")
env = env_orig.env.env
obs, _ = env.reset()
env.world_image = env.render_without_showing("human", draw_human_gaze=False, draw_human_goal=False)
plt_array = np.ones((env.world_image.shape[0], GRAPH_WIDTH, 3), dtype=np.uint8)*100

g_surface = None
def get_surface(plt_array, world_image):
    global g_surface
    w_shape = world_image.shape
    data_array = np.concatenate((plt_array, cv2.cvtColor(world_image, cv2.COLOR_BGR2RGB)), axis=1)
    g_surface = pygame.surfarray.make_surface(np.transpose(data_array, (1,0,2)))
    surface_resized = pygame.transform.smoothscale(g_surface, data_array.shape[:-1][::-1])
    return surface_resized


def get_joy_values():
    if joystick_count > 0:
        vx = -joystick.get_axis(1)
        vy = -joystick.get_axis(0)*0.5
        va = -joystick.get_axis(2)*0.5
        return np.array([vx, vy, va])
    else:
        print("no joy")
        return np.array([0, 0, 0])


if not env.window_initialised:
    pygame.init()
    env.screen = pygame.display.set_mode((int(env.RESOLUTION_X+GRAPH_WIDTH), int(env.RESOLUTION_Y)))
    pygame.display.set_caption("SocNavGym v0.2")
    env.window_initialised = True

env.world_image = env.render_without_showing("human", draw_human_gaze=False, draw_human_goal=False)
surface = get_surface(plt_array, env.world_image)
env.screen.blit(surface, (0,0))
for event in pygame.event.get():
    pass
pygame.display.update()


plt.ion()
dpi = 100
fig, axs = plt.subplots(2, 1, figsize=(GRAPH_WIDTH/dpi, env.world_image.shape[0]/dpi), dpi=dpi)
rotation = transforms.Affine2D().rotate_deg(90)
axs[0].set_xlim(-7, 7)
axs[0].set_ylim(-7, 7)
axs[0].set_aspect('equal')
axs[0].grid(True)
axs[0].transData = rotation + axs[0].transData
axs[1].set_ylim(-1, 1)
plt.tight_layout()




pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected!")
    print("No joystick detected!")
    print("No joystick detected!")
    print("No joystick detected!")
    joystick = None
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Initialized joystick: {joystick.get_name()}")
    stale_joystick = get_joy_values()


def dummy_control(obs):
    def force(vector):
        dist = np.linalg.norm(vector)
        angle = np.atan2(vector[1], vector[0])
        n = vector/dist
        return n, dist, angle

    # Start
    l_speed = np.array([0.,0.])
    a_speed = 0

    # People
    for person in obs["humans"].reshape(-1, 8):
        person_v, person_d, person_a = force(person[0:2])
        if person_d < 2:
            mult = max(0, 2.-person_d)
            l_speed -= 2.0 * mult * person_v

    # Goal
    goal = obs["robot"]
    goal_v, goal_d, goal_a = force(goal[:2])
    if goal_d > 2.0:
        l_speed += goal_v
        va = goal_a
    else:
        l_speed += 0.5 * goal_v
        va = 0.25 * np.atan2(goal[3], goal[4])


    # Set action
    vx = l_speed[0]
    vy = l_speed[1]
    return vx, vy, va


rewards = []
pause = False
episode = 0
while True:
    print(".", end="")
    sys.stdout.flush()
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
    if joystick_count > 0:
        joystic_data = get_joy_values() - stale_joystick
        vx = joystic_data[0]
        vy = joystic_data[1]
        va = joystic_data[2]
    else:
        vx, vy, va = dummy_control(obs)
    if done:
        patience -= 1
        if patience < 0:
            print(f"Episode {episode} finished.")
            if episode < MAX_EPISODES:
                print(f"Generating next expisode...")
                episode += 1
                obs, _ = env.reset()
                done = False
                patience = MAX_PATIENCE
                pause = False
                rewards = []
                print(f"Episode {episode} started.")
            else:
                print(f"{MAX_EPISODES} episodes should be enough")
                sys.exit(0)
        else:
            pass
            # print(np.sum(np.array(rewards)))
    elif pause is False:
        obs, reward, terminated, truncated, info = env.step([vx, vy, va])
        done = terminated or truncated
        # print(f"{obs['robot'].shape}")
        # print(f"{obs['humans'].shape}")
        rewards.append(float(reward))
    else:
        print("paused")

    # Clear the previous frame
    axs[0].clear()
    axs[0].set_xlim(-7, 7)
    axs[0].set_ylim(-7, 7)
    axs[0].grid(True)

    # Draw axis
    axs[0].plot([-10, 10], [  0,  0], 'k-', linewidth=1)
    axs[0].plot([  0,  0], [-10, 10], 'k-', linewidth=1)


    if info["relative_frame"] == "ROBOT_FR":
        # Draw goal
        robot = obs["robot"]
        robot_x = robot[0]
        robot_y = robot[1]
        robot_x2 = robot_x + robot[2]*robot[4]
        robot_y2 = robot_y + robot[2]*robot[3]
        axs[0].plot([robot_x, robot_x2], [robot_y, robot_y2], 'g-', linewidth=4)
        axs[0].add_patch(Circle((robot_x, robot_y), robot[2], fill=False, color='green', linewidth=2))
        # Draw robot
        axs[0].add_patch(Circle((0, 0), robot[6], fill=False, color='red', linewidth=2))
        axs[0].plot([0, robot[6]*1.5], [0, 0], 'r-', linewidth=4)
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
        axs[0].plot([left_x, right_x], [left_y, right_y], 'b-', linewidth=2)


    # Draw humans
    humans = obs["humans"].reshape(-1, 8)
    humans = humans[~np.all(humans == 0, axis=1)]
    for human in humans:
        # Calculate human's coordinates
        h_x = human[0]
        h_y = human[1]
        h_x2 = h_x + 0.35*human[3]
        h_y2 = h_y + 0.35*human[2]
        axs[0].plot([h_x, h_x2], [h_y, h_y2], '-', color="orange", linewidth=4)
        # Plot the human as a circle and a segment
        axs[0].add_patch(Ellipse((h_x, h_y), width=0.3, height=0.7, angle=np.atan2(human[2], human[3])*180/np.pi, fill=False, color='orange', linewidth=2))
        # axs[0].plot([left_x, right_x], [left_y, right_y], 'b-', linewidth=2)

    axs[1].clear()
    axs[1].set_ylim(-1.1, 1.1)
    if len(rewards) > 0:
        x_plot_values = np.array([x for x in range(len(rewards))])
        for vv in [0.002, 0.25, 0.5, 0.75, 1]:
            for s in [-1, 1]:
                colour = (vv+0.75)/2
                axs[1].plot([x_plot_values[0], x_plot_values[-1]], [vv*s, vv*s], "-", color=str(colour))
        y_plot_values = np.array(rewards)
        axs[1].plot(x_plot_values, y_plot_values, "b-")

    # Update the plot
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt_array = np.asarray(fig.canvas.buffer_rgba())[:,:,:-1]

    env.world_image = env.render_without_showing("human", draw_human_gaze=False, draw_human_goal=False)

    surface = get_surface(plt_array, env.world_image)
    env.screen.blit(surface, (0,0))
    pygame.display.update()

