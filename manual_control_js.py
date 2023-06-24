import torch
import time
import gym
import numpy as np
import socnavgym
import os
import pygame
import numpy as np 
import sys
import argparse
import time
import pickle
import json
from glob import glob
from pathlib import Path

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide2'
from PySide2 import QtWidgets
import pyqtgraph as pg

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num_episodes", required=True, help="number of episodes")
ap.add_argument("-j", "--joystick_id", required=False, default=0, help="Joystick identifier")
ap.add_argument("-c", "--config", required=False, default="./environment_configs/exp1_no_sngnn.yaml", help="Environment config file")
ap.add_argument("-r", "--record", required=False, default=False, help="Whether you want to record the observations, and actions or not")
ap.add_argument("-s", "--start", required=False, default=0, help="starting episode number")
args = vars(ap.parse_args())
episodes = int(args["num_episodes"])

pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
joystick = pygame.joystick.Joystick(int(args["joystick_id"]))
joystick.init()
axes = joystick.get_numaxes()


app = QtWidgets.QApplication(sys.argv)
plt = pg.plot()
# plt.setLogMode(y=True)
pg.setConfigOption('foreground', (0, 0, 0))
my_brush = pg.mkBrush('k', width=3)
default_brush = plt.foregroundBrush()
plt.setWindowTitle('rewards')
plt.setForegroundBrush(my_brush)
plt.addLegend()
plt.setForegroundBrush(default_brush)
plt.show()
plt.setBackground((200, 200, 200))
time.sleep(2)

try:
    with open('joystick_calibration.pickle', 'rb') as f:
        centre, values, min_values, max_values = pickle.load(f)
except:
    centre = {}
    values = {}
    min_values = {}
    max_values = {}
    for axis in range(joystick.get_numaxes()):
        values[axis] = 0.
        centre[axis] = 0.
        min_values[axis] = 0.
        max_values[axis] = 0.
    T = 3.
    print(f'Leave the controller neutral for {T} seconds')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            centre[axis] = joystick.get_axis(axis)
        time.sleep(0.05)
    T = 5.
    print(f'Move the joystick around for {T} seconds trying to reach the max and min values for the axes')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            value = joystick.get_axis(axis)-centre[axis]
            if value > max_values[axis]:
                max_values[axis] = value
            if value < min_values[axis]:
                min_values[axis] = value
        time.sleep(0.05)
    with open('joystick_calibration.pickle', 'wb') as f:
        pickle.dump([centre, values, min_values, max_values], f)
print(min_values)
print(max_values)


start = int(args["start"])
if os.path.isdir("./episode_recordings/"):
    for path in Path("./episode_recordings/").rglob('*.json'):
        path = str(path)
        start = max(int(path.split("/")[-1].split(".")[0]), start)
env = gym.make("SocNavGym-v1", config=args["config"])

total_sums = []
import time

for episode in range(episodes):
    env.reset()
    done = False

    step = -1
    prev_sum = 0
    x = []
    sums = []
    rewards = []
    sngnn = []
    total_reward = 0
    d = {}
    while not done:
        step += 1
        plt.clear()
        pygame.event.pump()
        for i in range(joystick_count):
            print(f'{i} ', end='')
            joystick = pygame.joystick.Joystick(int(args["joystick_id"]))
            axes = joystick.get_numaxes()
            for axis in range(axes):
                values[axis] = joystick.get_axis(axis)-centre[axis]

        # values[1] = max(-values[1], 0.)
        # forward_speed = (values[1]-0.5)*2/max_values[1]
        vel_x = -values[1]/max_values[1]
        vel_y = -values[0]/max_values[0]
        vel_a = -values[4]/max_values[4]
        if env.robot.type == "diff-drive": vel_y = 0


        obs, rew, terminated, truncated, info = env.step([vel_x, vel_y, vel_a])
        obs["action"] = np.array([vel_x, vel_y, vel_a], dtype=np.float32)
        obs["reward"] = np.array([rew], dtype=np.float32)
        for key in obs.keys():
            obs[key] = obs[key].tolist()
        d[step] = obs
        done = terminated or truncated
        total_reward += rew
        potential_field = info['DISCOMFORT_DSRNN']
        distance_reward = info['distance_reward']
        sngnn_reward = info['sngnn_reward']

        # print(f"potential_field: {potential_field} ", f"distance_reward: {distance_reward} " f"total reward: {rew}")
        print(f"sngnn_reward: {sngnn_reward} ", f"distance_reward: {distance_reward} " f"total reward: {rew}")
        
        x.append(step)
        rewards.append(rew)
        sngnn.append(info['sngnn_reward'])
        new_sum = prev_sum + rew
        sums.append(new_sum)
        prev_sum = new_sum
        app.processEvents()
        env.render()

        if done:
            print(f"Total reward : {total_reward}")
            env.reset()
            if args["record"]:
                if not os.path.isdir("./episode_recordings/"):
                    os.makedirs("./episode_recordings/")
                with open("./episode_recordings/" + str(episode+1+start).zfill(8) + ".json", "w") as f:
                    json.dump(d, f, indent=4)
            
            # for _ in range(10):
            #     x.append(step)
            #     rewards.append(rew)
            #     sngnn.append(info['sngnn_reward'])
            #     sums.append(sums[-1])
        plt.plot(x, rewards, pen=pg.mkPen((255, 0, 0), width=2), name='rewards')
        plt.plot(x, sngnn, pen=pg.mkPen((0, 230, 0), width=2), name='sngnn')
        plt.plot(x, sums, pen=pg.mkPen((0, 0, 150), width=2), name='SUM')

        # if done:
        #     for _ in range(10000):
        #         app.processEvents()
        #         time.sleep(0.01)
        app.processEvents()
        # time.sleep(1)
    
