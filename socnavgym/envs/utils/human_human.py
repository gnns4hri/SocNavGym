import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.human import Human
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2
from typing import List
import random
import math
import time as clock


class Human_Human_Interaction:
    """
    Class for Human-Human Interactions
    """

    def __init__(self, x, y, type:str, numOfHumans:int, radius:float, human_width, MAX_HUMAN_SPEED, goal_radius=None, noise=0, can_disperse=True) -> None:
        # center of interaction
        self.x = x
        self.y = y
        self.name = "human-human-interaction"
        assert numOfHumans >= 2, "Need at least 2 humans to interact"       
        if type != "moving" and type != "stationary":
            raise AssertionError("type should be \"moving\" or \"stationary\"")
        
        # indicates the type of interaction, whether it is moving or stationary
        self.type = type

        # radius of the interaction space
        self.radius = radius

        self.humans:List[Human] = []
        speed = random.uniform(0.0, MAX_HUMAN_SPEED)

        self.goal_radius = goal_radius
        self.goal_x = None
        self.goal_y = None
        self.noise_variance = noise
        self.can_disperse = can_disperse

        for _ in range(numOfHumans):
            if self.type == "stationary":
                self.add_human(Human(speed=0, width=human_width, goal_radius=self.goal_radius, policy=random.choice(["orca", "sfm"])))
            else:
                self.add_human(Human(speed=speed, width=human_width, goal_radius=self.goal_radius, policy=random.choice(["orca", "sfm"])))
    
        # arranging all the humans around a circle
        self.arrange_humans()


    def set_goal(self, x, y):
        self.goal_x = x
        self.goal_y = y
        for human in self.humans:
            human.set_goal(x, y)

    def add_human(self, h:Human):
        """
        Adding humans to the human list
        """
        self.humans.append(h)
    
    def has_reached_goal(self, offset=None):
        reached = True
        for human in self.humans:
            if not human.has_reached_goal(offset):
                reached = False
                break
        return reached

    def arrange_humans(self):
        n = len(self.humans)
        theta = 0       
        increment = 2*(np.pi)/n

        if self.type == "moving":
            # theta chosen randomly between -pi to pi
            orientation = (np.random.random()-0.5) * np.pi * 2
    
        for i in range(n):
            h = self.humans[i]
            h.x = self.x + self.radius * np.cos(theta + (np.random.random()-0.5)*np.pi/7)
            h.y = self.y + self.radius * np.sin(theta + (np.random.random()-0.5)*np.pi/7)
            
            if self.type == "stationary":
                # humans would face the center as if talking to each other
                h.orientation = theta - np.pi + (np.random.random()-0.5)*np.pi/7

            elif self.type == "moving":
                # humans moving in the same direction, in a direction one direction
                h.orientation = orientation
            
            theta += increment
            if theta >= np.pi: theta -= 2*np.pi
        
        x_com = 0
        y_com = 0

        for h in self.humans:
            x_com += h.x
            y_com += h.y

        x_com /= n
        y_com /= n

        self.x = x_com
        self.y = y_com

    def collides(self, obj:Object):
        """
        To check for collision of any interacting human with another object
        """
        if obj.name == "human-human-interaction":
            for h1 in self.humans:
                for h2 in obj.humans:
                    if(h1.collides(h2)):
                        return True
            return False
        
        elif obj.name == "human-laptop-interaction":
            for h1 in self.humans:
                if(h1.collides(obj.human)):
                    return True
            return False

        for h in self.humans:
            if h.collides(obj): return True
        return False

    def update(self, time, velocity=None, max_rotation_speed=math.pi/2.):
        if self.type == "stationary":
            pass

        elif self.type == "moving":            
            if velocity is None: raise AssertionError("velocity for update is None")
            n = len(self.humans)
            speeds = []
            rotating = False
            vel_human = (velocity[0], velocity[1])
            for human in self.humans:
                noise_x = np.random.normal(0, self.noise_variance)
                noise_y = np.random.normal(0, self.noise_variance)
                human_vel = (vel_human[0]+noise_x, vel_human[1]+noise_y)
                speed = np.linalg.norm(human_vel)
                new_orientation = atan2(human_vel[1], human_vel[0])
                set_o = human.set_new_orientation_with_limits(new_orientation, max_rotation_speed, time)
                if not set_o:
                    rotating = True
                speeds.append(speed)

            for human, s in zip(self.humans, speeds):
                if rotating: s = 0
                human.speed = s
                human.update(time)

            x_com = 0
            y_com = 0

            for h in self.humans:
                x_com += h.x
                y_com += h.y

            x_com /= n
            y_com /= n

            self.x = x_com
            self.y = y_com

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        
        for h in self.humans:
            h.draw(img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y)
        
        points = []
        for h in self.humans:
            points.append(
                [
                    w2px(h.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                    w2py(h.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
                ]
            )
        points = np.array(points).reshape((-1,1,2))
        cv2.polylines(img, [np.int32(points)], True, (0, 0, 255), 1)
    