import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2

class Robot(Object):
    def __init__(self, id=None, x=None, y=None, theta=None, radius=None, goal_x=None, goal_y=None, type="diff-drive") -> None:
        super().__init__(id, "robot")
        self.is_static = False
        self.radius = None  # radius of the robot
        self.goal_x = None  # x-coordinate of the goal
        self.goal_y = None  # y-coordinate of the goal
        self.type = None  # Type of the robot i.e holonomic or diff-drive

        # variables used for robot velocity
        self.vel_x = 0.0  # velocity in the direction that the robot is facing
        self.vel_y = 0.0  # velocity in the direction perpendicular to the robot's facing direction
        self.vel_a = 0.0  # angular velocity

        assert(type == "diff-drive" or type == "holonomic")
        self.set(id, x, y, theta, radius, goal_x, goal_y, type)

    def set(self, id, x, y, theta, radius, goal_x, goal_y, type):
        super().set(id, x, y, theta)
        self.radius = radius
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.type = type

    def update(self, time):
        """For updating the coordinates of the robot

        Args:
            time (float): Time passed.
        """        

        if self.type == "diff-drive":
            assert self.vel_y == 0.0,  "Cannot move in lateral direction for a differential drive robot"
        
        self.orientation += self.vel_a * time  # updating the robot orientation
        # restricting the robot's orientation value to be between [-np.pi, +np.pi]
        if self.orientation > 2*np.pi:
            self.orientation -= int(self.orientation/(2*np.pi))*(2*np.pi)
        if self.orientation < -2*np.pi:
            self.orientation += int(abs(self.orientation)/(2*np.pi))*(2*np.pi)

        if self.orientation > np.pi: self.orientation -= 2*np.pi
        elif self.orientation < -np.pi: self.orientation += 2*np.pi

        # updating the linear component
        self.x += self.vel_x * np.cos(self.orientation) * time
        self.y += self.vel_x * np.sin(self.orientation) * time

        # updating the perpendicular component
        self.x += self.vel_y * np.cos(np.pi/2 + self.orientation) * time
        self.y += self.vel_y * np.sin(np.pi/2 + self.orientation) * time
        
    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        black = (0,0,0) 
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        )  # calculating no. of pixels corresponding to the radius
       
        cv2.circle(
            img,
            (
                w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y),
            ),
            radius,
            black,
            -1,
        )  # drawing a black circle for the robot
        
        
        left = (
            w2px(self.x + self.radius*0.35*np.cos(self.orientation + np.pi/2), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y + self.radius*0.35*np.sin(self.orientation + np.pi/2), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )

        right = (
            w2px(self.x + self.radius*0.35*np.cos(self.orientation - np.pi/2), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y + self.radius*0.35*np.sin(self.orientation - np.pi/2), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )

        front = (
            w2px(self.x + self.radius*0.35*np.cos(self.orientation), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y + self.radius*0.35*np.sin(self.orientation), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )

        center = (
            w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )

        # drawing lines to get sense of the orientation of the robot.
        cv2.line(img, left, right, (27, 194, 169), 2)
        cv2.line(img, center, front, (27, 194, 169), 2)

    def draw_range(self, img, range, fov, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        center = (w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X), w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y))
        radius = w2px(self.x + range, PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        )  # calculating no. of pixels corresponding to the radius
       
        axesLength = (radius, radius)
        fov = fov * 180 / np.pi
        orientation = self.orientation * 180 / np.pi

        cv2.ellipse(
            img,
            center,
            axesLength,
            angle=-orientation,
            startAngle=(-fov/2),
            endAngle=(fov/2),
            color=(173, 137, 250), 
            thickness=-1
        )


