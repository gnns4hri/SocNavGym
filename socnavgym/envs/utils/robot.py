import sys
import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2

class Robot(Object):
    def __init__(self, id=None, x=None, y=None, theta=None, radius=None, goal_x=None, goal_y=None, goal_a=None, type="diff-drive") -> None:
        super().__init__(id, "robot")
        self.is_static = False
        self.radius = radius  # radius of the robot
        self.goal_x = goal_x  # x-coordinate of the goal
        self.goal_y = goal_y  # y-coordinate of the goal
        self.goal_a = goal_a
        self.type = type  # Type of the robot i.e holonomic or diff-drive

        # variables used for robot velocity
        self.vel_x = 0.0  # velocity in the direction that the robot is facing
        self.vel_y = 0.0  # velocity in the direction perpendicular to the robot's facing direction
        self.vel_a = 0.0  # angular velocity

        assert(type == "diff-drive" or type == "holonomic")
        self.set(id, x, y, theta, radius, goal_x, goal_y, goal_a, type)

    def set(self, id, x, y, theta, radius, goal_x, goal_y, goal_a, type):
        super().set(id, x, y, theta)
        self.radius = radius
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_a = goal_a
        self.type = type

    def update(self, time):
        """For updating the coordinates of the robot

        Args:
            time (float): Time passed.
        """        

        self.debug_data = {}
        self.debug_data["init_vel_y"] = self.vel_y

        if self.type == "diff-drive":
            self.vel_y = 0.0
        self.debug_data["after_vel_y"] = self.vel_y
        self.debug_data["vel_a"] = self.vel_a
        self.debug_data["time"] = time

        self.debug_data["orientation_init"] = self.orientation
        self.orientation += self.vel_a * time  # updating the robot orientation
        self.debug_data["orientation_with_vel_a*time"] = self.orientation

        # # restricting the robot's orientation value to be between [-np.pi, +np.pi]
        # if self.orientation > 2*np.pi:
        #     self.orientation -= int(self.orientation/(2*np.pi))*(2*np.pi)
        # if self.orientation < -2*np.pi:
        #     self.orientation += int(abs(self.orientation)/(2*np.pi))*(2*np.pi)

        self.debug_data["orientation_before_normalising"] = self.orientation

        while self.orientation > np.pi:
            self.orientation -= 2*np.pi
        while self.orientation < -np.pi:
            self.orientation += 2*np.pi

        self.debug_data["orientation_after_normalising"] = self.orientation

        # updating the linear component
        self.x += self.vel_x * np.cos(self.orientation) * time
        self.y += self.vel_x * np.sin(self.orientation) * time
        self.debug_data["x 2"] = self.x
        self.debug_data["y 2"] = self.y


        # updating the perpendicular component
        self.x += self.vel_y * np.cos(np.pi/2 + self.orientation) * time
        self.y += self.vel_y * np.sin(np.pi/2 + self.orientation) * time
        self.debug_data["x 3"] = self.x
        self.debug_data["y 3"] = self.y
        
    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        black = (0,0,0) 
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        try:
            # calculating no. of pixels corresponding to the radius
            radius = w2px(self.x + self.radius, PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X)
        except ValueError:
            print(f"{self.x=} {self.radius=} {PIXEL_TO_WORLD_X=} {MAP_SIZE_X=}")
            print(self.debug_data)
            sys.exit(-1)
       
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
            w2px(self.x + self.radius*0.3*np.cos(self.orientation + np.pi/2), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y + self.radius*0.3*np.sin(self.orientation + np.pi/2), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )

        right = (
            w2px(self.x + self.radius*0.3*np.cos(self.orientation - np.pi/2), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y + self.radius*0.3*np.sin(self.orientation - np.pi/2), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )

        front = (
            w2px(self.x + self.radius*0.85*np.cos(self.orientation), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y + self.radius*0.85*np.sin(self.orientation), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )
        back = (
            w2px(self.x - self.radius*0.5*np.cos(self.orientation), PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y - self.radius*0.5*np.sin(self.orientation), PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )
        center = (
            w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
            w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
        )


        # drawing a polygonlines to get sense of the orientation of the robot.
        points = np.array([left, right, front, left], dtype=np.int32)
        cv2.fillPoly(img,  [points], color=(27, 194, 169))
        cv2.line(img, back, center, (27, 194, 169), int(0.05*PIXEL_TO_WORLD_X))

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


