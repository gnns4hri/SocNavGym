import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2

class Human(Object):
    """
    Class for humans
    """

    def __init__(
        self, 
        id=None, 
        x=None, 
        y=None, 
        theta=None, 
        width=None, 
        speed=None, 
        goal_x=None, 
        goal_y=None, 
        goal_radius=None, 
        policy=None,
        prob_to_avoid_robot=0.05,
        type="dynamic",
        fov=2*np.pi
    ) -> None:
        super().__init__(id, "human")
        self.width = None  # diameter of the human
        self.is_static = False  # humans can move, so is_static is False
        self.speed = 0  # linear speed
        self.collided_object = None  # name of the object with which collision has happened
        self.goal_x = None  # x coordinate of the goal
        self.goal_y = None  # y coordinate of the goal
        self.goal_radius = None # goal radius
        self.policy = None  # policy is sfm or orca
        self.prob_to_avoid_robot = prob_to_avoid_robot
        self.fov = fov  # field of view
        self.type = type  # whether human is static or dynamic
        assert(self.type == "static" or self.type == "dynamic"), "type can be \"static\" or \"dynamic\" only."
        self.set(id, x, y, theta, width, speed, goal_x, goal_y, goal_radius, policy)

    def set_goal(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y

    def set(self, id, x, y, theta, width, speed, goal_x, goal_y, goal_radius, policy):
        super().set(id, x, y, theta)
        self.width = width
        if self.width is not None:
            self.length = width * 0.2  # thickness of the shoulder (for visualization)
            self.radius = width / 5  # radius of head (for visualization)
        if speed is not None:
            self.speed = speed  # speed
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_radius = goal_radius
        self.policy = policy

    def has_reached_goal(self, offset=None):
        if offset is None: offset = self.width/2
        if self.type == "static": return False  # static humans do not have goals, so they would not reach their goal
        # if self.width == None or self.goal_radius == None or self.goal_x==None or self.goal_y == None: return False
        distance_to_goal = np.sqrt((self.x-self.goal_x)**2 + (self.y-self.goal_y)**2)
        if distance_to_goal < (offset + self.goal_radius):
            return True
        else:
            return False
    @property
    def avoids_robot(self):
        n = np.random.random()
        if n <= self.prob_to_avoid_robot:
            return True
        else:
            return False
    
    def set_new_orientation_with_limits(self, orientation, max_rotation_speed, time):
        diffO = atan2(np.sin(orientation-self.orientation), np.cos(orientation-self.orientation))
        if abs(diffO)/time>max_rotation_speed:
            if diffO>0:
                diffO = max_rotation_speed*time
            else:
                diffO = -max_rotation_speed*time
            self.orientation = atan2(np.sin(self.orientation+diffO), np.cos(self.orientation+diffO))
            return False

        self.orientation = orientation
        return True


    def update_orientation(self, theta):
        if self.type == "static": return  # static humans do not change their orientation
        self.orientation = theta

    def update(self, time):
        """
        For updating the coordinates of the human for a single time step
        """
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"
        if self.type == "static": return  # static humans do not change their position
        moved = time * self.speed  # distance moved = speed x time
        self.x += moved * np.cos(self.orientation)  # updating x position
        self.y += moved * np.sin(self.orientation)  # updating y position

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        if self.color == None:
            color = (240, 114, 66)  # blue
        else:
            color = self.color
        assert self.width != None, "Width is None type."
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"

        # p1, p2, p3, p4 are the coordinates of the corners of the rectangle. calculation is done so as to orient the rectangle at an angle.

        p1 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p2 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p3 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p4 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]
        points = np.array([p1, p2, p3, p4])
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(
            img, [np.int32(points)], color
        )  # filling the rectangle made from the points with the specified color
        cv2.polylines(
            img, [np.int32(points)], True, (0, 0, 0), 2
        )  # bordering the rectangle

        black = (0, 0, 0)  # color for the head
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        )  # calculating no. of pixels corresponding to the radius

        cv2.circle(
            img,
            (
                w2px(
                    self.x + (self.width / 10) * np.cos(self.orientation),
                    PIXEL_TO_WORLD_X,
                    MAP_SIZE_X,
                ),
                w2py(
                    self.y + (self.width / 10) * np.sin(self.orientation),
                    PIXEL_TO_WORLD_Y,
                    MAP_SIZE_Y,
                ),
            ),
            radius,
            black,
            -1,
        )  # drawing a circle for the head of the human

    def draw_gaze_range(self, img, gaze_angle, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        center = (w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X), w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y))
        radius = w2px(self.x + np.sqrt(MAP_SIZE_X**2 + MAP_SIZE_Y**2), PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        )  # calculating no. of pixels corresponding to the radius
       
        axesLength = (radius, radius)
        gaze_angle = gaze_angle * 180 / np.pi
        orientation = self.orientation * 180 / np.pi

        cv2.ellipse(
            img,
            center,
            axesLength,
            angle=-orientation,
            startAngle=(-gaze_angle/2),
            endAngle=(gaze_angle/2),
            color=(218, 252, 81), 
            thickness=-1
        )
