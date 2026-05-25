import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2
### Hamna
### Hamna
### Hamna
import time as pytime
from collections import deque

MAX_TIME_TO_REACH_GOAL = 50
### Hamna
### Hamna
### Hamna

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
        fov=2*np.pi,
        pos_noise_std=None,
        angle_noise_std=None
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
        self.pos_noise_std = pos_noise_std if pos_noise_std!=None else 0
        self.angle_noise_std = angle_noise_std if angle_noise_std!=None else 0

### Hamna
### Hamna
### Hamna
        self.prev_x = None
        self.prev_y = None
        self.prev_orientation = None
        self.position_history = deque(maxlen=8)
        self.update_skip_probability = 0.05
        self.perception_delay_steps = 1
        self.delayed_orientation = theta if theta is not None else 0.0
        self.max_rotation_speed = np.pi / 6

        self.avoidance_turn = 0
        self.avoidance_cooldown = 0
        self.avoidance_duration = 4

        self.stuck_counter = 0
        self.last_step_x = x
        self.last_step_y = y

        self.preferred_speed = speed if speed is not None else 0.0
        self.goal_direction_bias = np.random.normal(0, 0.18)
        self.vx = 0.0
        self.vy = 0.0

        self.estimated_w = 0.0
        self.wander_angle =  np.random.uniform(-0.5, 0.5) #
        self.wander_strength = 0.4   # how much we bend (radians)
        self.wander_smooth = 0.7 #
        self.turn_bias = np.random.normal(0, 0.03)
        self.speed_scale = np.random.uniform(0.9, 1.1)
        self.speed_phase = np.random.uniform(0, 2 * np.pi)
### Hamna
### Hamna
### Hamna
        
        assert(self.type == "static" or self.type == "dynamic"), "type can be \"static\" or \"dynamic\" only."
        self.set(id, x, y, theta, width, speed, goal_x, goal_y, goal_radius, policy)

### Hamna
### Hamna
### Hamna
        self.initial_time = pytime.time()
### Hamna
### Hamna
### Hamna

    def set_goal(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
### Hamna
### Hamna
### Hamna
        self.initial_time = pytime.time()
### Hamna
### Hamna
### Hamna

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
        self.initial_x = x
        self.initial_y = y
        self.initial_orientation = theta
### Hamna
### Hamna
### Hamna
        self.prev_x = x
        self.prev_y = y
        self.stuck_counter = 0
        self.last_step_x = x
        self.last_step_y = y
        self.delayed_orientation = theta if theta is not None else 0.0
        self.goal_direction_bias = np.random.normal(0, 0.18)
        self.turn_bias = np.random.normal(0, 0.03)
                
        self.estimated_w = 0.0


        if x is not None and y is not None and theta is not None:
            self.position_history.clear()
            self.position_history.append((x, y, theta, pytime.time()))
### Hamna
### Hamna
### Hamna
    

    def has_reached_goal(self, offset=None):
        if offset is None: offset = self.width/2
        if self.type == "static": return False  # static humans do not have goals, so they would not reach their goal
        # if self.width == None or self.goal_radius == None or self.goal_x==None or self.goal_y == None: return False
        distance_to_goal = np.sqrt((self.x-self.goal_x)**2 + (self.y-self.goal_y)**2)
### Hamna
### Hamna
### Hamna
        if distance_to_goal < (offset + self.goal_radius) or pytime.time()-self.initial_time>MAX_TIME_TO_REACH_GOAL:
            return True
        else:
            return False
### Hamna
### Hamna
### Hamna

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

### Hamna
### Hamna
### Hamna
        angle_diff = atan2(np.sin(theta - self.orientation), np.cos(theta - self.orientation))

        if angle_diff > self.max_rotation_speed:#_________
            angle_diff = self.max_rotation_speed
        elif angle_diff < -self.max_rotation_speed:
            angle_diff = -self.max_rotation_speed

        self.orientation = atan2(
            np.sin(self.orientation + angle_diff),
            np.cos(self.orientation + angle_diff)
        )
### Hamna
### Hamna
### Hamna


    def update(self, time):
        """
        For updating the coordinates of the human for a single time step
        """
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"
        # if self.type == "static": return  # static humans do not change their position
        if self.type == "static":
            self.x = self.initial_x
            self.y = self.initial_y
            self.orientation = self.initial_orientation
### Hamna
### Hamna
### Hamna
            self.position_history.append((self.x, self.y, self.orientation, pytime.time()))
            return


        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_orientation = self.orientation

        if np.random.random() < self.update_skip_probability:
            self.position_history.append((self.x, self.y, self.orientation, pytime.time()))
            return

        r_moved = np.random.normal(0, self.pos_noise_std)
        self.speed_phase += 0.15#______
        rhythmic_scale = 1.0 + 0.06 * np.sin(self.speed_phase)#_____
        moved = time * (self.speed * self.speed_scale * rhythmic_scale) + r_moved
### Hamna
### Hamna
### Hamna
        r_angle = np.random.normal(0, self.angle_noise_std)
        self.initial_x = self.x
        self.initial_y = self.y
        self.initial_orientation = self.orientation


        
        self.orientation = self.orientation + r_angle


### Hamna
### Hamna
### Hamna
        if len(self.position_history) > self.perception_delay_steps: #___
            delayed_state = self.position_history[-1 - self.perception_delay_steps]  #kalper
            delayed_theta = delayed_state[1]
        else:
            delayed_theta = self.orientation

        self.delayed_orientation = delayed_theta #_______
#________
        new_wander = np.random.normal(0, self.wander_strength)
        self.wander_angle = self.wander_smooth * self.wander_angle + (1 - self.wander_smooth) * new_wander

        wander_factor = min(1.0, max(0.35, self.speed / 0.6))
        wandered_orientation = self.orientation + self.wander_angle + self.turn_bias
        self.x += moved * np.cos(wandered_orientation)
        self.y += moved * np.sin(wandered_orientation)
#_________

        self.vx = self.speed * np.cos(self.orientation)
        self.vy = self.speed * np.sin(self.orientation)

        self.position_history.append((self.x, self.y, self.orientation, pytime.time()))
### Hamna
### Hamna
### Hamna


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
