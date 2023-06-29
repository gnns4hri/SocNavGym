"""Utility functions"""

import numpy as np
import random

def w2px(x, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given x-coordinate in world frame, to get the x-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * (x + (MAP_SIZE / 2)))


def w2py(y, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given y-coordinate in world frame, to get the y-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * ((MAP_SIZE / 2) - y))

def get_coordinates_of_rotated_rectangle(x, y, orientation, length, width):
    """
    Gives the coordinates of the endpoints of a rectangle centered at (x, y) and has an orientation (given by orientation)
    Returns as a list
    """
    p1 = (
            x + length / 2 * np.cos(orientation)- width / 2 * np.sin(orientation),
            y + length / 2 * np.sin(orientation)+ width / 2 * np.cos(orientation)
        )
    

    p2 = (
            x - length / 2 * np.cos(orientation)- width / 2 * np.sin(orientation),
            y - length / 2 * np.sin(orientation)+ width / 2 * np.cos(orientation)
        )
    


    p3 = (
            x - length / 2 * np.cos(orientation)+ width / 2 * np.sin(orientation),
            y - length / 2 * np.sin(orientation)- width / 2 * np.cos(orientation)
        )
    

    p4 = (
            x + length / 2 * np.cos(orientation)+ width / 2 * np.sin(orientation),
            y + length / 2 * np.sin(orientation)- width / 2 * np.cos(orientation)
        )
    

    return [p1, p2, p3, p4]

def get_square_around_circle(x, y, r):
    p1 = (x+r, y+r)
    p2 = (x-r, y+r)
    p3 = (x-r, y-r)
    p4 = (x+r, y-r)
    return [p1, p2, p3, p4]


def get_coordinates_of_rotated_line(x, y, orientation, length):
    """
    Gives the coordinates of the endpoints of a line centered at (x, y) and has an orientation (given by orientation)
    Returns as a list
    """
    p1 = (
        x + (length/2)*np.cos(orientation),
        y + (length/2)*np.sin(orientation)
    )

    p2 = (
        x - (length/2)*np.cos(orientation),
        y - (length/2)*np.sin(orientation)
    )

    return [p2, p1]

def uniform_circular_sampler(center_x, center_y, radius):
    """
    For sampling uniformly in a circle with center at (center_x, center_y) and radius given by radius.
    """
    theta = 2 * np.pi * random.random()
    u = random.random() * radius
    point = (center_x + u*np.cos(theta), center_y + u*np.sin(theta))
    return point

def get_nearest_point_from_rectangle(center_x, center_y, length, width, orientation, point_x, point_y):
    
    # transformation matrix for coordinate frame along the edges of the rectangle
    tm = np.zeros((3,3), dtype=np.float32)
    # filling values as described
    tm[2,2] = 1
    tm[0,2] = 0
    tm[1,2] = 0
    tm[0,0] = tm[1,1] = np.cos(orientation)
    tm[1,0] = np.sin(orientation)
    tm[0,1] = -1*np.sin(orientation)

    tm_inv = np.linalg.inv(tm)
    l = get_coordinates_of_rotated_rectangle(center_x, center_y, orientation, length, width)
    rotated_points = []
    for point in l:
        coord = np.array([[point[0], point[1]]])
        # converting the coordinates to homogeneous coordinates
        homogeneous_coordinates = np.c_[coord, np.ones((coord.shape[0], 1))]
        # getting the rotated frame coordinates by multiplying with the transformation matrix
        coord_in_robot_frame = (tm_inv@homogeneous_coordinates.T).T
        ans =  coord_in_robot_frame[:, 0:2]
        rotated_points.append(ans)
    
    coord = np.array([[point_x, point_y]])
    # converting the coordinates to homogeneous coordinates
    homogeneous_coordinates = np.c_[coord, np.ones((coord.shape[0], 1))]
    # getting the rotated frame coordinates by multiplying with the transformation matrix
    coord_in_robot_frame = (tm_inv@homogeneous_coordinates.T).T
    ans =  coord_in_robot_frame[:, 0:2]
    query_point = ans    
    query_point = query_point.flatten()
    min_x = min(rotated_points[0][0][0], rotated_points[1][0][0], rotated_points[2][0][0], rotated_points[3][0][0])
    max_x = max(rotated_points[0][0][0], rotated_points[1][0][0], rotated_points[2][0][0], rotated_points[3][0][0])
    min_y = min(rotated_points[0][0][1], rotated_points[1][0][1], rotated_points[2][0][1], rotated_points[3][0][1])
    max_y = max(rotated_points[0][0][1], rotated_points[1][0][1], rotated_points[2][0][1], rotated_points[3][0][1])
    
    
    if query_point[0] <= min_x: dx = min_x
    elif query_point[0] >= max_x: dx = max_x
    else: dx = query_point[0]

    if query_point[1] <= min_y: dy = min_y
    elif query_point[1] >= max_y: dy = max_y
    else: dy = query_point[1]

    homogeneous_coordinates = np.c_[np.array([[dx, dy]]), np.ones((coord.shape[0], 1))]
    coord_in_orig_frame = (tm@homogeneous_coordinates.T).T
    ans = coord_in_orig_frame[:, 0:2]
    return (ans[0,0], ans[0,1])

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)
    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def convert_angle_to_minus_pi_to_pi(angle):
    if angle > 2*np.pi:
        angle -= int(angle/(2*np.pi))*(2*np.pi)
    if angle < -2*np.pi:
        angle += int(abs(angle)/(2*np.pi))*(2*np.pi)

    if angle > np.pi: angle -= 2*np.pi
    elif angle < -np.pi: angle += 2*np.pi

    return angle


def compute_time_to_collision(robot_x, robot_y, robot_vx, robot_vy, human_x, human_y, human_vx, human_vy, robot_radius, human_radius):
    robot_vel = np.array([robot_vx, robot_vy], dtype=np.float32)
    human_vel = np.array([human_vx, human_vy], dtype=np.float32)
    robot_pos = np.array([robot_x, robot_y], dtype=np.float32)
    human_pos = np.array([human_x, human_y], dtype=np.float32)

    relative_vel = human_vel - robot_vel
    relative_pos = robot_pos - human_pos

    if np.linalg.norm(relative_pos) <= (robot_radius + human_radius):
        return 0
    
    relative_pos_unit_vector = relative_pos / np.linalg.norm(relative_pos)
    vel_value_along = np.dot(relative_vel, relative_pos_unit_vector.T)

    if vel_value_along <= 0.0:
        return -1

    time_taken = np.linalg.norm(relative_pos) / vel_value_along
    
    human_new_pos = human_pos + human_vel * time_taken
    robot_new_pos = robot_pos + robot_vel * time_taken

    if np.linalg.norm(human_new_pos - robot_new_pos) <= (robot_radius + human_radius):
        return time_taken

    else:
        return -1
    