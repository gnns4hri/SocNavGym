import math
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import glob
import copy
import sys
import os
from pathlib import Path
sys.path.append("..")

from socnav_V2_API import *
from socnav import *

import sys
import math
import random

from beautify import *

from shapely.geometry import Point, Polygon

if len(sys.argv) < 2 or len(sys.argv) > 4:
    print("Please use this format: 'python3 showcase_static.py 'model_directory' 'file.json' resolution'")
    sys.exit(0)

# scenario_list = ["jsons_test/scenario1/S1_000000.json", "jsons_test/scenario1/S1_000004.json", "jsons_test/scenario2/S2_000000.json",
#                  "jsons_test/scenario2/S2F_00000.json", "jsons_test/scenario2/S2FL_000000.json"]

scenario_list = [sys.argv[2]]


def get_transformation_matrix_for_pose(x, z, angle):
    M = np.zeros((3, 3))
    M[0][0] = +math.cos(-angle)
    M[0][1] = -math.sin(-angle)
    M[0][2] = x
    M[1][0] = +math.sin(-angle)
    M[1][1] = +math.cos(-angle)
    M[1][2] = z
    M[2][2] = 1.0
    M = np.linalg.inv(M)
    return M


def radians_to_degrees(a):
    angle = a * 180.0 / math.pi
    while angle >= 180.0:
        angle -= 360
    while angle <= -180.0:
        angle += 360
    return angle


sngnn = SocNavAPI(base=sys.argv[1], device="cuda")  # change to cpu when no gpu


def transform(world, params):
    w = copy.deepcopy(world)
    if not tick:
        w["links"] = []
    return w

def set_in_range(v, a, b):
    if v > b:
        return b
    if v < a:
        return a
    return v

def extend_walls(walls, e, ang):
    new_walls = copy.deepcopy(walls)
    if len(walls) == 4:
        p1_r = (e, e);  p2_r = (e, -e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[0]['x1'] += p1[0];  new_walls[0]['y1'] += p1[1];  new_walls[0]['x2'] += p2[0];  new_walls[0]['y2'] += p2[1]
        p1_r = (e, -e);  p2_r = (-e, -e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[1]['x1'] += p1[0];  new_walls[1]['y1'] += p1[1];  new_walls[1]['x2'] += p2[0];  new_walls[1]['y2'] += p2[1]
        p1_r = (-e, -e);  p2_r = (-e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[2]['x1'] += p1[0];  new_walls[2]['y1'] += p1[1];  new_walls[2]['x2'] += p2[0];  new_walls[2]['y2'] += p2[1]
        p1_r = (-e, e);  p2_r = (e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[3]['x1'] += p1[0];  new_walls[3]['y1'] += p1[1];  new_walls[3]['x2'] += p2[0];  new_walls[3]['y2'] += p2[1]
    else:
        p1_r = (e, e);  p2_r = (e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[0]['x1'] += p1[0];  new_walls[0]['y1'] += p1[1];  new_walls[0]['x2'] += p2[0];  new_walls[0]['y2'] += p2[1]
        p1_r = (e, e);  p2_r = (-e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[1]['x1'] += p1[0];  new_walls[1]['y1'] += p1[1];  new_walls[1]['x2'] += p2[0];  new_walls[1]['y2'] += p2[1]
        p1_r = (-e, e);  p2_r = (-e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[2]['x1'] += p1[0];  new_walls[2]['y1'] += p1[1];  new_walls[2]['x2'] += p2[0];  new_walls[2]['y2'] += p2[1]
        p1_r = (-e, e);  p2_r = (-e, -e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[3]['x1'] += p1[0];  new_walls[3]['y1'] += p1[1];  new_walls[3]['x2'] += p2[0];  new_walls[3]['y2'] += p2[1]
        p1_r = (-e, -e);  p2_r = (e, -e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[4]['x1'] += p1[0];  new_walls[4]['y1'] += p1[1];  new_walls[4]['x2'] += p2[0];  new_walls[4]['y2'] += p2[1]
        p1_r = (e, -e);  p2_r = (e, -e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[5]['x1'] += p1[0];  new_walls[5]['y1'] += p1[1];  new_walls[5]['x2'] += p2[0];  new_walls[5]['y2'] += p2[1]
        p1_r = (e, -e);  p2_r = (e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[6]['x1'] += p1[0];  new_walls[6]['y1'] += p1[1];  new_walls[6]['x2'] += p2[0];  new_walls[6]['y2'] += p2[1]
        p1_r = (e, e);  p2_r = (e, e)
        p1 = (p1_r[0]*math.cos(ang) + p1_r[1]*math.sin(ang), -p1_r[0]*math.sin(ang) + p1_r[1]*math.cos(ang))
        p2 = (p2_r[0]*math.cos(ang) + p2_r[1]*math.sin(ang), -p2_r[0]*math.sin(ang) + p2_r[1]*math.cos(ang))
        new_walls[7]['x1'] += p1[0];  new_walls[7]['y1'] += p1[1];  new_walls[7]['x2'] += p2[0];  new_walls[7]['y2'] += p2[1]
        
    return new_walls


###
###  C O N F I G    B L O C K
###
base = "images_dataset/"
Path(base).mkdir(parents=True, exist_ok=True)

bins = int(sys.argv[3])  # 80
l_img = 6.5
# bins = int(sys.argv[1])
tick = 0
params = {}
x = z = angle = 0
M = get_transformation_matrix_for_pose(x, z, angle)

youbotL = 0.576
youbotW = 0.576
robot = Polygon(
    [
        [youbotW / 2, youbotL / 2],
        [-youbotW / 2, youbotL / 2],
        [-youbotW / 2, -youbotL / 2],
        [youbotW / 2, -youbotL / 2],
    ]
)


for scenario in scenario_list:
    fnamee = scenario
    if scenario.endswith('.json'):
        fnamee = fnamee[:-5]
    fnamee = fnamee.split('/')[-1]
    for tick in [1]:
        with open(scenario, "r") as f:
            data_sequence = json.loads(f.read())

        for id in range(len(data_sequence)):
            data_sequence[id]['command'] = [0., 0., 0.]
            data_sequence[id]['extended_walls'] = extend_walls(data_sequence[id]['walls'], 1.5, -data_sequence[id]['robot_pose']['a'])
        params["tick"] = tick
        num_str = str(tick).zfill(3)
        dst_str_a = base + fnamee + "_"
        dst_str_b_q1 = "_Q1.png"
        dst_str_b_q2 = "_Q2.png"

        print("Processing frame", tick)

        z_q1 = np.zeros((bins, bins))
        z_q2 = np.zeros((bins, bins))
        xs = np.linspace(-l_img, l_img, bins)
        ys = np.linspace(-l_img, l_img, bins)
        for x_i, x in enumerate(xs):
            if x_i % 5 == 0:
                print(x_i)
            for y_i, y in enumerate(ys):
                sn_sequence = []
                within_room = True
                white_zone = True
                last_frame_room = None
                cur_pose = data_sequence[-1]["robot_pose"]
                xn = x
                yn = y
                for data_structure in reversed(data_sequence):
                    diff_angle = data_structure["robot_pose"]["a"] - cur_pose["a"]
                    diff_x = data_structure["robot_pose"]["x"] - cur_pose["x"]
                    diff_y = data_structure["robot_pose"]["y"] - cur_pose["y"]
                    Mr = get_transformation_matrix_for_pose(-diff_x, -diff_y, diff_angle)
                    POS = np.array(
                        [[xn + cur_pose["x"]], [yn + cur_pose["y"]], [1.0]], dtype=float
                    )
                    POS = Mr.dot(POS)
                    xn = POS[0][0] - data_structure["robot_pose"]["x"]
                    yn = POS[1][0] - data_structure["robot_pose"]["y"]
                    cur_pose = data_structure["robot_pose"]
                    sn = SNScenario(data_structure["timestamp"])
                    POS = np.array(
                        [
                            [data_structure["goal"][0]["x"] * 10],
                            [data_structure["goal"][0]["y"] * -10],
                            [1.0],
                        ],
                        dtype=float,
                    )
                    POS = M.dot(POS)
                    POS /= 10
                    POS[1][0] *= -1
                    sn.add_goal(POS[0][0] - xn, POS[1][0] - yn)
                    sn.add_command(data_structure["command"])
                    for human in data_structure["people"]:
                        POS = np.array(
                            [[human["x"] * 10], [human["y"] * -10], [1.0]], dtype=float
                        )  # WARNING THE INPUT VECTOR MUST BE IN PILAR BACHILLER'S FR SYSTEM!!!!
                        POS = M.dot(POS)
                        POS /= 10
                        POS[1][0] *= -1
                        sn.add_human(
                            Human(
                                human["id"],
                                POS[0][0] - xn,
                                POS[1][0] - yn,
                                human["a"] - radians_to_degrees(angle),
                                human["vx"],
                                human["vy"],
                                human["va"],
                            )
                        )
                    for objectt in data_structure["objects"]:
                        POS = np.array(
                            [[objectt["x"] * 10], [objectt["y"] * -10], [1.0]], dtype=float
                        )  # WARNING THE INPUT VECTOR MUST BE IN PILAR BACHILLER'S FR SYSTEM!!!!
                        POS = M.dot(POS)
                        POS /= 10
                        POS[1][0] *= -1
                        sn.add_object(
                            Object(
                                objectt["id"],
                                POS[0][0] - xn,
                                POS[1][0] - yn,
                                objectt["a"] - radians_to_degrees(angle),
                                objectt["vx"],
                                objectt["vy"],
                                objectt["va"],
                                objectt["size_x"],
                                objectt["size_y"],
                            )
                        )
                    room_map = []
                    room_poly = []
                    for wall, ext_wall in zip(data_structure["walls"], data_structure["extended_walls"]):
                        new_ext_wall = {}
                        point1 = [ext_wall["x1"], ext_wall["y1"]]
                        POS1 = np.array(
                            [[point1[0] * 10], [point1[1] * -10], [1.0]], dtype=float
                        )  # WARNING THE INPUT VECTOR MUST BE IN PILAR BACHILLER'S FR SYSTEM!!!!
                        POS1 = M.dot(POS1)
                        POS1 /= 10
                        POS1[1][0] *= -1
                        point2 = [ext_wall["x2"], ext_wall["y2"]]
                        POS2 = np.array(
                            [[point2[0] * 10], [point2[1] * -10], [1.0]], dtype=float
                        )  # WARNING THE INPUT VECTOR MUST BE IN PILAR BACHILLER'S FR SYSTEM!!!!
                        POS2 = M.dot(POS2)
                        POS2 /= 10
                        POS2[1][0] *= -1

                        new_ext_wall["x1"] = POS1[0][0] - xn
                        new_ext_wall["y1"] = POS1[1][0] - yn
                        new_ext_wall["x2"] = POS2[0][0] - xn
                        new_ext_wall["y2"] = POS2[1][0] - yn

                        new_wall = {}
                        point1 = [wall["x1"], wall["y1"]]
                        POS1 = np.array(
                            [[point1[0] * 10], [point1[1] * -10], [1.0]], dtype=float
                        )  # WARNING THE INPUT VECTOR MUST BE IN PILAR BACHILLER'S FR SYSTEM!!!!
                        POS1 = M.dot(POS1)
                        POS1 /= 10
                        POS1[1][0] *= -1
                        point2 = [wall["x2"], wall["y2"]]
                        POS2 = np.array(
                            [[point2[0] * 10], [point2[1] * -10], [1.0]], dtype=float
                        )  # WARNING THE INPUT VECTOR MUST BE IN PILAR BACHILLER'S FR SYSTEM!!!!
                        POS2 = M.dot(POS2)
                        POS2 /= 10
                        POS2[1][0] *= -1

                        new_wall["x1"] = POS1[0][0] - xn
                        new_wall["y1"] = POS1[1][0] - yn
                        new_wall["x2"] = POS2[0][0] - xn
                        new_wall["y2"] = POS2[1][0] - yn

                        room_poly.append((new_wall["x1"], new_wall["y1"]))
                        room_poly.append((new_wall["x2"], new_wall["y2"]))

                        room_map.append(new_ext_wall)
                    sn.add_room(room_map)
                    if last_frame_room is None:
                        last_frame_room = room_poly


                    for interaction in data_structure["interaction"]:
                        sn.add_interaction([interaction["dst"], interaction["src"]])

                    # UNCOMMENT FOR NOT CONSIDERING THE SHAPE OF THE ROBOT
                    # robot = Point(0,0)

                    # UNCOMMENT FOR COMPUTING ALL THE POSITIONS
                    # sn_sequence.append(sn.to_json())

                    # COMMENT FOR COMPUTING ALL THE POSITIONS
                    if robot.within(Polygon(room_poly)):
                        if within_room:
                            sn_sequence.append(sn.to_json())
                    else:
                        within_room = False

                robot_point = Point(0,0)
                if robot_point.within(Polygon(last_frame_room)):
                    white_zone = False

                if (
                    within_room
                ):  # within_room: # use within_room to restrict even more the valid positions of the robot
                    graph = SocNavDataset(sn_sequence, "1", "test", verbose=False)
                    ret_gnn = sngnn.predictOneGraph(graph)[0]
                    v_q1 = set_in_range(ret_gnn[0].item(), 0., 1.,) * 255
                    # v_q2 = set_in_range(ret_gnn[1].item(), 0., 1.,) * 255
                else:
                    if white_zone:
                        v_q1 = 255
                        # v_q2 = 255
                    else:
                        v_q1 = 0
                        # v_q2 = 128
                # print(v)
                z_q1[y_i, x_i] = v_q1
                # z_q2[y_i, x_i] = v_q2

        z_q1 = z_q1.astype(np.uint8)
        z_q1 = cv2.flip(z_q1, 0)
        resized_q1 = z_q1

        rows, cols = resized_q1.shape[0:2]
        

        cv2.imwrite(dst_str_a + dst_str_b_q1, resized_q1)
