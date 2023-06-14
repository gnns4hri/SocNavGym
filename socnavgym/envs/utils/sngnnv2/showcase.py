import math
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import glob
import copy
import sys
import os

sys.path.append("..")

from socnav_V2_API import *
from socnav import *

import sys
import math
import random

from beautify import *

from shapely.geometry import Point, Polygon

if len(sys.argv) != 3:
    print(f"Usage: 'python3 {sys.argv[0]} model_directory resolution'")
    sys.exit(0)

scenario_list = ["jsons_test/S1_000000.json", "jsons_test/S1_000004.json", "jsons_test/S2_000000.json",
                 "jsons_test/S2F_00000.json", "jsons_test/S2FL_000000.json"]

# scenario_list = ["jsons_test/" + sys.argv[2]]


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


sngnn = SocNavAPI(device="cpu")  # change to cpu when no gpu


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



###
###  C O N F I G    B L O C K
###
base = "showcase"
bins = int(sys.argv[2])  # 80
l_img = 6.5
# bins = int(sys.argv[1])
tick = 0
params = {}
x = z = angle = 0
M = get_transformation_matrix_for_pose(x, z, angle)

youbotL = 0.580
youbotW = 0.376
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
        params["tick"] = tick
        num_str = str(tick).zfill(3)
        dst_str_a = base + "_" + fnamee + "_"
        dst_str_b_q1 = "_" + num_str + "_Q1.png"
        dst_str_b_q2 = "_" + num_str + "_Q2.png"

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
                                objectt["size_x],
                                objectt["size_y"],
                            )
                        )
                    room_map = []
                    room_poly = []
                    for wall in data_structure["walls"]:
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

                        room_map.append(new_wall)
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
                    v_q2 = set_in_range(ret_gnn[1].item(), 0., 1.,) * 255
                else:
                    if white_zone:
                        v_q1 = 255
                        v_q2 = 255
                    else:
                        v_q1 = 128
                        v_q2 = 128                        
                # print(v)
                z_q1[y_i, x_i] = v_q1
                z_q2[y_i, x_i] = v_q2

        z_q1 = z_q1.astype(np.uint8)
        z_q1 = cv2.flip(z_q1, 0)
        resized_q1 = cv2.resize(z_q1, (640, 640), interpolation=cv2.INTER_NEAREST)
        resized_q1 = beautify_image(cv2.cvtColor(resized_q1, cv2.COLOR_GRAY2BGR))



        z_q2 = z_q2.astype(np.uint8)
        z_q2 = cv2.flip(z_q2, 0)
        resized_q2 = cv2.resize(z_q2, (640, 640), interpolation=cv2.INTER_NEAREST)
        resized_q2 = beautify_image(cv2.cvtColor(resized_q2, cv2.COLOR_GRAY2BGR))

        # DRAW WALLS
        for wall in data_sequence[-1]["walls"]:
            x1_img = (wall["x1"] + l_img) * 640 / (2 * l_img)
            y1_img = (-wall["y1"] + l_img) * 640 / (2 * l_img)
            x2_img = (wall["x2"] + l_img) * 640 / (2 * l_img)
            y2_img = (-wall["y2"] + l_img) * 640 / (2 * l_img)
            thickness = 7
            color = (0, 0, 0)
            resized_q1 = cv2.line(
                resized_q1,
                (int(x1_img), int(y1_img)),
                (int(x2_img), int(y2_img)),
                color,
                thickness,
            )
            resized_q2 = cv2.line(
                resized_q2,
                (int(x1_img), int(y1_img)),
                (int(x2_img), int(y2_img)),
                color,
                thickness,
            )


        rows, cols = resized_q1.shape[0:2]
        Mr = cv2.getRotationMatrix2D((cols/2,rows/2),-data_sequence[-1]['robot_pose']['a']*180./math.pi,1)
        final_img_q1 = cv2.warpAffine(resized_q1, Mr, (rows, cols),borderValue = (255, 255, 255))
        final_img_q2 = cv2.warpAffine(resized_q2, Mr, (rows, cols),borderValue = (255, 255, 255))

        cv2.imwrite(dst_str_a + "clean" + dst_str_b_q1, final_img_q1)
        cv2.imwrite(dst_str_a + "clean" + dst_str_b_q2, final_img_q2)


        entities = {}
        scale = 50
        # DRAW Humans
        for human in data_sequence[-1]["people"]:
            x_img = (human["x"] + l_img) * 640 / (2 * l_img)
            y_img = (-human["y"] + l_img) * 640 / (2 * l_img)
            center_coordinates = (int(x_img), int(y_img))
            entities[human["id"]] = center_coordinates
            radius = 17
            thickness = 2
            color = (255, 0, 0)
            # import math

            # vx = int(math.cos(human["a"]) * scale)
            # vy = int(math.sin(human["a"]) * scale)
            # resized = cv2.line(resized, center_coordinates, (int(x_img)+vx, int(y_img)+vy), color, 2)
            resized_q1 = cv2.circle(resized_q1, center_coordinates, radius, color, thickness)
            resized_q2 = cv2.circle(resized_q2, center_coordinates, radius, color, thickness)            

        # DRAW Objects
        for object in data_sequence[-1]["objects"]:
            x_img = (object["x"] + l_img) * 640 / (2 * l_img)
            y_img = (-object["y"] + l_img) * 640 / (2 * l_img)
            center_coordinates = (int(x_img), int(y_img))
            entities[object["id"]] = center_coordinates
            radius = 12
            thickness = 2
            color = (0, 255, 0)
            resized_q1 = cv2.circle(resized_q1, center_coordinates, radius, color, thickness)
            resized_q2 = cv2.circle(resized_q2, center_coordinates, radius, color, thickness)            

        # DRAW Goals
        for goal in data_sequence[-1]["goal"]:
            x_img = (goal["x"] + l_img) * 640 / (2 * l_img)
            y_img = (-goal["y"] + l_img) * 640 / (2 * l_img)
            center_coordinates = (int(x_img), int(y_img))
            radius = 20
            thickness = 2
            color = (0, 155, 0)
            resized_q1 = cv2.circle(resized_q1, center_coordinates, radius, color, thickness)
            resized_q2 = cv2.circle(resized_q2, center_coordinates, radius, color, thickness)            

        # DRAW Interactions
        for interaction in data_sequence[-1]["interaction"]:
            resized_q1 = cv2.line(
                resized_q1,
                entities[interaction["src"]],
                entities[interaction["dst"]],
                (50, 0, 155),
                4,
            )
            resized_q2 = cv2.line(
                resized_q2,
                entities[interaction["src"]],
                entities[interaction["dst"]],
                (50, 0, 155),
                4,
            )


        # DRAW Humans' orientations
        for human in data_sequence[-1]["people"]:
            center_coordinates = entities[human["id"]]
            thickness = 2
            color = (255, 0, 0)
            import math

            vx = int(math.cos(human["a"]) * scale/2)
            vy = int(math.sin(human["a"]) * scale/2)
            resized_q1 = cv2.line(resized_q1, center_coordinates, (center_coordinates[0]+vx, center_coordinates[1]+vy), color, thickness)
            resized_q2 = cv2.line(resized_q2, center_coordinates, (center_coordinates[0]+vx, center_coordinates[1]+vy), color, thickness)            

        final_img_q1 = cv2.warpAffine(resized_q1, Mr, (rows, cols), borderValue = (255, 255, 255))
        final_img_q2 = cv2.warpAffine(resized_q2, Mr, (rows, cols), borderValue = (255, 255, 255))        
        cv2.imwrite(dst_str_a + "filled" + dst_str_b_q1, final_img_q1)
        cv2.imwrite(dst_str_a + "filled" + dst_str_b_q2, final_img_q2)        

