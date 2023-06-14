import json
import os
import sys
import math
import numpy as np
import copy
from pathlib import Path
from scipy.interpolate import splprep, splev
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from collections import deque

# minimum difference in x or y for a given entity between two consecutive frames
MIN_DIFF = 0.05
# Radius of the entities considered for calculating collision
ENTITY_RADIUS = 0.15
DIST_THRESHOLD = 0.2
TOTAL_DIVISIONS = 30
EXTRAPOLATION_FACTOR = .2


def get_transformation_matrix_for_pose(x, z, angle):
    M = np.zeros((3, 3))
    M[0][0] = +math.cos(-angle)
    M[0][1] = -math.sin(-angle)
    M[0][2] = x
    M[1][0] = +math.sin(-angle)
    M[1][1] = +math.cos(-angle)
    M[1][2] = z
    M[2][2] = 1.0
    return M


def compute_robot_pose(walls):
    if len(walls) == 4:
        w = walls[3]
        vw = (walls[0]['x2'] - walls[0]['x1'], walls[0]['y2'] - walls[0]['y1'])
        l = math.sqrt(vw[0] * vw[0] + vw[1] * vw[1])
    else:
        w = walls[5]
        # w['x1'], w['x2'] = w['x2'], w['x1']
        # w['y1'], w['y2'] = w['y2'], w['y1']
        vw = (walls[3]['x2'] - walls[3]['x1'], walls[3]['y2'] - walls[3]['y1'])
        l = math.sqrt(vw[0] * vw[0] + vw[1] * vw[1])

    ang = math.atan2(w['y2'] - w['y1'], w['x2'] - w['x1'])
    p = np.array([[(w['x1'] + w['x2']) / 2.], [(w['y1'] + w['y2']) / 2.], [1.0]], dtype=float)
    M = get_transformation_matrix_for_pose(0, 0, ang)
    p = M.dot(p)
    p[0][0] = -p[0][0]
    p[1][0] = l / 2. - p[1][0]
    return p[0][0], p[1][0], ang


if len(sys.argv) < 3:
    print("USAGE: python3 add_robot_pose_to_dataset.py old_file_or_directory new_directory")
    exit()

directory_path = sys.argv[1]
dest_directory = sys.argv[2]

Path(dest_directory).mkdir(parents=True, exist_ok=True)
Path(dest_directory + '_absolute').mkdir(parents=True, exist_ok=True)

if os.path.isfile(directory_path):
    filename = os.path.basename(directory_path)
    directory_path = directory_path.split(filename)[0]
    fileList = [filename]
else:
    fileList = os.listdir(directory_path)

for filename in fileList:
    # try:
    if not filename.endswith('.json'):
        continue

    save = filename
    if os.path.exists(dest_directory + '/' + save):
        continue
    print(filename)

    # Read JSON data into the datastore variable
    if filename:
        with open(directory_path + '/' + filename, 'r') as f:
            datastore = json.load(f)
            f.close()

    if len(datastore[0]['people'])!=len(datastore[-1]['people']) or len(datastore[0]['objects'])!=len(datastore[-1]['objects']):
        continue

    datastore_absolute = copy.deepcopy(datastore)
    robot_pose = dict()
    x, y, a = compute_robot_pose(datastore[-1]['walls'])
    M = get_transformation_matrix_for_pose(x, y, a)
    M0 = np.linalg.inv(M)
    p_q = dict()  # queues for persons
    for i, data in reversed(list(enumerate(datastore))):
        x, y, a = compute_robot_pose(data['walls'])
        robot_pose['x'] = x
        robot_pose['y'] = y
        robot_pose['a'] = a

        M = get_transformation_matrix_for_pose(x, y, a)
        # M = np.dot(M0, M)

        data['robot_pose'] = copy.deepcopy(robot_pose)
        datastore_absolute[i]['robot_pose'] = copy.deepcopy(robot_pose)

        for g in datastore_absolute[i]['goal']:
            point = np.array([[g['x']], [g['y']], [1.0]], dtype=float)
            point = M.dot(point)
            g['x'] = point[0][0]
            g['y'] = point[1][0]

        for p in datastore_absolute[i]['people']:
            point = np.array([[p['x']], [p['y']], [1.0]], dtype=float)
            point = M.dot(point)
            p['x'] = point[0][0]
            p['y'] = point[1][0]
            p['a'] = math.atan2(math.sin(p['a'] + a), math.cos(p['a'] + a))

        for o in datastore_absolute[i]['objects']:
            point = np.array([[o['x']], [o['y']], [1.0]], dtype=float)
            point = M.dot(point)
            o['x'] = point[0][0]
            o['y'] = point[1][0]
            o['a'] = math.atan2(math.sin(o['a'] + a), math.cos(o['a'] + a))

        for w in datastore_absolute[i]['walls']:
            point1 = np.array([[w['x1']], [w['y1']], [1.0]], dtype=float)
            point1 = M.dot(point1)
            point2 = np.array([[w['x2']], [w['y2']], [1.0]], dtype=float)
            point2 = M.dot(point2)
            w['x1'] = point1[0][0]
            w['y1'] = point1[1][0]
            w['x2'] = point2[0][0]
            w['y2'] = point2[1][0]

    datastore = list(reversed(datastore))
    datastore_absolute = list(reversed(datastore_absolute))
    for i in range(len(datastore)):
        entity_splines = []
        entity_coords = []
        collision_coords = []
        # Robot pose
        if i == 0:
            r_q = deque()
            r_q.append([datastore_absolute[i]['robot_pose']['x'], datastore_absolute[i]['robot_pose']['y'],
                        datastore[i]['timestamp']])
        else:
            r_q.append([datastore_absolute[i]['robot_pose']['x'], datastore_absolute[i]['robot_pose']['y'],
                        datastore[i]['timestamp']])
            x_r = [r_q[-1][0]]
            y_r = [r_q[-1][1]]

            if (r_q[-1][2] - r_q[0][2]) < 0.5:
                new_p = r_q[0]
                if math.fabs(new_p[0]-x_r[0]) > MIN_DIFF or math.fabs(new_p[1]-y_r[0]) > MIN_DIFF:
                    x_r.insert(0, r_q[0][0])
                    y_r.insert(0, r_q[0][1])
                time_inc = r_q[-1][2] - r_q[0][2]
            else:
                new_p = r_q[int(len(r_q)/2)]
                if math.fabs(new_p[0]-x_r[0]) > MIN_DIFF or math.fabs(new_p[1]-y_r[0]) > MIN_DIFF:
                    x_r.insert(0, new_p[0])
                    y_r.insert(0, new_p[1])
                new_p = r_q[0]
                if math.fabs(new_p[0]-x_r[0]) > MIN_DIFF or math.fabs(new_p[1]-y_r[0]) > MIN_DIFF:
                    x_r.insert(0, new_p[0])
                    y_r.insert(0, new_p[1])
                time_inc = r_q[int(len(r_q)/2)][2] - r_q[0][2]
                if (r_q[-1][2] - r_q[0][2]) >= 1.:
                    p = r_q.popleft()


            x_r = np.array(x_r)
            y_r = np.array(y_r)

            k = 2 if x_r.size > 2 else 1
            if len(x_r) > 1:
                # extrapolation_amount = EXTRAPOLATION_FACTOR * math.sqrt((x_r[-1] - x_r[-2]) ** 2 + (y_r[-1] - y_r[-2]) ** 2)
                extrapolation_amount = TOTAL_DIVISIONS/(len(x_r)-1)
                tck_r = splprep([x_r, y_r], k=k, s=0)
                ex_r, ey_r = splev(np.linspace(0, extrapolation_amount, TOTAL_DIVISIONS), tck_r[0][0:3], der=0)
            else:
                ex_r = [x_r[0]]*TOTAL_DIVISIONS
                ey_r = [y_r[0]]*TOTAL_DIVISIONS
            entity_splines.append([ex_r, ey_r, 'r'])
            entity_coords.append([x_r, y_r])


        for j, p in enumerate(datastore_absolute[i]['people']):
            if i == 0:
                p['vx'] = 0.0
                p['vy'] = 0.0
                p['va'] = 0.0
            else:
                p['vx'] = datastore_absolute[i]['people'][j]['x'] - datastore_absolute[i-1]['people'][j]['x']
                p['vy'] = datastore_absolute[i]['people'][j]['y'] - datastore_absolute[i-1]['people'][j]['y']
                p['va'] = datastore_absolute[i]['people'][j]['a'] - datastore_absolute[i-1]['people'][j]['a']

            # Calculate time to collision
            if i == 0:
                datastore[i]['people'][j]['t_collision'] = math.inf
                datastore_absolute[i]['people'][j]['t_collision'] = math.inf
                p_q[j] = deque()
                p_q[j].append([p['x'], p['y'], datastore[i]['timestamp']])
            else:
                p_q[j].append([p['x'], p['y'], datastore[i]['timestamp']])
                x_p = [p_q[j][-1][0]]
                y_p = [p_q[j][-1][1]]

                if (p_q[j][-1][2] - p_q[j][0][2]) < 0.5:
                    new_p = p_q[j][0]
                    if math.fabs(new_p[0]-x_p[0]) > MIN_DIFF or math.fabs(new_p[1]-y_p[0]) > MIN_DIFF:
                        x_p.insert(0, new_p[0])
                        y_p.insert(0, new_p[1])
                else:
                    new_p = p_q[j][int(len(p_q[j])/2)]
                    if math.fabs(new_p[0]-x_p[0]) > MIN_DIFF or math.fabs(new_p[1]-y_p[0]) > MIN_DIFF:
                        x_p.insert(0, new_p[0])
                        y_p.insert(0, new_p[1])
                    new_p = p_q[j][0]
                    if math.fabs(new_p[0]-x_p[0]) > MIN_DIFF or math.fabs(new_p[1]-y_p[0]) > MIN_DIFF:
                        x_p.insert(0, new_p[0])
                        y_p.insert(0, new_p[1])
                    if (p_q[j][-1][2] - p_q[j][0][2]) >= 1.:
                        p = p_q[j].popleft()

                x_p = np.array(x_p)
                y_p = np.array(y_p)

                k = 2 if x_p.size > 2 else 1
                if len(x_p) > 1:
                    # extrapolation_amount = EXTRAPOLATION_FACTOR * math.sqrt(
                        # (x_p[-1] - x_p[-2]) ** 2 + (y_p[-1] - y_p[-2]) ** 2)
                    extrapolation_amount = TOTAL_DIVISIONS/(len(x_p)-1)
                    tck_p = splprep([x_p, y_p], k=k, s=0)
                    ex_p, ey_p = splev(np.linspace(0, extrapolation_amount, TOTAL_DIVISIONS), tck_p[0][0:3], der=0)
                else:
                    ex_p = [x_p[0]]*TOTAL_DIVISIONS
                    ey_p = [y_p[0]]*TOTAL_DIVISIONS

                entity_splines.append([ex_p, ey_p, 'p'])
                entity_coords.append([x_p, y_p])

                collision = False
                for t in range(1, TOTAL_DIVISIONS):
                    point1 = Point(ex_p[t], ey_p[t])
                    point2 = Point(ex_r[t], ey_r[t])

                    dist1 = math.sqrt((ex_p[t] - ex_p[t-1]) ** 2 + (ey_p[t] - ey_p[t-1]) ** 2)
                    dist2 = math.sqrt((ex_r[t] - ex_r[t-1]) ** 2 + (ey_r[t] - ey_r[t-1]) ** 2)

                    if dist1 > 2.5 * ENTITY_RADIUS:
                        est1 = LineString([point1, Point(ex_p[t-1], ey_p[t-1])])
                    else:
                        est1 = point1.buffer(ENTITY_RADIUS)

                    if dist2 > 2.5 * ENTITY_RADIUS:
                        est2 = LineString([point2, Point(ex_r[t-1], ey_r[t-1])])
                    else:
                        est2 = point2.buffer(ENTITY_RADIUS)

                    if est1.intersects(est2):
                        collision = True

                    if collision:
                        collision_coords.append([ex_r[t], ey_r[t]])
                        break

                if t+1 < TOTAL_DIVISIONS:
                    datastore[i]['people'][j]['t_collision'] = (t+1) * time_inc
                    datastore_absolute[i]['people'][j]['t_collision'] = (t+1) * time_inc
                else:
                    datastore[i]['people'][j]['t_collision'] = math.inf
                    datastore_absolute[i]['people'][j]['t_collision'] = math.inf

                # print(datastore[i]['people'][j]['t_collision'])
                # if collision:
                #     plt.plot(ex_p, ey_p, 'o', x_p, y_p, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o',  ex_r[t], ey_r[t], 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2', 'collision'])
                # else:
                #     plt.plot(ex_p, ey_p, 'o', x_p, y_p, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2'])
                # plt.title("Figure " + str(i))
                # plt.axis([x_r.min() - 5, x_r.max() + 5, y_r.min() - 5, y_r.max() + 5])
                # plt.show()

                # if i == 15:
                #     sys.exit(0)

        for j, o in enumerate(datastore_absolute[i]['objects']):
            if i == 0:
                o['vx'] = 0.0
                o['vy'] = 0.0
                o['va'] = 0.0
            # elif j == len(datastore_absolute[i]['objects']):
            #     break
            else:
                o['vx'] = datastore_absolute[i]['objects'][j]['x'] - datastore_absolute[i-1]['objects'][j]['x']
                o['vy'] = datastore_absolute[i]['objects'][j]['y'] - datastore_absolute[i-1]['objects'][j]['y']
                o['va'] = datastore_absolute[i]['objects'][j]['a'] - datastore_absolute[i-1]['objects'][j]['a']

            # Calculate time to collision
            if i == 0:
                datastore[i]['objects'][j]['t_collision'] = math.inf
                datastore_absolute[i]['objects'][j]['t_collision'] = math.inf
            else:
                point1 = Point(o['x'], o['y'])

                entity_splines.append([o['x'], o['y'], 'o'])
                entity_coords.append([o['x'], o['y']])

                collision = False
                for t in range(1, TOTAL_DIVISIONS):
                    point2 = Point(ex_r[t], ey_r[t])
                    dist = math.sqrt((ex_r[t] - ex_r[t - 1]) ** 2 + (ey_r[t] - ey_r[t - 1]) ** 2)

                    if dist > 2.5 * ENTITY_RADIUS:
                        circle = point1.buffer(ENTITY_RADIUS).boundary
                        line = LineString([point2, Point(ex_r[t - 1], ey_r[t - 1])])
                        if circle.intersection(line):
                            collision = True
                    else:
                        circle1 = point1.buffer(ENTITY_RADIUS)
                        circle2 = point2.buffer(ENTITY_RADIUS)
                        if circle1.intersects(circle2):
                            collision = True

                    if collision:
                        collision_coords.append([ex_r[t], ey_r[t]])
                        break

                if t+1 < TOTAL_DIVISIONS:
                    datastore[i]['objects'][j]['t_collision'] = (t + 1) * time_inc
                    datastore_absolute[i]['objects'][j]['t_collision'] = (t + 1) * time_inc
                else:
                    datastore[i]['objects'][j]['t_collision'] = math.inf
                    datastore_absolute[i]['objects'][j]['t_collision'] = math.inf

                # print(datastore[i]['objects'][j]['t_collision'])
                # if collision:
                #     plt.plot(ex_o, ey_o, 'o', x_o, y_o, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o',  ex_r[t], ey_r[t], 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2', 'collision'])
                # else:
                #     plt.plot(ex_o, ey_o, 'o', x_o, y_o, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2'])
                # plt.title("Figure " + str(i))
                # plt.axis([x_r.min() - 5, x_r.max() + 5, y_r.min() - 5, y_r.max() + 5])
                # plt.show()
                #
                # if i == 15:
                #     sys.exit(0)

        for j, w in enumerate(datastore_absolute[i]['walls']):
            if i == 0:
                datastore[i]['walls'][j]['t_collision'] = math.inf
                datastore_absolute[i]['walls'][j]['t_collision'] = math.inf
            else:

                lineSegment = LineString([Point(w['x1'], w['y1']), Point(w['x2'], w['y2'])])
                lineSegmentLeft = lineSegment.parallel_offset(0.15, 'left')
                lineSegmentRight = lineSegment.parallel_offset(0.15, 'right')

                collision = False
                for t in range(1, len(ex_r)):
                    point1 = Point(ex_r[t], ey_r[t])
                    dist = math.sqrt((ex_r[t] - ex_r[t - 1]) ** 2 + (ey_r[t] - ey_r[t - 1]) ** 2)

                    if dist > 2.5 * ENTITY_RADIUS:
                        line = LineString([point1, Point(ex_r[t - 1], ey_r[t - 1])])
                        if line.intersects(lineSegment) or line.intersects(lineSegmentLeft) or\
                                line.intersects(lineSegmentRight):
                            collision = True
                    else:
                        circle1 = point1.buffer(ENTITY_RADIUS)
                        if circle1.intersects(lineSegment) or circle1.intersects(lineSegmentLeft) or\
                                circle1.intersects(lineSegmentRight):
                            collision = True

                    # for idx in range(len(ex_w)):
                    #     point2 = Point(ex_w[idx], ey_w[idx])
                    #     circle2 = point2.buffer(entity_radius)

                    #     if circle1.intersects(circle2):
                    #         collision = True
                    #         collision_coords.append([ex_r[t], ey_r[t]])
                    #         break

                    if collision:
                        collision_coords.append([ex_r[t], ey_r[t]])
                        break

                if t+1 < TOTAL_DIVISIONS:
                    datastore[i]['walls'][j]['t_collision'] = (t + 1) * time_inc
                    datastore_absolute[i]['walls'][j]['t_collision'] = (t + 1) * time_inc
                else:
                    datastore[i]['walls'][j]['t_collision'] = math.inf
                    datastore_absolute[i]['walls'][j]['t_collision'] = math.inf

                # print(datastore[i]['walls'][j]['t_collision'])
                # if collision:
                #     plt.plot(ex_w, ey_w, 'o', x_w, y_w, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o',  ex_r[t], ey_r[t], 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2', 'collision'])
                # else:
                #     plt.plot(ex_w, ey_w, 'o', x_w, y_w, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2'])
                # plt.title("Figure " + str(i))
                # plt.axis([x_w.min() - 5, x_w.max() + 5, y_w.min() - 5, y_w.max() + 5])
                # plt.show()
                #
                # if i == 15:
                #     sys.exit(0)

        if i == 0:
            datastore[i]['goal'][0]['t_collision'] = math.inf
            datastore_absolute[i]['goal'][0]['t_collision'] = math.inf
        else:
            point1 = Point(datastore_absolute[i]['goal'][0]['x'], datastore_absolute[i]['goal'][0]['y'])
            entity_splines.append([datastore_absolute[i]['goal'][0]['x'], datastore_absolute[i]['goal'][0]['y'], 't'])
            entity_coords.append([datastore_absolute[i]['goal'][0]['x'], datastore_absolute[i]['goal'][0]['y']])

            collision = False
            for t in range(TOTAL_DIVISIONS):
                point2 = Point(ex_r[t], ey_r[t])
                dist = math.sqrt((ex_r[t] - ex_r[t - 1]) ** 2 + (ey_r[t] - ey_r[t - 1]) ** 2)

                if dist > 2.5 * ENTITY_RADIUS:
                    circle = point1.buffer(ENTITY_RADIUS).boundary
                    line = LineString([point2, Point(ex_r[t - 1], ey_r[t - 1])])
                    if circle.intersection(line):
                        collision = True
                else:
                    circle1 = point1.buffer(ENTITY_RADIUS)
                    circle2 = point2.buffer(ENTITY_RADIUS)
                    if circle1.intersects(circle2):
                        collision = True

                if collision:
                    collision_coords.append([ex_r[t], ey_r[t]])
                    break

            if t+1 < TOTAL_DIVISIONS:
                datastore[i]['goal'][0]['t_collision'] = (t + 1) * time_inc
                datastore_absolute[i]['goal'][0]['t_collision'] = (t + 1) * time_inc
            else:
                datastore[i]['goal'][0]['t_collision'] = math.inf
                datastore_absolute[i]['goal'][0]['t_collision'] = math.inf


        # Plot the whole scenario
        # colour = {'p': 'bo', 'o': 'go', 'r': 'co', 't': 'ro', 'w': 'mo'}
        # for e in entity_splines:
        #     plt.plot(e[0], e[1], colour[e[2]])
        # for e in entity_coords:
        #     plt.plot(e[0], e[1], 'yd')
        # for c in collision_coords:
        #     plt.plot(c[0], c[1], 'ko')
        #
        # plt.title("Frame " + str(i))
        # max_x = max_y = min_x = min_y = 0.
        # for w in datastore_absolute[i]['walls']:
        #     plt.plot([w['x1'], w['x2']], [w['y1'], w['y2']], 'm')
        #     max_x = max(w['x1'], w['x2']) if max(w['x1'], w['x2']) > max_x else max_x
        #     max_y = max(w['y1'], w['y2']) if max(w['y1'], w['y2']) > max_y else max_y
        #     min_x = min(w['x1'], w['x2']) if min(w['x1'], w['x2']) < min_x else min_x
        #     min_y = min(w['y1'], w['y2']) if min(w['y1'], w['y2']) < min_x else min_x
        #
        # plt.axis([min_x - 1, max_x + 1, min_y - 1, max_y + 1])
        # plt.show()

    datastore = list(reversed(datastore))
    datastore_absolute = list(reversed(datastore_absolute))

    with open(dest_directory + '/' + save, 'w') as outfile:
        json.dump(datastore, outfile, indent=4, sort_keys=True)
        outfile.close()

    with open(dest_directory + '_absolute' + '/' + save, 'w') as outfile:
        json.dump(datastore_absolute, outfile, indent=4, sort_keys=True)
        outfile.close()

    # except BaseException as err:
    #     with open('jsons_problems.txt', 'a') as f:
    #         f.write(filename + "\n")
    #         f.write(f"Unexpected {err}, {type(err)}" + "\n" + "\n")

    #     if type(err) == KeyboardInterrupt:
    #         exit()
    #     else:
    #         continue
