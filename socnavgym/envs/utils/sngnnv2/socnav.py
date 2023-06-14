import os
import sys
import json
import copy
from collections import namedtuple
import math

import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import numpy as np

grid_width = 18  # 30 #18
output_width = 73  # 121 #73
area_width = 800.  # Spatial area of the grid

threshold_human_wall = 1.5
limit = 2000  # Limit of graphs to load
path_saves = 'saves/'  # This variable is necessary due tu a bug in dgl.DGLDataset source code
graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_feats', 'edge_types',
                                     'edge_norms', 'position_by_id', 'typeMap', 'labels', 'w_segments'])


#  human to wall distance
def dist_h_w(h, wall):
    if 'xPos' in h.keys():
        hxpos = float(h['xPos']) / 100.
        hypos = float(h['yPos']) / 100.
    else:
        hxpos = float(h['x'])
        hypos = float(h['y'])

    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))


# Calculate the closet node in the grid to a given node by its coordinates
def closest_grid_node(grid_ids, w_a, w_i, x, y):
    c_x = int((x * (w_i / w_a) + (w_i / 2)))
    c_y = int((y * (w_i / w_a) + (w_i / 2)))
    if 0 <= c_x < grid_width and 0 <= c_y < grid_width:
        return grid_ids[c_x][c_y]
    return None


def closest_grid_nodes(grid_ids, w_a, w_i, r, x, y):
    c_x = int((x * (w_i / w_a) + (w_i / 2)))
    c_y = int((y * (w_i / w_a) + (w_i / 2)))
    cols, rows = (int(math.ceil(r * w_i / w_a)), int(math.ceil(r * w_i / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c, r] for c in rangeC for r in rangeR]
    grid_nodes = []
    r_g = r * w_i / w_a
    for p in p_arr:
        if math.sqrt(p[0] * p[0] + p[1] * p[1]) <= r_g:
            if 0 <= (c_x + p[0]) < grid_width and 0 <= (c_y + p[1]) < grid_width:
                grid_nodes.append(grid_ids[c_x + p[0]][c_y + p[1]])

    return grid_nodes


def get_node_descriptor_header():
    # Node Descriptor Table
    node_descriptor_header = ['R', 'H', 'O', 'L', 'W',
                              'h_dist', 'h_dist2', 'h_ang_sin', 'h_ang_cos', 'h_orient_sin', 'h_orient_cos',
                              'o_dist', 'o_dist2', 'o_ang_sin', 'o_ang_cos', 'o_orient_sin', 'o_orient_cos',
                              'r_m_h', 'r_m_h2', 'r_hs', 'r_hs2',
                              'w_dist', 'w_dist2', 'w_ang_sin', 'w_ang_cos', 'w_orient_sin', 'w_orient_cos']
    return node_descriptor_header


def get_relations():
    rels = {'p_r', 'o_r', 'p_p', 'p_o', 'w_r', 'g_r', 'w_p'}  # add 'w_w' for links between wall nodes
    # p = person
    # r = room
    # o = object
    # w = wall
    # g = goal
    for e in list(rels):
        rels.add(e[::-1])
    rels.add('self')
    rels = sorted(list(rels))
    num_rels = len(rels)

    return rels, num_rels


def get_features():
    time_one_hot = ['is_t_0', 'is_t_m1', 'is_t_m2']
    # time_sequence_features = ['is_first_frame', 'time_left']
    human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                             'hum_orientation_sin', 'hum_orientation_cos',
                             'hum_dist', 'hum_inv_dist']
    object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                              'obj_orientation_sin', 'obj_orientation_cos',
                              'obj_x_size', 'obj_y_size',
                              'obj_dist', 'obj_inv_dist']
    room_metric_features = ['room_humans', 'room_humans2']
    robot_features = ['robot_adv_vel', 'robot_rot_vel']
    wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                            'wall_dist', 'wall_inv_dist']
    goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
    grid_metric_features = ['grid_x_pos', 'grid_y_pos']
    node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal']
    all_features = node_types_one_hot + time_one_hot + human_metric_features + robot_features + \
                   object_metric_features + room_metric_features + wall_metric_features + goal_metric_features

    feature_dimensions = len(all_features)

    return all_features, feature_dimensions


#################################################################
# Different initialize alternatives:
#################################################################
N_INTERVALS = 3
FRAMES_INTERVAL = 1.

MAX_ADV = 3.5
MAX_ROT = 4.
MAX_HUMANS = 15


def initializeAlt1(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations()
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    i_w = 0
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        if i_w >= len(w_segments):
            iters = int(dist / 97) + 1
            w_segments.append(iters)
        if w_segments[i_w] > 1:  # WE NEED TO CHECK THIS PART
            v = (p2 - p1) / w_segments[i_w]
            for i in range(w_segments[i_w]):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
        i_w += 1

    # Compute the number of nodes
    # one for the robot + room walls   + humans    + objects              + room(global node)
    n_nodes = 1 + len(walls) + len(data['people']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features()
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    if 'label_Q1' in data.keys():
        labels = np.array([float(data['label_Q1']), float(data['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['people']) / MAX_HUMANS
    features[room_id, all_features.index('room_humans2')] = (len(data['people']) ** 2) / (MAX_HUMANS ** 2)
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # humans
    for h in data['people']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id += 1
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        orientation = h['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = xpos
        features[h['id'], all_features.index('hum_y_pos')] = ypos
        features[h['id'], all_features.index('human_a_vel')] = va
        features[h['id'], all_features.index('human_x_vel')] = vx
        features[h['id'], all_features.index('human_y_vel')] = vy
        features[h['id'], all_features.index('hum_dist')] = dist
        features[h['id'], all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('p_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_p')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('o_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_o')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 'g'  # 'g' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('g_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_g'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('g_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_g')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # walls
    wids = dict()
    for w_i, wall in enumerate(walls, 0):
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'
        max_used_id += 1

        dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

        # Links to room node
        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_r'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('r_w'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('w_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_w')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]

        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = wall.ypos / 1000.
        features[wall_id, all_features.index('wall_dist')] = dist
        features[wall_id, all_features.index('wall_inv_dist')] = 1. - dist  # 1./(1.+dist*10.)

    for h in data['people']:
        number = 0
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(wids[wall])
                dst_nodes.append(h['id'])
                edge_types.append(rels.index('w_p'))
                edge_norms.append([1. / number])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('w_p')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

    for wall in walls:
        number = 0
        for h in data['people']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for h in data['people']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(h['id'])
                dst_nodes.append(wids[wall])
                edge_types.append(rels.index('p_w'))
                edge_norms.append([1. / number])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('p_w')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
                         (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLdir)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLinv)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, w_segments


#################################################################
# Class to load the dataset
#################################################################

class SocNavDataset(DGLDataset):
    def __init__(self, path, alt, mode='train', raw_dir='data/', init_line=-1, end_line=-1, loc_limit=limit,
                 force_reload=False, verbose=True, debug=False):
        if type(path) is str:
            self.path = raw_dir + "/" + path
        else:
            self.path = path

        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.graphs = []
        self.labels = []
        self.data = dict()
        self.grid_data = None
        self.data['typemaps'] = []
        self.data['coordinates'] = []
        self.data['identifiers'] = []
        self.debug = debug
        self.limit = loc_limit
        self.force_reload = force_reload

        if self.mode == 'test':
            self.force_reload = True

        # Define device. GPU if it is available
        self.device = 'cpu'

        if self.debug:
            self.limit = 1 + (0 if init_line == -1 else init_line)

        if self.alt == '1':
            self.dataloader = initializeAlt1
        else:
            print('Introduce a valid initialize alternative')
            sys.exit(-1)

        super(SocNavDataset, self).__init__("SocNav", raw_dir=raw_dir, force_reload=self.force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = 'graphs_' + self.mode + '_alt_' + self.alt + '_s_' + str(limit) + '.bin'
        info_path = 'info_' + self.mode + '_alt_' + self.alt + '_s_' + str(limit) + '.pkl'
        return graphs_path, info_path

    def generate_final_graph(self, raw_data):
        rels, num_rels = get_relations()
        room_graph_data = graphData(*self.dataloader(raw_data))
        if self.grid_data is not None:
            # Merge room and grid graph
            src_nodes = th.cat([self.grid_data.src_nodes, (room_graph_data.src_nodes + self.grid_data.n_nodes)], dim=0)
            dst_nodes = th.cat([self.grid_data.dst_nodes, (room_graph_data.dst_nodes + self.grid_data.n_nodes)], dim=0)
            edge_types = th.cat([self.grid_data.edge_types, room_graph_data.edge_types], dim=0)
            edge_norms = th.cat([self.grid_data.edge_norms, room_graph_data.edge_norms], dim=0)

            # Link each node in the room graph to the correspondent grid graph.
            for r_n_id in range(1, room_graph_data.n_nodes):
                r_n_type = room_graph_data.typeMap[r_n_id]
                x, y = room_graph_data.position_by_id[r_n_id]
                closest_grid_nodes_id = closest_grid_nodes(self.grid_data.labels, area_width, grid_width, 25., x * 1000,
                                                           y * 1000)
                for g_id in closest_grid_nodes_id:
                    src_nodes = th.cat([src_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                    dst_nodes = th.cat([dst_nodes, th.tensor([r_n_id + self.grid_data.n_nodes], dtype=th.int32)], dim=0)
                    edge_types = th.cat([edge_types, th.LongTensor([rels.index('g_' + r_n_type)])], dim=0)
                    edge_norms = th.cat([edge_norms, th.Tensor([[1.]])])

                    src_nodes = th.cat([src_nodes, th.tensor([r_n_id + self.grid_data.n_nodes], dtype=th.int32)], dim=0)
                    dst_nodes = th.cat([dst_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                    edge_types = th.cat([edge_types, th.LongTensor([rels.index(r_n_type + '_g')])], dim=0)
                    edge_norms = th.cat([edge_norms, th.Tensor([[1.]])])

            # Compute typemaps, coordinates, number of nodes, features and labels for the merged graph.
            n_nodes = room_graph_data.n_nodes + self.grid_data.n_nodes
            typeMapRoomShift = dict()
            coordinates_roomShift = dict()

            for key in room_graph_data.typeMap:
                typeMapRoomShift[key + len(self.grid_data.typeMap)] = room_graph_data.typeMap[key]
                coordinates_roomShift[key + len(self.grid_data.position_by_id)] = room_graph_data.position_by_id[key]

            position_by_id = {**self.grid_data.position_by_id, **coordinates_roomShift}
            typeMap = {**self.grid_data.position_by_id, **typeMapRoomShift}
            labels = room_graph_data.labels
            features = th.cat([self.grid_data.features, room_graph_data.features], dim=0)
        else:
            src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, \
            position_by_id, typeMap, labels, wall_segments = room_graph_data

        self.data['typemaps'].append(typeMap)
        self.data['coordinates'].append(position_by_id)
        self.data['identifiers'].append(raw_data['identifier'])
        self.data['descriptor_header'] = get_node_descriptor_header()

        self.labels.append(labels)

        try:
            final_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=n_nodes, idtype=th.int32, device=self.device)
            final_graph.ndata['h'] = features.to(self.device)
            edge_types.to(self.device, dtype=th.long)
            edge_norms.to(self.device, dtype=th.float64)
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})
            return final_graph
        except Exception:
            raise

    def load_one_graph(self, data):
        graph_data = graphData(*self.dataloader(data[0]))
        w_segments = graph_data.w_segments

        graphs_in_interval = [graph_data]
        frames_in_interval = [data[0]]
        for frame in data[1:]:
            if math.fabs(frame['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
                continue
            graphs_in_interval.append(graphData(*self.dataloader(frame, w_segments)))
            frames_in_interval.append(frame)
            if len(graphs_in_interval) == N_INTERVALS:
                break

        src_nodes, dst_nodes, edge_types, edge_norms, n_nodes, feats, edge_feats, typeMap, coordinates = \
            self.merge_graphs(graphs_in_interval)

        try:
            # Create merged graph:
            final_graph = dgl.graph((src_nodes, dst_nodes),
                                    num_nodes=n_nodes,
                                    idtype=th.int32, device=self.device)

            # Add merged features and update edge labels:
            final_graph.ndata['h'] = feats.to(self.device)
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms, 'he': edge_feats})

            # Append final data
            self.graphs.append(final_graph)
            self.labels.append(graphs_in_interval[0].labels)
            self.data['typemaps'].append(typeMap)
            self.data['coordinates'].append(coordinates)
            self.data['identifiers'].append(data[0]['ID'])
            self.data['descriptor_header'] = get_node_descriptor_header()

        except Exception:
            print("Error loading one graph")
            raise

    def merge_graphs(self, graphs_in_interval):
        all_features, n_features = get_features()
        new_features = ['is_t_0', 'is_t_m1', 'is_t_m2']
        f_list = []
        src_list = []
        dst_list = []
        edge_types_list = []
        edge_norms_list = []
        edge_feats_list = []
        typeMap = dict()
        coordinates = dict()
        n_nodes = 0
        rels, num_rels = get_relations()
        g_i = 0
        offset = graphs_in_interval[0].n_nodes
        for g in graphs_in_interval:
            # Shift IDs of the typemap and coordinates lists
            for key in g.typeMap:
                typeMap[key + (offset * g_i)] = g.typeMap[key]
                coordinates[key + (offset * g_i)] = g.position_by_id[key]
            n_nodes += g.n_nodes
            f_list.append(g.features)
            edge_feats_list.append(g.edge_feats)
            # Add temporal edges
            src_list.append(g.src_nodes + (offset * g_i))
            dst_list.append(g.dst_nodes + (offset * g_i))
            edge_types_list.append(g.edge_types)
            edge_norms_list.append(g.edge_norms)
            if g_i > 0:
                # Temporal connections and edges labels
                new_src_list = []
                new_dst_list = []
                new_etypes_list = []
                new_enorms_list = []
                new_edge_feats_list = []
                for node in range(g.n_nodes):
                    new_src_list.append(node + offset * (g_i - 1))
                    new_dst_list.append(node + offset * g_i)
                    new_etypes_list.append(num_rels + (g_i - 1) * 2)
                    new_enorms_list.append([1.])

                    new_src_list.append(node + offset * g_i)
                    new_dst_list.append(node + offset * (g_i - 1))
                    new_etypes_list.append(num_rels + (g_i - 1) * 2 + 1)
                    new_enorms_list.append([1.])

                    # Edge features
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[num_rels + (g_i - 1) * 2] = 1
                    edge_features[-1] = 0
                    new_edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[num_rels + (g_i - 1) * 2 + 1] = 1
                    edge_features[-1] = 0
                    new_edge_feats_list.append(edge_features)

                new_edge_feats = th.stack(new_edge_feats_list)

                src_list.append(th.IntTensor(new_src_list))
                dst_list.append(th.IntTensor(new_dst_list))
                edge_types_list.append(th.LongTensor(new_etypes_list))
                edge_norms_list.append(th.Tensor(new_enorms_list))
                edge_feats_list.append(new_edge_feats)
            for f in new_features:
                if g_i == new_features.index(f):
                    g.features[:, all_features.index(f)] = 1
                else:
                    g.features[:, all_features.index(f)] = 0
            g_i += 1

        src_nodes = th.cat(src_list, dim=0)
        dst_nodes = th.cat(dst_list, dim=0)
        edge_types = th.cat(edge_types_list, dim=0)
        edge_norms = th.cat(edge_norms_list, dim=0)
        edge_feats = th.cat(edge_feats_list, dim=0)
        feats = th.cat(f_list, dim=0)

        return src_nodes, dst_nodes, edge_types, edge_norms, n_nodes, feats, edge_feats, typeMap, coordinates

    #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):
        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            for line in open(self.path).readlines():
                if linen % 1000 == 0:
                    print(linen)

                if linen + 1 >= self.limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1
                if self.init_line >= 0 and linen < self.init_line:
                    continue
                if linen > self.end_line >= 0:
                    continue

                raw_data = json.loads(line)
                final_graph = self.generate_final_graph(raw_data)
                self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)

        elif type(self.path) is str and self.path.endswith('.txt'):
            linen = -1
            print(self.path)
            with open(self.path) as set_file:
                ds_files = set_file.read().splitlines()

            print("number of files for ", self.path, len(ds_files))

            for ds in ds_files:
                if linen % 1000 == 0:
                    print(linen)
                with open(ds) as json_file:
                    data = json.load(json_file)
                self.load_one_graph(data)
                if linen + 1 >= limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1
            self.labels = th.tensor(self.labels, dtype=th.float64)

        elif type(self.path) == list and len(self.path) >= 1:
            self.load_one_graph(self.path)
            self.labels = th.tensor(np.array(self.labels), dtype=th.float64)
        elif type(self.path) == list and type(self.path[0]) == str:
            raw_data = json.loads(self.path)
            final_graph = self.generate_final_graph(raw_data)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        else:
            final_graph = self.generate_final_graph(self.path)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug:
            return
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        save_info(info_path, {'typemaps': self.data['typemaps'],
                              'coordinates': self.data['coordinates'],
                              'identifiers': self.data['identifiers'],
                              'descriptor_header': self.data['descriptor_header']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['typemaps'] = load_info(info_path)['typemaps']
        self.data['coordinates'] = load_info(info_path)['coordinates']
        self.data['descriptor_header'] = load_info(info_path)['descriptor_header']
        self.data['identifiers'] = load_info(info_path)['identifiers']

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
