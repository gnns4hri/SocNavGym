import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import math
from pathlib import Path
from torch_geometric.data import Dataset, Data, Batch
from typing import List
from tqdm import tqdm

from .data_conversions import clone_sequence, sequence_to_tensor
from .data_normalization import tensor_transform_to_goal_fr
from . import metrics



class SocNavHomoDataset(Dataset):
    def __init__(self, data_input, data_path=None, context_path=None, overwrite_contexts=None, transform=None, pre_transform=None, pre_filter=None, 
                 timestamp_threshold=0.1, reload=False, data_augmentation=False, label_weights = None, W_GROUPS = 20):
        self.data_input = data_input
        self.data_path = data_path
        self.MAX_TTC = 10
        self.metrics_features = ['dist_to_goal_pos', 'dist_to_goal_angle', 'success', 'hum_exists', 'wall_exist', 'dist_nearest_hum', 'dist_nearest_object', 
                                 'dist_wall', 'human_collision_flag', 'object_collision_flag', 'wall_collision_flag',
                                 'social_space_intrusionA', 'num_near_humansA', 'num_near_humansA2', 'social_space_intrusionB',
                                 'num_near_humansB', 'num_near_humansB2', 'social_space_instrusionC', 'num_near_humansC',
                                 'num_near_humansC2', 'min_ttc', 'min_ttc2', 'max_fear', 'max_panic', 'global_dist_nearest_hum',
                                 'path_efficiency_ratio', 'step_ratio', 'episode_end']    
        self.max_values = {'scale': 10.0, 'max_v': 10.0, 'max_va': 2*np.pi, 'max_acc' : 3.0, 'max_c': 100.0, 'dist_to_goal_pos': 10, 
                           'dist_to_goal_angle': np.pi, 'success': 1, 'hum_exists': 1, 'wall_exist': 1, 'dist_nearest_hum': 10, 'dist_nearest_object': 10, 
                                 'dist_wall': 10, 'human_collision_flag': 1, 'object_collision_flag': 1, 'wall_collision_flag': 1,
                                 'social_space_intrusionA': 1, 'num_near_humansA': 10, 'num_near_humansA2': 10, 'social_space_intrusionB': 1,
                                 'num_near_humansB': 10, 'num_near_humansB2': 10, 'social_space_intrusionC': 1, 'num_near_humansC': 10,
                                 'num_near_humansC2': 10, 'min_ttc': self.MAX_TTC, 'min_ttc2': self.MAX_TTC**2, 'max_fear': 10, 'max_panic': 10, 'global_dist_nearest_hum': 10,
                                 'path_efficiency_ratio': 1, 'step_ratio': 1, 'episode_end': 1}

        self.graphs = []
        self.metrics = []        
        self.labels = []
        self.label_range = []
        self.slengths = []

        self.W_GROUPS = W_GROUPS
        self.set_label_weights = label_weights
        self.label_weights = [0]*self.W_GROUPS


        self.context_df = pd.read_csv(context_path, index_col='context')
        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        self.overwrite_contexts = overwrite_contexts
        self.data_augmentation = data_augmentation
        self.all_features = self.set_all_features()
        self.timestamp_threshold = timestamp_threshold


        self.init_data(data_input, data_path, context_path, overwrite_contexts, transform, pre_transform, pre_filter, timestamp_threshold, reload, data_augmentation)

        self.process()
        # self.process_structure(data_input)


    @property
    def has_process(self):
        return False


    @property
    def has_download(self):
        return False
 

    @property
    def raw_file_names(self):
        return None

 
    @property
    def processed_file_names(self):
        return []
    

    def init_data(self, data_list_file, data_path, context_path, overwrite_contexts, transform, pre_transform, pre_filter, timestamp_threshold, reload, data_augmentation):
        super(SocNavHomoDataset, self).__init__(None, transform, pre_transform, pre_filter)


    @property
    def raw_paths(self) -> List[str]:
        return []

    @property
    def processed_dir(self):
        return self.root

    def process(self):
        if isinstance(self.data_input, str):
            with open(os.path.join(self.data_path, self.data_input), 'r') as f:
                self.file_names = f.read().splitlines()
        elif isinstance(self.data_input, list) and isinstance(self.data_input[0], str):
            self.file_names = self.data_input
        else:
            self.file_names = []
            self.process_structure(self.data_input)

        ndata = 0
        for raw_file in tqdm(self.file_names, total=len(self.file_names), disable=len(self.file_names) == 0):   # SocNavGym: no progress bar for a trajectory given in memory
            with open(os.path.join(self.data_path, raw_file), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                self.process_structure(json_data)
            ndata+=1
            # if ndata>50:
            #     break

        # print('distribution', self.label_weights)
        if self.set_label_weights is not None:
            self.label_weights = self.set_label_weights
        else:
            self.label_weights = torch.tensor(self.label_weights, dtype = torch.float32)
            proportion = self.label_weights/ndata
            valid_ranges = (proportion>0).sum()
            self.label_weights[proportion>0] = 1./(proportion[proportion>0]*valid_ranges)
            self.label_weights = self.label_weights/torch.sum(self.label_weights)

        # print('weights', self.label_weights)
       

    def process_structure(self, json_data):
        if 'context_description' in json_data.keys() and self.overwrite_contexts is None:
            context_desc = json_data['context_description']
        elif self.overwrite_contexts is not None:
            context_desc = self.overwrite_contexts
        else:
            sys.exit(12)

        context = self.context_df.loc[context_desc.rstrip()].to_dict()

        walls = json_data.get('walls', [])
        rating = json_data.get('label', 0.)
        lenght = 0

        tensor_dict, lenght = sequence_to_tensor(json_data, self.timestamp_threshold, context)

        tensor_dict_to_goal = tensor_transform_to_goal_fr(tensor_dict)

        self.full_dict = metrics.compute_metrics(tensor_dict_to_goal)

        normalized_dict = metrics.normalize_features(self.full_dict, self.max_values)

        trayectoria = []
        traj_metrics = []
        for i in range(lenght):
            grafo, frame_metrics = self._structure_to_graph(normalized_dict, i)
            trayectoria.append(grafo)
            traj_metrics.append(frame_metrics)

        traj_metrics = torch.stack(traj_metrics)
        self.graphs.append(trayectoria)
        self.metrics.append(traj_metrics)
        self.labels.append(torch.tensor([rating], dtype=torch.float32))
        self.slengths.append(torch.tensor(lenght, dtype=torch.long))

        l_range = min(int(rating*100)//self.W_GROUPS, self.W_GROUPS-1)
        self.label_range.append(l_range)
        self.label_weights[l_range] += 1

        if self.data_augmentation:
            cloned_trajectory = clone_sequence(trayectoria)
            t_data_mirrored = self.mirror_sequence(cloned_trajectory)
            self.graphs.append(t_data_mirrored)
            self.metrics.append(traj_metrics)
            self.labels.append(torch.tensor([rating], dtype=torch.float32))
            self.slengths.append(torch.tensor(lenght, dtype=torch.long))
            self.label_range.append(l_range)


    def get_all_features(self):
        return self.all_features

    def get_context_features(self):
        return self.context_features

    def get_metrics_features(self):
        return self.metrics_features

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        metrics = self.metrics[idx]
        label = self.labels[idx] 
        slength = self.slengths[idx] 
        weight = self.label_weights[self.label_range[idx]]

        return data, metrics, label, slength, weight


    
    def set_all_features(self):
        self.type_features = ['robot', 'goal', 'human', 'object', 'wall']
        self.robot_features = ['r_x', 'r_y','r_sin_a', 'r_cos_a', 'dlin_gr', 'dang_gr', 'r_vx', 'r_vy', 
                               'r_va', 'r_acc_x', 'r_acc_y','r_w', 'r_l']
        self.goal_features = ['g_x', 'g_y','g_sin_a', 'g_cos_a', 
                              'th_pos', 'th_angle']

        self.human_features = ['h_x', 'h_y','h_sin_a', 'h_cos_a','dist_hr', 'dcenter_hr']

        self.object_features = ['o_x', 'o_y','o_sin_a', 'o_cos_a',
                                'o_w', 'o_l', 'dist_or', 'dcenter_or']

        self.wall_features = ['w_x', 'w_y','w_sin_a', 'w_cos_a',
                              'w_w', 'w_l', 'dist_wr']


        self.all_features = self.type_features + self.robot_features + self.goal_features + self.human_features + self.object_features + self.wall_features

        return self.all_features


    def _structure_to_graph(self, dict, index, full_conexo=True):
        num_node_features = len(self.all_features)
        # x = torch.zeros((num_nodes, num_node_features), dtype=torch.float32)

        entity_idx = {}
        graph_feats = []

        id = 0
        # robot node
        entity_idx['robot'] = [id]
        id += 1
        node_feats = torch.zeros(num_node_features, dtype=torch.float32)
        rx, ry = dict['robot']['x'][index], dict['robot']['y'][index]
        node_feats[self.all_features.index('robot')] = 1.
        node_feats[self.all_features.index('r_x')] = rx
        node_feats[self.all_features.index('r_y')] = ry
        node_feats[self.all_features.index('r_sin_a')] = math.sin(dict['robot']['a'][index])
        node_feats[self.all_features.index('r_cos_a')] = math.cos(dict['robot']['a'][index])
        # node_feats[self.all_features.index('vx')] = dict['robot']['vx'][index]
        # node_feats[self.all_features.index('vy')] = dict['robot']['vy'][index]
        # node_feats[self.all_features.index('va')] = dict['robot']['va'][index]        
        # node_feats[self.all_features.index('acc_x')] = dict['robot']['acc_x'][index]
        # node_feats[self.all_features.index('acc_y')] = dict['robot']['acc_y'][index]
        node_feats[self.all_features.index('r_w')] = dict['robot']['w'][index]
        node_feats[self.all_features.index('r_l')] = dict['robot']['l'][index]
        node_feats[self.all_features.index('dlin_gr')] = dict['computed_metrics']['dist_to_goal_pos'][index]
        node_feats[self.all_features.index('dang_gr')] = dict['computed_metrics']['dist_to_goal_angle'][index]

        graph_feats.append(node_feats)

        # goal node
        entity_idx['goal'] = [id]
        id += 1
        node_feats = torch.zeros(num_node_features, dtype=torch.float32)
        node_feats[self.all_features.index('goal')] = 1.
        node_feats[self.all_features.index('g_x')] = dict['goal']['x'][index]
        node_feats[self.all_features.index('g_y')] = dict['goal']['y'][index]
        node_feats[self.all_features.index('g_sin_a')] = math.sin(dict['goal']['a'][index])
        node_feats[self.all_features.index('g_cos_a')] = math.cos(dict['goal']['a'][index])
        node_feats[self.all_features.index('th_pos')] = dict['goal']['th_p'][index]
        node_feats[self.all_features.index('th_angle')] = dict['goal']['th_a'][index]

        graph_feats.append(node_feats)

        # human nodes
        num_humans = dict['people']['x'].shape[1]
        people_exist = dict['people']['exists']
        entity_idx['human'] = []
        for i in range(num_humans):
            if people_exist[index, i].item():
                entity_idx['human'].append(id)
                id += 1
                node_feats = torch.zeros(num_node_features, dtype=torch.float32)
                px = dict['people']['x'][index, i].item()
                py = dict['people']['y'][index, i].item()
                pa = dict['people']['a'][index, i].item()
                dist_to_robot = dict['metrics']['dist_human'][index, i].item()
                # if index>0 and dict['people']['id'][index][i] == dict['people']['id'][index-1][i]:
                #     diff_time = dict['timestamp'][index]-dict['timestamp'][index-1]
                #     vx = (px-dict['people']['x'][index-1, i].item())/diff_time
                #     vy = (py-dict['people']['y'][index-1, i].item())/diff_time
                # else:
                #     vx = 0.
                #     vy = 0.
                dcenter_to_robot = math.sqrt((px-rx)**2+(py-ry)**2)
                node_feats[self.all_features.index('human')] = 1.
                node_feats[self.all_features.index('h_x')] = px
                node_feats[self.all_features.index('h_y')] = py
                # node_feats[self.all_features.index('vx')] = vx
                # node_feats[self.all_features.index('vy')] = vy
                node_feats[self.all_features.index('h_sin_a')] = math.sin(pa)
                node_feats[self.all_features.index('h_cos_a')] = math.cos(pa)
                node_feats[self.all_features.index('dist_hr')] = dist_to_robot
                node_feats[self.all_features.index('dcenter_hr')] = dcenter_to_robot

                graph_feats.append(node_feats)

        # object nodes
        num_objects = dict['objects']['x'].shape[1]
        objects_exist = dict['objects']['exists']
        entity_idx['object'] = []        
        for i in range(num_objects):
            if objects_exist[index, i].item():
                entity_idx['object'].append(id)
                id += 1
                node_feats = torch.zeros(num_node_features, dtype=torch.float32)
                ox = dict['objects']['x'][index, i].item()
                oy = dict['objects']['y'][index, i].item()
                oa = dict['objects']['a'][index, i].item()
                dist_to_robot = dict['metrics']['dist_object'][index, i].item()
                dcenter_to_robot = math.sqrt((ox-rx)**2+(oy-ry)**2)
                node_feats[self.all_features.index('object')] = 1.
                node_feats[self.all_features.index('o_x')] = ox
                node_feats[self.all_features.index('o_y')] = oy
                node_feats[self.all_features.index('o_sin_a')] = math.sin(oa)
                node_feats[self.all_features.index('o_cos_a')] = math.cos(oa)
                node_feats[self.all_features.index('dist_or')] = dist_to_robot
                node_feats[self.all_features.index('o_w')] = dict['objects']['w'][index, i]
                node_feats[self.all_features.index('o_l')] = dict['objects']['l'][index, i]
                node_feats[self.all_features.index('dcenter_or')] = dcenter_to_robot

                graph_feats.append(node_feats)

        # wall nodes
        walls = dict['walls']
        entity_idx['wall'] = []
        if walls is not None:
            raw_points, id_walls, a_walls, l_walls = self._get_walls(walls)
            for pt, idW, wa, la in zip(raw_points, id_walls, a_walls, l_walls):
                entity_idx['wall'].append(id)
                id += 1
                node_feats = torch.zeros(num_node_features, dtype=torch.float32)
                wx = pt[0]
                wy = pt[1]
                dist_to_robot = dict['metrics']['dist_walls'][index, idW].item()
                # dist_to_robot = math.sqrt((wx-rx)**2+(wy-ry)**2)
                node_feats[self.all_features.index('wall')] = 1.
                node_feats[self.all_features.index('w_x')] = wx
                node_feats[self.all_features.index('w_y')] = wy
                node_feats[self.all_features.index('w_sin_a')] = math.sin(wa)
                node_feats[self.all_features.index('w_cos_a')] = math.cos(wa)
                node_feats[self.all_features.index('dist_wr')] = dist_to_robot
                node_feats[self.all_features.index('w_w')] = la
                node_feats[self.all_features.index('w_l')] = 0.01

                graph_feats.append(node_feats)


        context_values = []
        for c_key in dict['context'].keys():
            val = dict['context'][c_key][index].item()
            context_values.append(val)

        context_tensor = torch.tensor(context_values, dtype=torch.float32)

        metrics_values = []
        for m_key in dict['computed_metrics'].keys():
            val = dict['computed_metrics'][m_key][index].item()
            metrics_values.append(val)

        metrics_frame = torch.tensor(metrics_values, dtype=torch.float32)

        metrics_frame = torch.cat((metrics_frame, context_tensor), dim=0).float()


        graph_feats = torch.stack(graph_feats, dim = 0)
        num_nodes = graph_feats.shape[0]

        # FULLY-CONNECTED
        # nodes_idx = torch.arange(num_nodes)
        # graph_edge_index = torch.combinations(nodes_idx, r=2).t()
        # graph_edge_index = torch.cat([graph_edge_index, graph_edge_index.flip(0)], dim=1)

        # CONNECTIONS TO THE ROBOT NODE
        list_N = list(range(1, num_nodes))
        list_0 = [0]*(num_nodes - 1)
        graph_edge_index = np.zeros((2, 2*(num_nodes-1)))
        graph_edge_index[0, :] = list_N + list_0 
        graph_edge_index[1, :] = list_0 + list_N 
        graph_edge_index = torch.tensor(graph_edge_index, dtype=torch.long)


        data = Data(x=graph_feats, edge_index=graph_edge_index)

        return data, metrics_frame

    def _get_walls(self, walls):
        wall_points = []
        id_walls =[]
        angle_walls = []
        length_walls = []

        wx = walls['x']
        wy = walls['y']

        num_walls = len(wx)//2

        for i in range(num_walls):
            # if i!=1:
            #     continue
            x1 = wx[2 * i].item()
            y1 = wy[2 * i].item()
            x2 = wx[2 * i + 1].item()
            y2 = wy[2 * i + 1].item()

            dx, dy = x2 - x1, y2 - y1
            distance = (dx**2 + dy**2)**0.5
            angle = np.arctan2(dy, dx) + np.pi/2
            curr_x = (x1 + x2)/2
            curr_y = (y1 + y2)/2
            wall_points.append([curr_x, curr_y])
            id_walls.append(i)
            angle_walls.append(angle)
            length_walls.append(distance)

        return wall_points, id_walls, angle_walls, length_walls


    def _sample_walls(self, walls, dist_points=0.2):
        wall_points = []
        id_walls =[]
        angle_walls = []

        wx = walls['x']
        wy = walls['y']

        num_walls = len(wx)//2

        for i in range(num_walls):
            # if i!=1:
            #     continue
            x1 = wx[2 * i].item()
            y1 = wy[2 * i].item()
            x2 = wx[2 * i + 1].item()
            y2 = wy[2 * i + 1].item()

            dx, dy = x2 - x1, y2 - y1
            distance = (dx**2 + dy**2)**0.5

            angle = np.arctan2(dy, dx) + np.pi/2
            
            if distance < 1e-6:
                wall_points.append([x1, y1])
                continue
                
            num_points = max(int(distance / dist_points), 1)
            
            for p in range(num_points + 1):
                t = p / num_points
                curr_x = x1 + t * dx
                curr_y = y1 + t * dy
                wall_points.append([curr_x, curr_y])
                id_walls.append(i)
                angle_walls.append(angle)
                
        points_tensor = torch.tensor(wall_points, dtype=torch.float)
        unique_points = torch.unique(points_tensor, dim=0)
        
        return points_tensor, id_walls, angle_walls
    
    def mirror_sequence(self, sequence):
        new_sequence = sequence

        for frame in new_sequence:
            frame.x[:,self.all_features.index('r_y')] = -frame.x[:,self.all_features.index('r_y')]
            frame.x[:,self.all_features.index('r_sin_a')] = -frame.x[:,self.all_features.index('r_sin_a')]
            frame.x[:,self.all_features.index('r_vy')] = -frame.x[:,self.all_features.index('r_vy')]
            frame.x[:,self.all_features.index('r_va')] = -frame.x[:,self.all_features.index('r_va')]            
            frame.x[:,self.all_features.index('r_acc_y')] = -frame.x[:,self.all_features.index('r_acc_y')]

            frame.x[:,self.all_features.index('g_y')] = -frame.x[:,self.all_features.index('g_y')]
            frame.x[:,self.all_features.index('g_sin_a')] = -frame.x[:,self.all_features.index('g_sin_a')]

            frame.x[:,self.all_features.index('h_y')] = -frame.x[:,self.all_features.index('h_y')]
            frame.x[:,self.all_features.index('h_sin_a')] = -frame.x[:,self.all_features.index('h_sin_a')]

            frame.x[:,self.all_features.index('o_y')] = -frame.x[:,self.all_features.index('o_y')]
            frame.x[:,self.all_features.index('o_sin_a')] = -frame.x[:,self.all_features.index('o_sin_a')]

            frame.x[:,self.all_features.index('w_y')] = -frame.x[:,self.all_features.index('w_y')]
            frame.x[:,self.all_features.index('w_sin_a')] = -frame.x[:,self.all_features.index('w_sin_a')]

        return new_sequence



def collate(batch):
    sequences, metrics, labels, sequence_lengths, weights = zip(*batch)  

    flat_graphs = [frame for traj in sequences for frame in traj]

    batched_graphs = Batch.from_data_list(flat_graphs)   

    metrics_tensor = torch.cat(metrics, dim=0)  
    labels_tensor = torch.stack(labels)  
    slengths_tensor = torch.stack(sequence_lengths)
    weights_tensor = torch.stack(weights)


    return batched_graphs, metrics_tensor, labels_tensor, slengths_tensor, weights_tensor
