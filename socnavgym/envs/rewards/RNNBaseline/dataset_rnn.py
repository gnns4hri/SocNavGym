import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from shapely.geometry import Point
import math
import json
import numpy as np
import os
import pandas as pd
import sys
import random
import copy

sys.path.append(os.path.join(os.path.dirname(__file__)+'../tools/data_transformation'))
sys.path.append(os.path.dirname(__file__))


from data_conversions import sequence_to_tensor, clone_sequence
from data_mirroring import mirror_tDic_sequence, tensor_transform_with_random_mirroring
from data_normalization import tensor_transform_to_goal_fr

from torch.utils.data import DataLoader

import metrics



import metrics


class TrajectoryDataset(Dataset):
    def __init__(self, data_file, contextQ_file, path = '../dataset', limit= -1,  frame_threshold = 0.1, label_exists = True, 
                 overwrite_context = False, data_augmentation = False, load_time_augmentation = True, reload = False):
        self.data = []
        self.labels = []
        self.path = path
        self.data_file = data_file

        if type(data_file) is str:
            DA = '_DA_' if data_augmentation and load_time_augmentation else ''
            self.reload_fname = '.'.join(data_file.split('.')[:-1]) + '_' + contextQ_file.split('/')[-1].split('.csv')[0] + DA + '.pytorch'
        else:
            self.reload_fname = ''
            reload = False
        self.label_exists = label_exists
        self.limit = limit
        self.overwrite_contexts = overwrite_context
        self.frame_threshold = frame_threshold
        self.data_augmentation = data_augmentation
        self.load_time_augmentation = load_time_augmentation
        self.context_df = pd.read_csv(contextQ_file, index_col='context') 

        self.robot_features = ['robot_x', 'robot_y', 'robot_a', 'speed_x', 'speed_y', 'speed_a', 'acceleration_x', 'acceleration_y']

        self.metrics_features = ['success', 'hum_exists', 'wall_exists', 'dist_nearest_hum', 'dist_nearest_obj', 'dist_wall', 'dist_goal',
                            'hum_collision_flag', 'object_collision_flag', 'wall_collision_flag', 'social_space_intrusionA',
                            'social_space_intrusionB', 'social_space_intrusionC', 'num_near_humansA', 'num_near_humansB', 'num_near_humansC',
                           'num_near_humansA2', 'num_near_humansB2', 'num_near_humansC2',
                             'min_time_to_collision', 'min_time_to_collision2', 'max_fear', 'max_panic',
                             'global_dist_nearest_hum', 'path_efficiency_ratio', 'step_ratio', 'episode_end']
        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        self.goal_features = ['goal_pos_threshold', 'goal_angle_threshold']
        
        self.all_features = self.robot_features + self.metrics_features + self.goal_features + self.context_features

        self.MAX_TTC = 10
        self.MAX_LSPEED = 2

        self.max_metric_values = {'robot_x': 10, 'robot_y': 10, 'robot_a': np.pi, 'speed_x': self.MAX_LSPEED, 
                            'speed_y': self.MAX_LSPEED, 'speed_a': np.pi, 'acceleration_x': 3, 
                            'acceleration_y': 3, 'success': 1, 'hum_exists': 1, 'wall_exists': 1,
                            'dist_nearest_hum': 10, 'dist_nearest_obj': 10, 'dist_wall': 10, 'dist_goal': 10,
                            'hum_collision_flag': 1, 'object_collision_flag': 1, 'wall_collision_flag': 1, 
                            'social_space_intrusionA': 1, 'social_space_intrusionB': 1, 'social_space_intrusionC': 1,
                            'num_near_humansA': 10, 'num_near_humansB': 10, 'num_near_humansC': 10, 
                            'min_time_to_collision': self.MAX_TTC, 'max_fear': 10, 'max_panic': 10,
                            'num_near_humansA2': 100, 'num_near_humansB2': 100, 'num_near_humansC2': 100,
                            'min_time_to_collision2': self.MAX_TTC**2,
                            'global_dist_nearest_hum': 10, 'path_efficiency_ratio': 1, 'step_ratio': 1, 'episode_end': 1,
                            'goal_pos_threshold': 10, 'goal_angle_threshold': np.pi}

        for var in self.context_features:
            self.max_metric_values[var] = 100

        if reload is True:
            if os.path.exists(self.reload_fname):
                loaded = torch.load(self.reload_fname)
                self.data = loaded['data']
                self.labels = loaded['labels']
                print("number of trajectories for ", self.data_file, len(self.data))
                return


        if type(self.data_file) is str and self.data_file.endswith('.txt'):
            print(self.data_file)
            with open(self.data_file) as set_file:
                ds_files = set_file.read().splitlines()

            print("number of files for ", self.data_file, len(ds_files))
        elif type(self.data_file) is str and self.data_file.endswith('.json'):
            ds_files = [self.data_file]
        elif type(self.data_file) is list:
            ds_files = self.data_file

        self.ds_files = ds_files
        for i, filename in enumerate(ds_files):
            if filename.endswith('.json'):
                file_path = os.path.join(self.path, filename)
                with open(file_path, 'r', encoding="utf-8") as f:
                    try:
                        t_data = json.load(f)
                    except:
                        print("FileName :", file_path)


                    if 'context_description' in t_data.keys(): #not self.overwrite_contexts:
                        context_desc = t_data['context_description']
                    else:
                        context_desc = self.overwrite_contexts
                    context = self.context_df.loc[context_desc.rstrip()].to_dict()

                    tensor_dict = sequence_to_tensor(t_data, self.frame_threshold, context)
                    tensor_dict_to_goal = tensor_transform_to_goal_fr(tensor_dict)
                    if self.label_exists and 'label' in t_data: 
                        rating = t_data['label']
                    else:
                        rating = 0.0
                    self.data.append(tensor_dict_to_goal)
                    self.labels.append(rating)

                    if self.data_augmentation and self.load_time_augmentation:
                        tensor_dict_clone = clone_sequence(tensor_dict_to_goal)
                        t_data_mirrored = mirror_tDic_sequence(tensor_dict_clone)
                        self.data.append(t_data_mirrored)
                        self.labels.append(rating)

                    
            if i%1000 == 0:
                print(i)
            if i + 1 >= self.limit and self.limit > 0:
                print('Stop including more samples to speed up dataset loading')
                break
        if reload:
            torch.save({
                'data': self.data,
                'labels': self.labels
                }, self.reload_fname)


      
    def get_all_features(self):
        return self.all_features

    def get_context_features(self):
        return self.context_features

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.data_augmentation and not self.load_time_augmentation:
            data = tensor_transform_with_random_mirroring(data)

        self.current_trajectory = data

        label = self.labels[idx]

        metrics_sequence = metrics.compute_metrics(data)
        final_tensor = metrics.normalize_and_cat_features(metrics_sequence, self.max_metric_values, self.all_features)

        traj = final_tensor.to(torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return traj, label

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Separate sequences and labels
    sequence_lengths = [s.shape[0]-1 for s in sequences]
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)  # Convert labels to tensor
    return sequences, labels, torch.tensor(sequence_lengths, dtype=torch.long)

