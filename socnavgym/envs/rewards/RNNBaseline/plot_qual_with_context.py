from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import sys
from pathlib import Path
from dataset_rnn import TrajectoryDataset, collate_fn
from rnn import RNNModel
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from torch_geometric.data import Batch
import torch
import os
from utils import plot_qualitative_multiple
import yaml
import argparse

parser = argparse.ArgumentParser(
                    prog='plot_qual_with_context',
                    description='Plots estimated ratings for different trajectories considering several contexts.')
parser.add_argument('--model', type=str, nargs="?", required = True, help="Model used for rating prediction.")
parser.add_argument('--config', type=str, nargs="?", required = True,  help="Configuration file including the paths to the data and other arguments.")
parser.add_argument('--context', type=str, nargs="?", required = True,  help="Context file for the generation of the context variables.")
parser.add_argument('--dataroot', type=str, nargs="?", default='.', help="Data root directory.")
parser.add_argument('--saveplot', type=str, nargs="?", default=None, help="File where the plot will be saved.")
parser.add_argument('--show', type=bool, default=True, action=argparse.BooleanOptionalAction, help="Show the plots on a window.")

args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)

# Global configuration
checkpoint_path = args.model
print(checkpoint_path)
model_name = checkpoint_path #os.path.splitext(os.path.basename(checkpoint_path))[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']

if 'use_new_context' in checkpoint.keys():
    USE_NEW_CONTEXT = checkpoint['use_new_context']
else:
    USE_NEW_CONTEXT = False
if 'rnn_type' in checkpoint.keys():
    RNN_TYPE = checkpoint['rnn_type']
else:
    RNN_TYPE = 'GRU'
if 'frame_threshold' in checkpoint.keys():
    FRAME_THRESHOLD = checkpoint['frame_threshold']
else:
    FRAME_THRESHOLD = 0.1

if 'linear_layers' in checkpoint.keys():
    LINEAR_LAYERS = checkpoint['linear_layers']
else:
    LINEAR_LAYERS = []

if 'activation' in checkpoint.keys():
    ACTIVATION = checkpoint['activation']
else:
    ACTIVATION = 'linear'

if 'context_features' in checkpoint.keys():
    CONTEXT_FEATURES = checkpoint['context_features']
else:
    CONTEXT_FEATURES = 0

    
model = RNNModel(input_size, hidden_size, num_layers, rnn_type = RNN_TYPE, linear_layers = LINEAR_LAYERS, activation=ACTIVATION, context_vars=CONTEXT_FEATURES).to(device)
    
model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)


model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

config_file = args.config
config_data = yaml.load(open(config_file, "r"), Loader=yaml.Loader)

contextQ_file = args.context

FILES = config_data["FILES"]
NAMES = config_data["NAMES"]
COLORS = config_data["COLORS"]
CONTEXTS = config_data["CONTEXTS"]
ABBREVIATED_CONTEXTS = config_data["ABBREVIATED_CONTEXTS"]

data_root = args.dataroot

BATCH_SIZE = 32
FIG_COLS = 2
FIG_ROWS = len(config_data["CONTEXTS"])//2

fig, axes = plt.subplots(FIG_ROWS, FIG_COLS, figsize=(5*FIG_ROWS, 3*FIG_COLS))

axes = axes.reshape(1, -1)

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Hide any unused subplots
# for i in range(4, len(axes_flat)):
#     axes_flat[i].set_visible(False)

sample_step = 1

W_SPACE = 0.01

for i_d, d in enumerate(FILES):
    d = os.path.join(data_root, d)
    q_preds = {}
    q_indices = {}
    for abbreviated_context, context in zip(ABBREVIATED_CONTEXTS, CONTEXTS):
        print(d)
        qual_set = TrajectoryDataset(d, contextQ_file, path = data_root, frame_threshold = FRAME_THRESHOLD, label_exists= False, 
                               overwrite_context=context, data_augmentation=False)
        all_features = qual_set.get_all_features()
        qual_loader = DataLoader(qual_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers = 4)

        dist_idx = all_features.index('robot_y')
        predictions = []
        actual_labels = []
        val_loss = 0
        distances = []
        with torch.no_grad():
            for trajectories, _, slengths in qual_loader:
                
                for t in trajectories:
                    distances_t = []
                    for s in t:
                        distances_t.append(s[dist_idx].to('cpu')*10)
                    min_dist = np.max(np.array(distances_t))
                    max_dist = np.min(np.array(distances_t))
                    dist = min_dist if abs(min_dist) > abs(max_dist) else max_dist
                    distances.append(dist)        
                trajectories = trajectories.to(device)
                slengths = slengths.to(device)
                preds = model(trajectories, slengths)
                predictions += preds.tolist()
            # print(distances)
        sort_idx = np.argsort(np.array(distances))
        idx = np.arange(0, len(predictions), sample_step)
        q_indices[abbreviated_context] = np.array(distances)[sort_idx]
        q_preds[abbreviated_context] = np.array(predictions)[sort_idx]
        print(context, predictions)

    for idx, context in enumerate(q_preds.keys()):
        add_x_label = add_y_label = False
        if idx % FIG_COLS == 0: # We want to remove stuff on the left for the second row
            add_y_label = True
        if idx//FIG_COLS == FIG_ROWS-1:     # We want to remove stuff on the bottom for the first row
            add_x_label = True
        axes_flat[idx] = plot_qualitative_multiple(q_preds[context], NAMES[i_d], context, COLORS[i_d], sample_step=sample_step,
                                                   ax=axes_flat[idx], add_xlabel=add_x_label, add_ylabel=add_y_label,
                                                   indices = q_indices[context])
plt.tight_layout(pad=0)
if args.saveplot is not None:
    print("Saving figure as:", args.saveplot)
    plt.savefig(args.saveplot, format=args.saveplot.split(".")[-1], pad_inches=0)
if args.show is True:
    plt.show()


