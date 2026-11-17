import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate
from collections import namedtuple
import numpy as np
import math
import json
import os

from socnavgym.envs.rewards.RNNBaseline import metrics
import yaml
import argparse

DEFAULT_SEED = 42

#help = "You can specify a training task file other than `train.yaml` and/or overwrite the config-based seed."
#parser = argparse.ArgumentParser(description=help)
#parser.add_argument('--task', type=str, default="/home/majid/university/SocNavGym/socnavgym/envs/rewards/RNNBaseline/RNN_Model.yaml", help="yaml file path")
#args = parser.parse_args()


#config_data = yaml.load(open(args.task, "r"), Loader=yaml.Loader)
current_directory = os.path.dirname(__file__)
config_path = os.path.join(current_directory, "RNN_Model.yaml")
print(config_path)
config_data = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

LOSS = config_data["LOSS"]
TRAIN_FILE = config_data["TRAIN_FILE"]
DEV_FILE = config_data["DEV_FILE"]
TEST_FILE = config_data["TEST_FILE"]
BATCH_SIZE = config_data["BATCH_SIZE"]
MAX_EPOCHS = config_data["MAX_EPOCHS"]
MAX_PATIENCE = config_data["MAX_PATIENCE"]
Q_TEST_DIR = config_data["Q_TEST_DIR"]
SAVE_PLOTS_LOCALLY = config_data["SAVE_PLOTS_LOCALLY"]
HIDDEN_SIZE = config_data["HIDDEN_SIZE"]
SEED = config_data["SEED"]
DATA_LIMIT = config_data["DATAPOINT_LIMIT"]
LR = config_data["LR"]
RNN_TYPE = config_data["RNN_TYPE"]
NUM_LAYERS = config_data["NUM_LAYERS"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_NAME = '_'.join([RNN_TYPE, LOSS, str(NUM_LAYERS), str(LR), str(HIDDEN_SIZE)])
torch.manual_seed(SEED)
np.random.seed(SEED)
# robot_features = ['r_x', 'r_y', 'r_angle', 'speed_x', 'speed_y', 'speed_a', 'width']
# metrics_features = ['success', 'hum_exists', 'wall_exists', 'dist_nearest_hum', 'dist_nearest_obj', 'dist_wall', 'dist_goal',
#                     'hum_collision_flag', 'object_collision_flag', 'wall_collision_flag', 
#                     'social_space_intrusion', 'num_near_humans', 'acceleration', 'min_time_to_collision', 'max_fear', 'max_panic']
# context_features = ['urgency', 'risk', 'importance', 'distance_from_human', 'minimum_speed', 'average_speed', 'maximum_speed']
# goal_features = ['x', 'y', 'angle', 'pos_threshold', 'angle_threshold']
# all_features = robot_features + metrics_features + context_features + goal_features

FRAME_THRESHOLD = 0.5
SOCIAL_SPACE_THRESHOLD = 0.4
Patience = 50

# vector = namedtuple('vector', all_features)
#print("Num Features:", len(all_features))
# Custom Dataset for loading JSON files

class SocNav3Data(Dataset):
    def __init__(self, path, limit= -1,  label_exists = True, 
                 overwrite_context = False):
        self.data = []
        self.labels = []
        self.path = path
        self.label_exists = label_exists
        self.limit = limit
        self.overwrite_contexts = overwrite_context


        if type(self.path) is str and self.path.endswith('.txt'):
            #print("running txt case")
            with open(self.path) as set_file:
                ds_files = set_file.read().splitlines()
            #print("number of files for ", self.path, len(ds_files))

            dir = os.path.dirname(self.path)

            for i, filename in enumerate(ds_files):
                filename = os.path.normpath(filename)
                if not os.path.isabs(filename):
                    filename = os.path.join(dir, filename)
                if filename.endswith('.json'):
                    with open(filename, 'r') as f:
                        t_data = json.load(f)
                        # Adjust keys based on your file structure
                        trajectory = self.gather_data(t_data)
                        #if self.label_exists: 
                            #rating = t_data['label']
                        #else:
                        rating = 0.0
                        self.data.append(trajectory)
                        self.labels.append(rating)
                #if i%1000 == 0:
                    #print('x')
                if i + 1 >= self.limit and self.limit > 0:
                    print('Stop including more samples to speed up dataset loading')
                    break

        if type(self.path) is str and self.path.endswith('.json'):
            #print("running JSON case")
            filename = os.path.normpath(self.path)
            with open(filename, 'r') as f:
                        t_data = json.load(f)
                        # Adjust keys based on your file structure
                        trajectory = self.gather_data(t_data)
                        if self.label_exists: 
                            rating = t_data['label']
                        else:
                            rating = 0.0
                        self.data.append(trajectory)
                        self.labels.append(rating)

        

    def get_metrics(self, frame, walls, prev_frame):
        cur_metrics = {}
        cur_timestamp = frame['timestamp']
        # print("Previous Frame",prev_frame)
        prev_timestamp = prev_frame['timestamp']
        window = cur_timestamp - prev_timestamp
        for feature in metrics_features:
            cur_metrics[feature] = 0
        #Get robot features for later
        r_x = frame['robot']['x']
        r_y = frame['robot']['y']
        r_a = frame['robot']['angle']
        r_vx = frame['robot']['speed_x']
        r_vy = frame['robot']['speed_y']
        x_moved = abs(r_x - prev_frame['robot']['x'])
        y_moved = abs(r_y - prev_frame['robot']['y'])
        dist_moved = math.sqrt((x_moved)**2 + (y_moved)**2)
        # cur_metrics['distance_covered'] = dist_moved/100
        r_vlin = math.sqrt(r_vx**2 + r_vy**2)
        # acc_x = 0 
        # acc_y = 0
        if window != 0:
            prev_vlin = math.sqrt(prev_frame['robot']['speed_x']**2 + prev_frame['robot']['speed_y']**2)
            cur_acceleration = prev_vlin/window
        cur_metrics['acceleration'] = cur_acceleration
        r_radius = frame['robot']['shape']['length']    #since length and width are the same
        #Goal Features
        g_x = frame['goal']['x']
        g_y = frame['goal']['y']
        g_a = frame['goal']['angle']
        g_dist =  max(0, math.sqrt((g_x - r_x)**2 + (g_y - r_y)**2))
        cur_metrics['dist_goal'] = g_dist
        g_thr = frame['goal']['pos_threshold']
        a_thr = frame['goal']['angle_threshold']
        #Check for goal success
        if g_dist <= g_thr and abs(g_a - r_a) <= a_thr:
            cur_metrics['success'] = 1
        #Calculate human metrics
        min_hdist = 9    #Initialising distance with a large value
        h_radius = 0.35
        min_ttc = float('inf')     #Time to collision
        max_fear = -1
        max_panic = -1
        for human in frame['people']:
            cur_metrics['hum_exists'] = 1
            h_id = human['id']
            h_x = human['x']
            h_y = human['y']
            h_a = human['angle']
            # human_shape = Point(o_x, o_y).buffer(h_radius) 
            dist_to_robot = max(0, math.sqrt((h_x - r_x)**2 + (h_y - r_y)**2) - (r_radius + h_radius))
            min_hdist = min(min_hdist, dist_to_robot)
            if min_hdist == 0:
                cur_metrics['hum_collision_flag'] = 1
            #Calculate num of humans in vicinity
            if min_hdist < SOCIAL_SPACE_THRESHOLD:
                cur_metrics['num_near_humans'] += 1
                cur_metrics['social_space_intrusion'] = 1
        cur_ttc = metrics.get_ttc(frame, prev_frame)
        valid_ttc_exists = False    
        # Process each dictionary in the list
        for item in cur_ttc:
            # Handle ttc - exclude -1 values
            if item['ttc'] != -1:
                valid_ttc_exists = True
                min_ttc = min(min_ttc, item['ttc'])
                
            # Handle fear - find maximum value
            if item['fear'] > max_fear:
                max_fear = item['fear']
                
            # Handle panic - find maximum value
            if item['panic'] > max_panic:
                max_panic = item['panic']
        
        # If no valid ttc values were found, set min_ttc to -1
        if not valid_ttc_exists:
            min_ttc = -1
        # print("Data from get ttc:",cur_ttc)
        # print(f"min ttc :{min_ttc}, max_fear :{max_fear}, max_panic :{max_panic}")
        cur_metrics['min_time_to_collision'] = min_ttc
        cur_metrics['max_fear'] = max_fear
        cur_metrics['max_panic'] = max_panic
        # min_ttc = min(min_ttc, cur_ttc)
        cur_metrics['dist_nearest_hum'] = min_hdist
        # cur_metrics['time_to_collision'] = min_ttc
        min_odist = 9
        robot = Point(r_x, r_y).buffer(r_radius)
        for object in frame['objects']:
            o_x = object['x']
            o_y = object['y']
            o_angle = object['angle']
            dist_to_robot = metrics.get_dist_from_obj(object, o_x, o_y, o_angle, robot)
            min_odist = min(min_odist, dist_to_robot)
            if min_odist == 0:
                cur_metrics['object_collision_flag'] = 1
        cur_metrics['dist_nearest_obj'] = min_odist

        #Calculate distance to wall
        if len(walls)>0:
            cur_metrics['wall_exists'] = 1
            min_wdist = 9
            for wall in walls:
                w_x1, w_y1 = wall[0], wall[1]
                w_x2, w_y2 = wall[2], wall[3]
                w_dist = metrics.get_wall_distance(r_x, r_y, r_radius, w_x1, w_y1, w_x2, w_y2)
                min_wdist = min(w_dist, min_wdist)
            cur_metrics['dist_wall'] = min_wdist
        else:
            cur_metrics['dist_wall'] = -1

        
        #Calculate jerk and acceleration

        #Calculate TTC

        return cur_metrics

    def gather_data(self, data):
        sequence = data['sequence']
        trajectory_data = []
        walls = data['walls']
        last_timestamp = sequence[0]['timestamp']
        prev_index = 0
        for i, frame in enumerate(sequence):
            current_timestamp = frame['timestamp']
            if current_timestamp-last_timestamp >= FRAME_THRESHOLD:
                frame_features = []
                #add robot features
                robot = frame['robot']
                frame_features.append(robot['x'])
                frame_features.append(robot['y'])
                frame_features.append(robot['angle'])
                frame_features.append(robot['speed_x'])
                frame_features.append(robot['speed_y'])
                frame_features.append(robot['speed_a'])
                frame_features.append(robot['shape']['length'])
                if i == 0:
                    prev_frame = frame
                else:
                    prev_frame = sequence[i-1]
                # print("Prev Frame here:", prev_frame)
                cur_metrics = self.get_metrics(frame, walls, prev_frame)
                for feature in metrics_features:
                    frame_features.append(cur_metrics.get(feature, 0.0))
                try:
                    context = data['context']
                except KeyError:
                    #print("No context, handcrafted context values being used.")
                    context = {"urgency":45, "importance":75, "risk":15 , "distance_from_human":45, "minimum_speed":35, "average_speed":45, "maximum_speed":35}
                for feature in context_features:
                    # print(f"Contexts :{context.get(feature)}")
                    frame_features.append(context.get(feature, 0.0)/100)
                #add goal 
                goal = frame['goal']
                for feature in goal_features:
                    frame_features.append(goal.get(feature, 0.0))
                last_timestamp = current_timestamp
                # print(frame_features, len(frame_features))
                prev_index = i
                trajectory_data.append(frame_features)
                detailed_feats = dict(map(lambda i,j : (i,j) , all_features, frame_features))
                # print(f"Frame Features :{detailed_feats}")
        if trajectory_data:  # Only add non-empty sequences
            return trajectory_data
        else:
            print("sequence too short!! Returning empty sequence")
            return [[0.0] * len(all_features)]  # Return a dummy sequence if empty
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to tensors; ensure consistent dimensions 
        traj = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return traj, label

# Define the RNN model (using LSTM here)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        if RNN_TYPE == "GRU":
            self.layer = nn.GRU(input_size, hidden_size, num_layers,
                                batch_first=True)
        elif RNN_TYPE == "LSTM":
            self.layer = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # To ensure output is between 0 and 1

    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        out, _ = self.layer(x)
        # Use the last output of the sequence for prediction
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Separate sequences and labels
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)  # Convert labels to tensor
    return sequences, labels

def train_model(input_size, hidden_size=256, num_layers=10, 
                batch_size=32, num_epochs=10, patience =50, learning_rate=0.00005, 
                checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    model_name = SAVE_NAME+'.pytorch'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    
    train_dataset = SocNav3Data(TRAIN_FILE, limit=DATA_LIMIT)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = SocNav3Data(DEV_FILE, limit= DATA_LIMIT)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = RNNModel(input_size, hidden_size, num_layers)
    if LOSS == "mse":
        criterion = nn.MSELoss()
    elif LOSS == "bce":
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    min_val_loss = float('inf')
    patience_counter = patience  # Using your global Patience variable
    train_losses, val_losses = [], []
    
    print(f"Starting training for {num_epochs} epochs (or until early stopping)")
    print(f"Model will be saved to {checkpoint_path} when validation loss improves")
    
    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        epoch_loss = 0
        for trajectories, labels in train_dataloader:
            outputs = model(trajectories)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for trajectories, labels in val_dataloader:
                outputs = model(trajectories)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        full_preds = []
        for filename, files, qual_loader in qual_loaders:
            with torch.no_grad():
                # print(f"======================\n{filename=}")
                for trajectories, _ in qual_loader:
                    # Pass the whole batched graph sequence to the model at once
                    preds = model(trajectories)
                    full_preds += preds.tolist()
            # video_metrics[filename] = dict(map(lambda i,j : (i,j) , files, full_preds))
        plot_qualitative_ratings(full_preds, epoch=epoch)
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        # Checkpointing - save when validation loss improves
        if val_loss < min_val_loss:
            improvement = min_val_loss - val_loss
            min_val_loss = val_loss
            patience_counter = patience  # Reset patience counter
            
            # Save the checkpoint with detailed information
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'train_loss': train_loss,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            print(f"Checkpoint saved at epoch {epoch+1}")
            print(f"Val Loss: {val_loss:.6f} (improved by {improvement:.6f})")
            
            # Optionally save a timestamped version too if you want a history
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # history_path = os.path.join(checkpoint_dir, f'model_e{epoch+1}_{timestamp}.pth')
            # torch.save(checkpoint, history_path)
        else:
            patience_counter -= 1
            print(f"****Epoch {epoch+1}: Val Loss: {val_loss:.6f} (best: {min_val_loss:.6f}, patience: {patience_counter}/{patience})")
        
        if patience_counter == 0:
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {min_val_loss:.6f}")
            break
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Plot Training Progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join('plots', SAVE_NAME+'_progression'+'.png')
    plt.savefig(fig_path)
    plt.show()
    print(f"\nTraining completed. Best model saved at: {checkpoint_path}")
    return model, checkpoint_path

    # Run Inference on Test Set
def plot_predictions_vs_expected(predictions, labels):
    """
    Create a line plot comparing predicted vs expected labels, sorted by expected labels.
    Also calculates and prints the Mean Squared Error (MSE) manually.
    
    Args:
        predictions: numpy array of model predictions
        labels: numpy array of true labels
    """
    # Convert to numpy arrays if they aren't already
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Sort by expected labels
    sort_idx = np.argsort(labels)
    sorted_labels = labels[sort_idx]
    sorted_predictions = predictions[sort_idx]
    
    # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse = np.mean((sorted_predictions - sorted_labels) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")

    mae = np.mean(np.abs(sorted_predictions - sorted_labels))
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_labels, label='Expected', color='blue')
    plt.plot(sorted_predictions, label='Predicted', color='red')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Expected vs Predicted Labels')
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    fig_path = os.path.join('plots', SAVE_NAME+'_pve'+'.png')
    plt.savefig(fig_path)
    plt.show()
    plt.close()
    return mse, mae

def plot_qualitative_ratings(predicted_ratings, model_name='RNN baseline', context="Context not provided", epoch = 9999):
    """
    Create, show, and save a plot of predicted ratings across distances.
   
    Args:
        predicted_ratings: List of ratings to plot
        model_name: Name of the model to be displayed in the legend
        context: Plot title/context
        save_path: Path to save the figure (if None, won't save)
    """
    # Create a single figure and axis
    value = 0.1
    index = 0
    increment = 0.08
    factor = 1.08
    distances = []
    for i in range(len(predicted_ratings)):
        # print(predicted_rating, filename)
        distances.append(value)
        distances.append(-1*value)
        # print(value)
        value += increment
        increment *= factor
        index += 2
    # print("Length of distances: ",len(distances))
    distances = distances[:len(distances)//2]
    distances.sort()
    plt.figure(figsize=(15, 9))
    # Default styles if none provided
    styles = [
            {'color': '#0066CC', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 1},  # Blue
            {'color': '#CC0000', 'marker': 's', 'linestyle': '-', 'linewidth': 2, 'markersize': 1},  # Red
        ]
    datasets = [(distances, predicted_ratings, model_name)]
    for idx, (distance, predicted_rating, label) in enumerate(datasets):
        # Plot predicted ratings with a different style
        modified_style = styles[(idx + 1) % len(styles)].copy()
        modified_style['linestyle'] = '-'  # Use dashed line for predicted
        plt.plot(distance,
                predicted_rating,
                label=f'{label} (Predicted)',
                **modified_style,
                alpha=0.8)
    title = f"Qualitative Predictions for {model_name}"
    # Customize the plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.8, facecolor='white')
    # Set y-axis limits between 0 and 1 since ratings are normalized
    plt.ylim(0, 1)# Adjust layout to prevent label cutoff
    plt.tight_layout()
    save_dir = os.path.join('plots', SAVE_NAME+'_q_test')
    os.makedirs(save_dir, exist_ok=True)
    save_loc = os.path.join(save_dir, 'w_'+str(epoch))
    if SAVE_PLOTS_LOCALLY:
        plt.savefig(save_loc)
    # plt.show()
    plt.close()

def perform_qualitative_tests(model, checkpoint_path, batch_size=32):
    # LOAD THE DATASET
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    video_metrics = {}
    full_preds = []
    for filename, files, qual_loader in qual_loaders:
        with torch.no_grad():
            # print(f"======================\n{filename=}")
            for trajectories, _ in qual_loader:
                # Pass the whole batched graph sequence to the model at once
                preds = model(trajectories)
                full_preds += preds.tolist()
                
        video_metrics[filename] = dict(map(lambda i,j : (i,j) , files, full_preds))
    plot_qualitative_ratings(full_preds)
    print("Video Metrics: ",video_metrics)
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        for trajectories, label in test_dataloader:
            outputs = model(trajectories)
            predictions.extend(outputs.squeeze().tolist())
            labels.extend(label.squeeze().tolist())
    
    #plot predictions
    mse, mae = plot_predictions_vs_expected(predictions, labels)
    test_results = {
        'mse': mse,
        'mae': mae
    }
    print("Inference on test_set_socnav3_02_25_25.txt completed.")
    return video_metrics,test_results
#test_dataset = SocNav3Data(TEST_FILE)
#test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#qual_loaders = []

#for filename in os.listdir(Q_TEST_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join("qual_txt", filename)
        print(f"Processing file: {file_path}")
        with open(file_path) as set_file:
            files = set_file.read().splitlines()
        qual_set = SocNav3Data(file_path, label_exists= False)
        qual_loader = DataLoader(qual_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        qual_loaders.append((filename, files, qual_loader))
    else:
        print("Skipping", filename)
#model, checkpoint = train_model(len(all_features), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,patience= MAX_PATIENCE, batch_size= BATCH_SIZE)
#model =  RNNModel(len(all_features), 256, 10)
#checkpoint = os.path.join("checkpoints","GRU_5_5e-05.pytorch" ) 
# video_metrics, test_results = predict(model, checkpoint)
# evaluation_results = {
#     'model_name': 'MLP Baseline',
#     'model_metrics': test_results,
#     'video_predictions': video_metrics,
# }

# output_filename = f"GRU_256_10_eval.json"
# with open(output_filename, 'w') as f:
#     json.dump(evaluation_results, f, indent=4)

# print(f"\nEvaluation results saved to {output_filename}")

# qual_set = SocNav3Data("dev_set_socnav3_skip_control.txt")
# qual_loaders = []
# for filename in os.listdir(Q_TEST_DIR):
#     if filename.endswith(".txt"):
#         file_path = os.path.join("qual_txt", filename)
#         print(f"Processing file: {file_path}")
#         with open(file_path) as set_file:
#             files = set_file.read().splitlines()
#         qual_set = SocNav3Data(file_path, label_exists= False)
#         qual_loader = DataLoader(qual_set,  batch_size=32, shuffle=False, collate_fn=collate_fn)
#         qual_loaders.append((filename, files, qual_loader))
#     else:
#         print("Skipping", filename)