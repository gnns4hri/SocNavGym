from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate, translate
import numpy as np
import math
import torch

EPS = 0.01

def get_dist_from_obj(object, o_x, o_y, o_angle, robot):
    o_shape = object['shape']['type']
    if o_shape == 'circle':
        o_length = object['shape']['length']
        object_shape = Point(o_x, o_y).buffer(o_length/2)  # o_length is the radius
    elif o_shape == 'rectangle':
        o_length = object['shape']['length']
        o_width = object['shape']['width']
        half_length, half_width = o_length / 2, o_width / 2
        rect = Polygon([
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ])
        # Rotate the rectangle
        rotated_rect = rotate(rect, o_angle, origin=(0, 0), use_radians=True)
        
        # Translate (move) the rectangle to the object's actual position
        object_shape = translate(rotated_rect, xoff=o_x, yoff=o_y)
    else:
        raise ValueError("Invalid object shape. Must be 'circle' or 'rectangle'.")
    distance = robot.distance(object_shape)
    return distance


def get_wall_distance(r_x, r_y, r_radius, w_x1, w_y1, w_x2, w_y2):
    # Define robot as a circle
    robot = Point(r_x, r_y).buffer(r_radius)
    
    # Define wall as a line segment
    wall = LineString([(w_x1, w_y1), (w_x2, w_y2)])
    
    # Compute the minimum distance between the robot's boundary and the wall
    distance = robot.distance(wall)
    
    return round(distance, 2)

SOCIAL_SPACE_THRESHOLD = 0.4

def dist_to_humans(frame):
    r_x, r_y = frame['robot']['x'], frame['robot']['y']
    r_radius = frame['robot']['shape']['length']/2. 
    h_radius = 0.3

    d_humans = []
    for human in frame['people']:
        h_x = human['x']
        h_y = human['y']

        dist_to_robot = max(0, math.sqrt((h_x - r_x)**2 + (h_y - r_y)**2) - (r_radius + h_radius))
        d_humans.append(dist_to_robot)
    
    return d_humans



def dist_to_objects(frame):
    r_x, r_y = frame['robot']['x'], frame['robot']['y']
    r_radius = frame['robot']['shape']['length']/2. 

    robot = Point(r_x, r_y).buffer(r_radius)
    d_objects = []
    for obj in frame['objects']:
        o_x = obj['x']
        o_y = obj['y']
        o_angle = obj['angle']
        dist_to_robot = get_dist_from_obj(obj, o_x, o_y, o_angle, robot)
        d_objects.append(dist_to_robot)
    
    return d_objects

def dist_to_walls(frame, walls):
    r_x, r_y = frame['robot']['x'], frame['robot']['y']
    r_radius = frame['robot']['shape']['length']/2. 

    d_walls = []
    for wall in walls:
        w_x1, w_y1 = wall[0], wall[1]
        w_x2, w_y2 = wall[2], wall[3]
        w_dist = get_wall_distance(r_x, r_y, r_radius, w_x1, w_y1, w_x2, w_y2)
        d_walls.append(w_dist)
    
    return d_walls

def get_ttc(cur_frame, prev_frame):
    robot_pose = np.array([cur_frame['robot']['x'], cur_frame['robot']['y']])
    time_diff = cur_frame['timestamp'] - prev_frame['timestamp']
    robot_pose_prev = np.array([prev_frame['robot']['x'], prev_frame['robot']['y']])
    if time_diff>0:
        robot_vel = (robot_pose-robot_pose_prev)/time_diff
    else:
        robot_vel = np.array([0., 0.])
    # robot_vel = np.array([cur_frame['robot']['speed_x'], cur_frame['robot']['speed_y']])
    human_radius = 0.3 ## Let's assume human radius is 0.3
    length =  cur_frame['robot']['shape']['length']
    width =  cur_frame['robot']['shape']['width']
    robot_radius = np.linalg.norm([length, width])/2
    
    radii_sum = human_radius + robot_radius ## Sum of human and robot radii
    radii_sum_sq = radii_sum * radii_sum
    
    calc_metrics = []
    for human in cur_frame['people']:
        current_metrics = {}        
        ttc = -1
        cost_panic = -1
        cost_fear = -1
        C = np.array([human['x'], human['y']]) - robot_pose # Difference between centers
        C_sq = C.dot(C)
        
        human_vel = np.array([0., 0.])
        for prev_human in prev_frame['people']:
            if prev_human['id'] == human['id']:
                pose_diff = np.array([human['x'] - prev_human['x'], human['y'] - prev_human['y']])
                if time_diff>0:
                    human_vel = pose_diff/time_diff
                break
        
        if C_sq < radii_sum_sq:
            ttc = 0                         ## Human and robot are already in Collision
        else:
            V = robot_vel - human_vel       ## Difference between human and robot velocities
            C_dot_V = C.dot(V)              ## Dot product between the vectors
            if C_dot_V > 0:
                V_sq = V.dot(V)
                f = (C_dot_V * C_dot_V) - (V_sq * (C_sq - radii_sum_sq))
                if f > 0:
                    ttc = (C_dot_V - np.sqrt(f)) / V_sq
                else:
                    g = np.sqrt(V_sq * C_sq - C_dot_V * C_dot_V)
                    if((g - (np.sqrt(V_sq) * radii_sum)) > EPS):
                        cost_panic = np.sqrt(V_sq / C_sq) * (g / (g - (np.sqrt(V_sq) * radii_sum))) ## Panic cost

        if ttc > EPS:
            cost_fear = 1.0/ttc
        elif ttc>=0:
            cost_fear = 10.
        ## fear cost

        current_metrics['id'] = human['id']            
        current_metrics['ttc'] = ttc
        current_metrics['fear'] = cost_fear
        current_metrics['panic'] = cost_panic
        calc_metrics.append(current_metrics)       
        
    return calc_metrics

def compute_metrics(tDict_sequence):
    robot = tDict_sequence['robot']
    goal = tDict_sequence['goal']
    people = tDict_sequence['people']
    objects = tDict_sequence['objects']
    walls = tDict_sequence['walls']
    metrics_ft = tDict_sequence['metrics']

    dist_to_goal_pos = torch.sqrt(torch.pow((robot['x']-goal['x']), 2) + torch.pow((robot['y']-goal['y']), 2)) 
    angle_diff = robot['a']-goal['a']
    dist_to_goal_angle = torch.abs(torch.arctan2(torch.sin(angle_diff), torch.cos(angle_diff)))

    success = torch.logical_and((dist_to_goal_pos<(goal['th_p'])), (dist_to_goal_angle<goal['th_a']))

    hum_exists = (torch.sum(people['exists'], dim = 1)>0).float()
    obj_exists = (torch.sum(objects['exists'], dim = 1)>0).float()
    wall_exists = torch.tensor([torch.numel(walls['x'])>0]).repeat(robot['x'].shape).float().to(torch.float64)

    if people['exists'].numel() > 0:
        dist_human = torch.where(people['exists'], metrics_ft['dist_human'], torch.inf)
        dist_nearest_hum = torch.min(dist_human, dim = 1).values
    else:
        dist_human = torch.tensor([torch.inf]).repeat((robot['x'].shape[0],1)).float().to(torch.float64)
        dist_nearest_hum = dist_human.squeeze()
    
    if objects['exists'].numel() > 0:
        dist_nearest_object = torch.min(torch.where(objects['exists'], metrics_ft['dist_object'], torch.inf), dim = 1).values
    else:
        dist_nearest_object = torch.tensor([torch.inf]).repeat(robot['x'].shape).float().to(torch.float64)

    inf_tensor = torch.full((wall_exists.shape[0], 1), torch.inf).to(torch.float64)
    dist_wall = torch.min(torch.cat((metrics_ft['dist_walls'], inf_tensor), dim=1), dim = 1).values

    human_collision_flag = (dist_nearest_hum<=0.).float()
    object_collision_flag = (dist_nearest_object<=0.).float()
    wall_collision_flag = (dist_wall<=0.).float()

    social_space_intrusionA = (dist_nearest_hum<SOCIAL_SPACE_THRESHOLD).float()
    num_near_humansA = torch.sum(dist_human<SOCIAL_SPACE_THRESHOLD, dim = 1).float()
    num_near_humansA2 = torch.pow(num_near_humansA, 2)

    social_space_intrusionB = (dist_nearest_hum<SOCIAL_SPACE_THRESHOLD*1.5).float()
    num_near_humansB = torch.sum(dist_human<SOCIAL_SPACE_THRESHOLD*1.5, dim = 1).float()
    num_near_humansB2 = torch.pow(num_near_humansB, 2)

    social_space_intrusionC = (dist_nearest_hum<SOCIAL_SPACE_THRESHOLD*2.0).float()
    num_near_humansC = torch.sum(dist_human<SOCIAL_SPACE_THRESHOLD*2.0, dim = 1).float()
    num_near_humansC2 = torch.pow(num_near_humansC, 2)

    if people['exists'].numel() > 0:
        valid_ttc = torch.logical_and(people['exists'], metrics_ft['ttc']>=0.)
        ttc = torch.where(valid_ttc, metrics_ft['ttc'], torch.inf)
        min_ttc = torch.min(ttc, dim = 1).values
        panic = torch.where(torch.logical_and(people['exists'], metrics_ft['panic']>=0), metrics_ft['panic'], 0.)
        max_panic = torch.max(panic, dim = 1).values
        fear = torch.where(torch.logical_and(people['exists'], metrics_ft['fear']>=0), metrics_ft['fear'], 0.)
        max_fear = torch.max(fear, dim = 1).values
    else:
        ttc = torch.tensor([torch.inf]).repeat(robot['x'].shape).to(torch.float64)
        min_ttc = dist_human.squeeze()
        max_panic = torch.zeros(min_ttc.shape, dtype=torch.float64)
        max_fear = torch.zeros(min_ttc.shape, dtype=torch.float64)

    global_dist_nearest_hum = torch.cummin(dist_nearest_hum, 0).values

    acum_dist_travelled = torch.cumsum(robot['dist_travelled'], 0)
    initial_dist_to_goal = torch.tensor([dist_to_goal_pos[0]]).repeat(dist_to_goal_pos.shape).to(torch.float64)
    path_efficiency_ratio = torch.clamp(torch.div(initial_dist_to_goal, acum_dist_travelled), 0., 1.)

    step_ratio = tDict_sequence['indices']/tDict_sequence['indices'][-1]
    episode_end = torch.zeros(dist_to_goal_pos.shape, dtype=torch.float64)
    episode_end[-1] = 1.

    cur_time = tDict_sequence['timestamp']
    prev_time = torch.zeros(cur_time.shape, dtype = torch.float64)
    prev_time[1:] = cur_time[:-1]
    prev_time[0] = prev_time[1]-1
    prev_speed_x = torch.zeros(robot['vx'].shape, dtype = torch.float64)
    prev_speed_x[1:] = robot['vx'][:-1]
    prev_speed_x[0] = prev_speed_x[1]
    prev_speed_y = torch.zeros(robot['vy'].shape, dtype = torch.float64)
    prev_speed_y[1:] = robot['vy'][:-1]
    prev_speed_y[0] = prev_speed_y[1]

    diff_time = cur_time-prev_time
    acceleration_x = (robot['vx']-prev_speed_x)/diff_time
    acceleration_y = (robot['vy']-prev_speed_y)/diff_time

    tDict_sequence['robot']['acc_x'] = acceleration_x
    tDict_sequence['robot']['acc_y'] = acceleration_y

    metrics_dict = {
        'dist_to_goal_pos': dist_to_goal_pos,
        'dist_to_goal_angle': dist_to_goal_angle,
        'success': success,
        'hum_exists': hum_exists,
        'wall_exist': wall_exists,
        'dist_nearest_hum': dist_nearest_hum,
        'dist_nearest_object': dist_nearest_object,
        'dist_wall': dist_wall,
        'human_collision_flag': human_collision_flag,
        'object_collision_flag': object_collision_flag,
        'wall_collision_flag': wall_collision_flag, 
        'social_space_intrusionA': social_space_intrusionA, 
        'num_near_humansA': num_near_humansA, 
        'num_near_humansA2': num_near_humansA2, 
        'social_space_intrusionB': social_space_intrusionB, 
        'num_near_humansB': num_near_humansB, 
        'num_near_humansB2': num_near_humansB2, 
        'social_space_intrusionC': social_space_intrusionC, 
        'num_near_humansC': num_near_humansC, 
        'num_near_humansC2': num_near_humansC2, 
        'min_ttc': min_ttc, 
        'min_ttc2': torch.pow(min_ttc, 2), 
        'max_fear': max_fear, 
        'max_panic': max_panic, 
        'global_dist_nearest_hum': global_dist_nearest_hum, 
        'path_efficiency_ratio': path_efficiency_ratio, 
        'step_ratio': step_ratio, 
        'episode_end': episode_end, 
        # 'acceleration_x': acceleration_x, 
        # 'acceleration_y': acceleration_y
    }

    tDict_sequence['computed_metrics'] = metrics_dict

    return tDict_sequence

def normalize_features(tDict_sequence, max_values):
    for key, value in tDict_sequence.items():
        if key == 'robot':
            tDict_sequence['robot']['x'] = torch.clamp(value['x'], -max_values['scale'], max_values['scale']) / max_values['scale']
            tDict_sequence['robot']['y'] = torch.clamp(value['y'], -max_values['scale'], max_values['scale']) / max_values['scale']
            tDict_sequence['robot']['w'] = torch.clamp(value['w'], -max_values['scale'], max_values['scale']) / max_values['scale']
            tDict_sequence['robot']['l'] = torch.clamp(value['l'], -max_values['scale'], max_values['scale']) / max_values['scale']
            tDict_sequence['robot']['vx'] = torch.clamp(value['vx'], -max_values['max_v'], max_values['max_v']) / max_values['max_v']
            tDict_sequence['robot']['vy'] = torch.clamp(value['vy'], -max_values['max_v'], max_values['max_v']) / max_values['max_v'] 
            tDict_sequence['robot']['va'] = torch.clamp(value['va'], -max_values['max_va'] , max_values['max_va']) / max_values['max_va'] 
            tDict_sequence['robot']['acc_x'] = torch.clamp(value['acc_x'], -max_values['max_acc'], max_values['max_acc']) / max_values['max_acc']
            tDict_sequence['robot']['acc_y'] = torch.clamp(value['acc_y'], -max_values['max_acc'], max_values['max_acc']) / max_values['max_acc'] 
        
        elif key == 'goal':
            tDict_sequence['goal']['th_p'] = torch.clamp(value['th_p'], -max_values['scale'], max_values['scale']) / max_values['scale']
            max_th_a = max_values['max_va']
            tDict_sequence['goal']['th_a'] = torch.clamp(value['th_a'], -max_th_a, max_th_a) / max_th_a

        elif key in ['people', 'objects']:
            if value['x'].numel() == 0 or value['x'].size(1) == 0:
                tDict_sequence[key]['x'] = value['x']
                tDict_sequence[key]['y'] = value['y']
                if key == 'objects':
                    tDict_sequence[key]['w'] = value['w']
                    tDict_sequence[key]['l'] = value['l']
            else:
                mask = value['exists']
                tDict_sequence[key]['x'] = torch.where(mask, torch.clamp(value['x'], -max_values['scale'], max_values['scale']) / max_values['scale'], 0.0)
                tDict_sequence[key]['y'] = torch.where(mask, torch.clamp(value['y'], -max_values['scale'], max_values['scale']) / max_values['scale'], 0.0)
                if key == 'objects':
                    tDict_sequence[key]['w'] = torch.where(mask, torch.clamp(value['w'], -max_values['scale'], max_values['scale']) / max_values['scale'], 0.0)
                    tDict_sequence[key]['l'] = torch.where(mask, torch.clamp(value['l'], -max_values['scale'], max_values['scale']) / max_values['scale'], 0.0)

        elif key == 'metrics':
            tDict_sequence[key]['dist_human'] = torch.clamp(value['dist_human'], -max_values['scale'], max_values['scale']) / max_values['scale']
            tDict_sequence[key]['dist_object'] = torch.clamp(value['dist_object'], -max_values['scale'], max_values['scale']) / max_values['scale']
            tDict_sequence[key]['dist_walls'] = torch.clamp(value['dist_walls'], -max_values['scale'], max_values['scale']) / max_values['scale']

        elif key == 'computed_metrics':
            for metric_name in tDict_sequence['computed_metrics'].keys():
                max_val = max_values.get(metric_name)
                tDict_sequence[key][metric_name] = torch.clamp(tDict_sequence['computed_metrics'][metric_name], -max_val, max_val)/max_val  

        elif key == 'walls':
                tDict_sequence['walls']['x'] = torch.clamp(value['x'], -max_values['scale'], max_values['scale']) / max_values['scale']
                tDict_sequence['walls']['y'] = torch.clamp(value['y'], -max_values['scale'], max_values['scale']) / max_values['scale'] 

        elif key == 'context':
            for k in tDict_sequence['context'].keys():
                tDict_sequence['context'][k] = tDict_sequence['context'][k] / max_values['max_c']

    return tDict_sequence
