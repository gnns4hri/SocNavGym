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
                # print(f"C_dot_v :{C_dot_V}, f :{f}, V_sq :{V_sq}")
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
    metrics_sequence = {}

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
    metrics_sequence['dist_goal'] = dist_to_goal_pos
    metrics_sequence['success'] = success.float()

    hum_exists = (torch.sum(people['exists'], dim = 1)>0).float()
    metrics_sequence['hum_exists'] = hum_exists
    
    wall_exists = torch.tensor([torch.numel(walls['x'])>0]).repeat(robot['x'].shape).float().to(torch.float64)
    metrics_sequence['wall_exists'] = wall_exists

    if people['exists'].numel() > 0:
        dist_human = torch.where(people['exists'], metrics_ft['dist_human'], torch.inf)
        dist_nearest_hum = torch.min(dist_human, dim = 1).values
    else:
        dist_human = torch.tensor([torch.inf]).repeat((robot['x'].shape[0],1)).float().to(torch.float64)
        dist_nearest_hum = dist_human.squeeze()
    
    metrics_sequence['dist_nearest_hum'] = dist_nearest_hum

    obj_exists = (torch.sum(objects['exists'], dim = 1)>0).float()
    if objects['exists'].numel() > 0:
        dist_nearest_object = torch.min(torch.where(objects['exists'], metrics_ft['dist_object'], torch.inf), dim = 1).values
    else:
        dist_nearest_object = torch.tensor([torch.inf]).repeat(robot['x'].shape).float().to(torch.float64)
    metrics_sequence['dist_nearest_obj'] = dist_nearest_object

    inf_tensor = torch.full((wall_exists.shape[0], 1), torch.inf).to(torch.float64)

    dist_wall = torch.min(torch.cat((metrics_ft['dist_walls'], inf_tensor), dim=1), dim = 1).values
    metrics_sequence['dist_wall'] = dist_wall

    human_collision_flag = (dist_nearest_hum<=0.).float()
    metrics_sequence['hum_collision_flag'] = human_collision_flag

    object_collision_flag = (dist_nearest_object<=0.).float()
    metrics_sequence['object_collision_flag'] = object_collision_flag

    wall_collision_flag = (dist_wall<=0.).float()
    metrics_sequence['wall_collision_flag'] = wall_collision_flag

    social_space_intrusionA = (dist_nearest_hum<SOCIAL_SPACE_THRESHOLD).float()
    metrics_sequence['social_space_intrusionA'] = social_space_intrusionA
    num_near_humansA = torch.sum(dist_human<SOCIAL_SPACE_THRESHOLD, dim = 1).float()
    metrics_sequence['num_near_humansA'] = num_near_humansA
    metrics_sequence['num_near_humansA2'] = torch.pow(num_near_humansA, 2)

    social_space_intrusionB = (dist_nearest_hum<SOCIAL_SPACE_THRESHOLD*1.5).float()
    metrics_sequence['social_space_intrusionB'] = social_space_intrusionB
    num_near_humansB = torch.sum(dist_human<SOCIAL_SPACE_THRESHOLD*1.5, dim = 1).float()
    metrics_sequence['num_near_humansB'] = num_near_humansB
    metrics_sequence['num_near_humansB2'] = torch.pow(num_near_humansB, 2)

    social_space_intrusionC = (dist_nearest_hum<SOCIAL_SPACE_THRESHOLD*2.0).float()
    metrics_sequence['social_space_intrusionC'] = social_space_intrusionC
    num_near_humansC = torch.sum(dist_human<SOCIAL_SPACE_THRESHOLD*2.0, dim = 1).float()
    metrics_sequence['num_near_humansC'] = num_near_humansC
    metrics_sequence['num_near_humansC2'] = torch.pow(num_near_humansC, 2)

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

    metrics_sequence['min_time_to_collision'] = min_ttc
    metrics_sequence['min_time_to_collision2'] = torch.pow(min_ttc,2)
    metrics_sequence['max_fear'] = max_fear
    metrics_sequence['max_panic'] = max_panic

    global_dist_nearest_hum = torch.cummin(dist_nearest_hum, 0).values
    metrics_sequence['global_dist_nearest_hum'] = global_dist_nearest_hum

    acum_dist_travelled = torch.cumsum(robot['dist_travelled'], 0)
    initial_dist_to_goal = torch.tensor([dist_to_goal_pos[0]]).repeat(dist_to_goal_pos.shape).to(torch.float64)
    path_efficiency_ratio = torch.clamp(torch.div(initial_dist_to_goal, acum_dist_travelled), 0., 1.)
    metrics_sequence['path_efficiency_ratio'] = path_efficiency_ratio

    metrics_sequence['step_ratio'] = tDict_sequence['indices']/tDict_sequence['indices'][-1]
    metrics_sequence['episode_end'] = torch.zeros(dist_to_goal_pos.shape, dtype=torch.float64)
    metrics_sequence['episode_end'][-1] = 1.

    metrics_sequence['robot_x'] = robot['x']
    metrics_sequence['robot_y'] = robot['y']
    metrics_sequence['robot_a'] = robot['a']
    metrics_sequence['speed_x'] = robot['vx']
    metrics_sequence['speed_y'] = robot['vy']
    metrics_sequence['speed_a'] = robot['va']

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
    metrics_sequence['acceleration_x'] = (robot['vx']-prev_speed_x)/diff_time
    metrics_sequence['acceleration_y'] = (robot['vy']-prev_speed_y)/diff_time

    metrics_sequence['goal_pos_threshold'] = goal['th_p']
    metrics_sequence['goal_angle_threshold'] = goal['th_a']

    for var in tDict_sequence['context']:
        metrics_sequence[var] = tDict_sequence['context'][var]

    return metrics_sequence

def normalize_and_cat_features(metrics_sequence, max_metrics, features):
    metrics_tensor_list = []

    for f in features:
        ft_tensor = metrics_sequence[f]
        ft_max = max_metrics[f]
        ft_tensor_norm = torch.clamp(ft_tensor, -ft_max, ft_max)/ft_max
        metrics_tensor_list.append(torch.unsqueeze(ft_tensor_norm, dim=1))

    final_tensor = torch.cat(metrics_tensor_list, dim=1)

    return final_tensor
