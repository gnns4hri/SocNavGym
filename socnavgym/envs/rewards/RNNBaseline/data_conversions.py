import math
import torch
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),'../../baseline'))
import metrics

    
def sequence_to_tensor(data, frame_threshold, context):
    sequence = data['sequence']
    last_i = len(sequence)-1
    robot = {'x':[], 'y':[], 'a':[], 'vx':[], 'vy':[], 'va':[]}
    goal = {'x':[], 'y':[], 'a':[], 'th_p':[], 'th_a':[]}
    people = {'x':[], 'y':[], 'a':[]}
    objects = {'x':[], 'y':[], 'a':[], 'w':[], 'l':[]}
    metrics_ft = {'dist_human':[], 'ttc':[], 'panic':[], 'fear':[], 'dist_object':[],
                'dist_walls':[]}
    filtered_sequence = []
    last_timestamp = -float('inf')
    prev_index = 0
    for i, frame in enumerate(sequence):
        current_timestamp = frame['timestamp']
        if current_timestamp-last_timestamp >= frame_threshold or i==last_i:
            if i==0:
                inm_prev_frame = frame
                inm_prev_timestamp = -float('inf')
            else:
                inm_prev_frame = sequence[i-1]
                inm_prev_timestamp = sequence[i-1]['timestamp']
            prev_frame = sequence[prev_index]
            prev_index = i
            people_list = {}
            people_list['id'] = [p['id'] for p in frame['people']]            
            people_list['x'] = [p['x'] for p in frame['people']]
            people_list['y'] = [p['y'] for p in frame['people']]
            people_list['a'] = [p['angle'] for p in frame['people']]
            objects_list = {}
            objects_list['x'] = [o['x'] for o in frame['objects']]
            objects_list['y'] = [o['y'] for o in frame['objects']]
            objects_list['a'] = [o['angle'] for o in frame['objects']]
            objects_list['w'] = [o['shape']['width'] for o in frame['objects']]
            objects_list['l'] = [o['shape']['length'] for o in frame['objects']]
            objects_list['shape'] = [o['shape']['type'] for o in frame['objects']]
            objects_list['type'] = [o['type'] for o in frame['objects']]
            r_x, r_y = frame['robot']['x'], frame['robot']['y']
            g_x, g_y = frame['goal']['x'], frame['goal']['y']
            cur_ttc = metrics.get_ttc(frame, prev_frame)
            d_humans = metrics.dist_to_humans(frame)
            d_objects = metrics.dist_to_objects(frame)
            d_walls = metrics.dist_to_walls(frame, data['walls'])
            frame['people_list'] = people_list
            frame['objects_list'] = objects_list
            frame['ttc'] = [m['ttc'] for m in cur_ttc]
            frame['panic'] = [m['panic'] for m in cur_ttc]
            frame['fear'] = [m['fear'] for m in cur_ttc]
            frame['dist_human'] = d_humans
            frame['dist_object'] = d_objects
            frame['dist_wall'] = d_walls
            diff_time = current_timestamp-inm_prev_timestamp
            diff_angle = frame['robot']['angle'] - inm_prev_frame['robot']['angle']
            frame['robot_vx'] = (r_x - inm_prev_frame['robot']['x'])/diff_time
            frame['robot_vy'] = (r_y - inm_prev_frame['robot']['y'])/diff_time
            frame['robot_va'] = (math.atan2(math.sin(diff_angle), math.cos(diff_angle)))/diff_time
            frame['dist_travelled'] = math.sqrt((r_x - prev_frame['robot']['x'])**2 +
                                            (r_y - prev_frame['robot']['y'])**2)
            frame['index'] = i
            last_timestamp = current_timestamp
            filtered_sequence.append(frame)

    timestamp = torch.tensor([f['timestamp'] for f in filtered_sequence], dtype=torch.float64)
    indices = torch.tensor([f['index'] for f in filtered_sequence], dtype=torch.float64)
    robot['x'] = torch.tensor([f['robot']['x'] for f in filtered_sequence]).to(torch.float64)
    robot['y'] = torch.tensor([f['robot']['y'] for f in filtered_sequence]).to(torch.float64)
    robot['a'] = torch.tensor([f['robot']['angle'] for f in filtered_sequence]).to(torch.float64)
    robot['vx'] = torch.tensor([f['robot_vx'] for f in filtered_sequence]).to(torch.float64)
    robot['vy'] = torch.tensor([f['robot_vy'] for f in filtered_sequence]).to(torch.float64)
    robot['va'] = torch.tensor([f['robot_va'] for f in filtered_sequence]).to(torch.float64)
    robot['shape'] = filtered_sequence[0]['robot']['shape']
    robot['dist_travelled'] = torch.tensor([f['dist_travelled'] for f in filtered_sequence]).to(torch.float64)
    goal['x'] = torch.tensor([f['goal']['x'] for f in filtered_sequence]).to(torch.float64)
    goal['y'] = torch.tensor([f['goal']['y'] for f in filtered_sequence]).to(torch.float64)
    goal['a'] = torch.tensor([f['goal']['angle'] for f in filtered_sequence]).to(torch.float64)
    goal['th_p'] = torch.tensor([f['goal']['pos_threshold']+0.1 for f in filtered_sequence]).to(torch.float64)
    goal['th_a'] = torch.tensor([f['goal']['angle_threshold'] for f in filtered_sequence]).to(torch.float64)

    max_people = max(len(frame['people_list']['x']) for frame in filtered_sequence)
    pmask_list = []
    id_list = []
    x_list = []
    y_list = []
    a_list = []
    drobot_list = []
    ttc_list = []
    panic_list = []
    fear_list = []
    for f in filtered_sequence:
        people_id = f['people_list']['id']
        people_x = f['people_list']['x']
        people_y = f['people_list']['y']
        people_a = f['people_list']['a']
        people_dist = f['dist_human']
        ttc = f['ttc']
        panic = f['panic']
        fear = f['fear']
        n_people = len(people_x)
        pmask_list.append(torch.tensor([True]*n_people + [False]*(max_people-n_people)))
        if n_people < max_people:
            people_id += [0.]* (max_people - n_people)
            people_x += [0.]* (max_people - n_people)
            people_y += [0.]* (max_people - n_people)
            people_a += [0.]* (max_people - n_people)
            people_dist += [0.]* (max_people - n_people)
            ttc += [0.]* (max_people - n_people)
            panic += [0.]* (max_people - n_people)
            fear += [0.]* (max_people - n_people)
        id_list.append(people_id)
        x_list.append(torch.tensor(people_x).to(torch.float64))
        y_list.append(torch.tensor(people_y).to(torch.float64))
        a_list.append(torch.tensor(people_a).to(torch.float64))
        drobot_list.append(torch.tensor(people_dist).to(torch.float64))
        ttc_list.append(torch.tensor(ttc).to(torch.float64))
        panic_list.append(torch.tensor(panic).to(torch.float64))
        fear_list.append(torch.tensor(fear).to(torch.float64))
    people['id'] = id_list
    people['x'] = torch.stack(x_list, 0)
    people['y'] = torch.stack(y_list, 0)
    people['a'] = torch.stack(a_list, 0)        
    people['exists'] = torch.stack(pmask_list, 0)
    metrics_ft['dist_human'] = torch.stack(drobot_list, 0)
    metrics_ft['ttc'] = torch.stack(ttc_list, 0)
    metrics_ft['panic'] = torch.stack(panic_list, 0)
    metrics_ft['fear'] = torch.stack(fear_list, 0)

    max_objects = max(len(frame['objects_list']['x']) for frame in filtered_sequence)
    omask_list = []
    x_list = []
    y_list = []
    a_list = []
    w_list = []
    l_list = []
    shape_list = []
    type_list = []
    drobot_list = []
    for f in filtered_sequence:
        objects_x = f['objects_list']['x']
        objects_y = f['objects_list']['y']
        objects_a = f['objects_list']['a']
        objects_w = f['objects_list']['w']
        objects_l = f['objects_list']['l']
        objects_shape = f['objects_list']['shape']
        objects_type = f['objects_list']['type']
        object_dist = f['dist_object']
        n_objects = len(objects_x)
        omask_list.append(torch.tensor([True]*n_objects + [False]*(max_objects-n_objects)))
        if n_objects < max_objects:
            objects_x += [0.]* (max_objects - n_objects)
            objects_y += [0.]* (max_objects - n_objects)
            objects_a += [0.]* (max_objects - n_objects)
            objects_w += [0.]* (max_objects - n_objects)
            objects_l += [0.]* (max_objects - n_objects)
            objects_shape += ['none']* (max_objects - n_objects)
            objects_type += ['none']* (max_objects - n_objects)
            object_dist += [0.]* (max_objects - n_objects)

        x_list.append(torch.tensor(objects_x).to(torch.float64))
        y_list.append(torch.tensor(objects_y).to(torch.float64))
        a_list.append(torch.tensor(objects_a).to(torch.float64))
        w_list.append(torch.tensor(objects_w).to(torch.float64))
        l_list.append(torch.tensor(objects_l).to(torch.float64))
        shape_list.append(objects_shape)
        type_list.append(objects_type)
        drobot_list.append(torch.tensor(object_dist).to(torch.float64))

    objects['x'] = torch.stack(x_list, 0)
    objects['y'] = torch.stack(y_list, 0)
    objects['a'] = torch.stack(a_list, 0)        
    objects['w'] = torch.stack(w_list, 0)
    objects['l'] = torch.stack(l_list, 0)     
    objects['shape'] = shape_list
    objects['type'] = type_list
    objects['exists'] = torch.stack(omask_list)   
    metrics_ft['dist_object'] = torch.stack(drobot_list)
    wallsX_list = []
    wallsY_list = []
    for w in data['walls']:
        wallsX_list.append(w[0])
        wallsX_list.append(w[2])
        wallsY_list.append(w[1])
        wallsY_list.append(w[3])
    walls_x = torch.tensor(wallsX_list).to(torch.float64)
    walls_y = torch.tensor(wallsY_list).to(torch.float64)

    walls = {'x':walls_x, 'y':walls_y}        
    drobot_list = []
    for frame in filtered_sequence:
        d_walls = torch.tensor(frame['dist_wall']).to(torch.float64)
        drobot_list.append(d_walls)
    metrics_ft['dist_walls'] = torch.stack(drobot_list)

    context_ft = {}
    for var in context:
        context_ft[var] = torch.tensor([context[var]]).repeat(robot['x'].shape).float().to(torch.float64)


    tensor_dict = { 'timestamp': timestamp,
                    'indices': indices,
                    'robot': robot,
                    'goal': goal,
                    'people': people,
                    'objects': objects,
                    'walls': walls,
                    'metrics': metrics_ft,
                    'context': context_ft}
   
    return tensor_dict

def tensor_to_sequence(tDict_sequence):
	tDict_sequence_copy = clone_sequence(tDict_sequence)
	sequence = []
	timestamp = tDict_sequence_copy['timestamp'].tolist()
	robot = tDict_sequence_copy['robot']
	robot['x'] = robot['x'].tolist()
	robot['y'] = robot['y'].tolist()
	robot['a'] = robot['a'].tolist()    
	robot['vx'] = robot['vx'].tolist()
	robot['vy'] = robot['vy'].tolist()
	robot['va'] = robot['va'].tolist()    
	robot_shape = robot['shape']

	goal = tDict_sequence_copy['goal']
	goal['x'] = goal['x'].tolist()
	goal['y'] = goal['y'].tolist()
	goal['a'] = goal['a'].tolist()    
	goal['th_p'] = goal['th_p'].tolist()    
	goal['th_a'] = goal['th_a'].tolist()    

	people = tDict_sequence_copy['people']
	people['x'] = people['x'].tolist()
	people['y'] = people['y'].tolist()
	people['a'] = people['a'].tolist()    
	people['exists'] = people['exists'].tolist()    

	objects = tDict_sequence_copy['objects']    
	objects['x'] = objects['x'].tolist()
	objects['y'] = objects['y'].tolist()
	objects['a'] = objects['a'].tolist()    
	objects['w'] = objects['w'].tolist()
	objects['l'] = objects['l'].tolist()    
	objects['exists'] = objects['exists'].tolist()    

	for s in range(len(timestamp)):
		frame = {}
		frame['timestamp'] = timestamp[s]
		frame['robot'] = {'x': robot['x'][s], 'y': robot['y'][s],
                          'angle': robot['a'][s], 'speed_x': robot['vx'][s],
                          'speed_y': robot['vy'][s], 'speed_a': robot['va'][s],
                          'shape': robot_shape}
		frame['goal'] = {'x': goal['x'][s], 'y': goal['y'][s],
                          'angle': goal['a'][s], 'pos_threshold': goal['th_p'][s],
                          'angle_threshold': goal['th_a'][s]}
		people_list = []
		for p, exists in enumerate(people['exists'][s]):
			if exists:
				people_list.append({'id':people['id'][s][p], 'x': people['x'][s][p],
                                    'y': people['y'][s][p],  'angle': people['a'][s][p]})
		frame['people'] = people_list

		objects_list = []
		for o, exists in enumerate(objects['exists'][s]):
			if exists:
				object_shape = {'type':objects['shape'][s][o], 'width': objects['w'][s][o],
                                'length': objects['l'][s][o]}
				objects_list.append({'type':objects['type'][s][o], 'x': objects['x'][s][o],
                                     'y': objects['y'][s][o], 'angle': objects['a'][s][o],
                                     'shape': object_shape})
		frame['objects'] = objects_list

		sequence.append(frame)

	walls_x = tDict_sequence_copy['walls']['x'].tolist()
	walls_y = tDict_sequence_copy['walls']['y'].tolist()
	walls = []
	for i_w in range(0, len(walls_x), 2):
		wall=[]
		wall.append(walls_x[i_w])
		wall.append(walls_y[i_w])
		wall.append(walls_x[i_w+1])
		wall.append(walls_y[i_w+1])
		walls.append(wall)

	final_data = {'sequence':sequence, 'walls':walls}

	return final_data

def clone_sequence(tDict_sequence):
    new_tDict_sequence = {}


    for k in tDict_sequence:
        if type(tDict_sequence[k]) is torch.Tensor: #not dict:
            new_tDict_sequence[k] = tDict_sequence[k].clone()
        elif type(tDict_sequence[k]) is dict:
            dict_vars = {}
            for t in tDict_sequence[k]:
                if type(tDict_sequence[k][t]) is torch.Tensor:
                    dict_vars[t] = tDict_sequence[k][t].clone()
                else:
                    dict_vars[t] = copy.deepcopy(tDict_sequence[k][t])
            new_tDict_sequence[k] = dict_vars

    return new_tDict_sequence
