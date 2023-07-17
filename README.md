# SocNavGym : An environment for Social Navigation

## Table of Contents
1. [Description](https://github.com/gnns4hri/SocNavGym/tree/main#description)
2. [Installation](https://github.com/gnns4hri/SocNavGym/tree/main#dependencies)
3. [Usage](https://github.com/gnns4hri/SocNavGym/tree/main#usage)
4. [Sample Code](https://github.com/gnns4hri/SocNavGym/tree/main#sample-code)
5. [About the environment](https://github.com/gnns4hri/SocNavGym/tree/main#about-the-environment)
6. [Conventions](https://github.com/gnns4hri/SocNavGym/tree/main#conventions)
7. [Observation Space](https://github.com/gnns4hri/SocNavGym/tree/main#observation-space)
8. [Action Space](https://github.com/gnns4hri/SocNavGym/tree/main#action-space)
9. [Info Dict](https://github.com/gnns4hri/SocNavGym/tree/main#info-dict)
10. [Reward Function](https://github.com/gnns4hri/SocNavGym/tree/main#reward-function)
11. [Writing Custom Reward Functions](https://github.com/gnns4hri/SocNavGym/tree/main#writing-custom-reward-functions)
12. [Config File](https://github.com/gnns4hri/SocNavGym/tree/main#config-file)
13. [Wrappers](https://github.com/gnns4hri/SocNavGym/tree/main#wrappers)
14. [Training Agents](https://github.com/gnns4hri/SocNavGym/tree/main#training-agents)
15. [Evaluating Agents](https://github.com/gnns4hri/SocNavGym/tree/main#evaluating-agents)
16. [Manually Controlling the Robot](https://github.com/gnns4hri/SocNavGym/tree/main#manually-controlling-the-robot)
17. [Tutorials](https://github.com/gnns4hri/SocNavGym/tree/main#tutorials)


## Description
This repository contains the implementation of our paper "SocNavGym: A Reinforcement Learning Gym for Social Navigation", published in IEEE ROMAN, 2023. 

## Installation
1. Install Python-RVO2 by following the instructions given in [this](https://github.com/sybrenstuvel/Python-RVO2/) repository.
2. Install DGL (Deep Graph Library) for your system using [this](https://www.dgl.ai/pages/start.html) link.
3. For installing the environment using pip: 
    ```bash
    python3 -m pip install socnavgym
    ```

    For installing from source:
    ```bash
    git clone https://github.com/gnns4hri/SocNavGym.git
    python3 -m pip install .  # to install the environment to your Python libraries. This is optional. If you don't run this, then just make sure that your current working directory is the root of the repository when importing socnavgym.
    ```
4. The Deep RL agents are written using Stable Baselines3. We used the following command to install SB3 for our experiments 
    ```bash
    pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
    ```
    This is NOT a necessity for the environment to run. If you're going to use the `stable_dqn.py` we recommend installing by running the above command. Installing stable-baselines3 normally using pip might work, but we do not guarantee since it wasn't tested.

## Usage
```python
import socnavgym
import gym
env = gym.make('SocNavGym-v1', config="<PATH_TO_CONFIG>")  
```
## Sample Code
```python
import socnavgym
import gym
env = gym.make("SocNavGym-v1", config="./environment_configs/exp1_no_sngnn.yaml") 
obs, _ = env.reset()


for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        env.reset()
```

## About the environment
```SocNavGym-v1``` is a highly customisable environment and the parameters of the environment can be controlled using the config files. There are a few config files in [environment_configs/](https://github.com/gnns4hri/SocNavGym/tree/main/environment_configs). For a better understanding of each parameter refer to [this](https://github.com/gnns4hri/SocNavGym#config-file) section. Other than the robot, the environment supports entities like plants, tables, laptops. The environment also models interactions between humans, and human-laptop. It can also contain moving crowds, and static crowds, and the ability to form new crowds and interactions, as well as disperse existing crowds and interactions. The environment follows the OpenAI Gym format implementing the `step`, `render` and `reset` functions. The environment uses the latest Gym API (gym 0.26.2).

## Conventions
* X-axis points in the direction of zero-angle.
* The orientation of the entities the angle between the X-axis of the entity and the X-axis of the ground frame.
* Origin is at the center of the room

## Observation Space
The observation returned when ```env.step(action)``` is called, consists of the following (all in the<b> robot frame</b> unless you're using the [`WorldFrameObservations`](https://github.com/gnns4hri/SocNavGym#wrappers) wrapper):

The observation is of the type `gym.Spaces.Dict`. The dictionary has the following keys:
1. ```"robot"``` : This is a vector of shape (9,) of which the first six values represent the one-hot encoding of the robot, i.e ```[1, 0, 0, 0, 0, 0]```. The next two values represent the goal's x and y coordinates in the robot frame and the last value is the robot's radius.

2. The other keys present in the observation are ```"humans"```, ```"plants"```, ```"laptops"```, ```"tables"``` and ```"walls"```. Every entity (human, plant, laptop, table, or wall) would have an observation vector given by the structure below:
    <table  style=text-align:center>
        <tr>
            <th colspan="6"  style=text-align:center>Encoding</th>
            <th colspan="2" style=text-align:center>Relative Position Coordinates</th>
            <th colspan="2" style=text-align:center>Relative Orientation</th>
            <th style=text-align:center>Radius</th>
            <th colspan="2" style=text-align:center>Relative Speeds</th>
            <th style=text-align:center>Gaze</th>
        </tr>
        <tr>
            <td style=text-align:center>enc0</td>
            <td style=text-align:center>enc1</td>
            <td style=text-align:center>enc2</td>
            <td style=text-align:center>enc3</td>
            <td style=text-align:center>enc4</td>
            <td style=text-align:center>enc5</td>
            <td style=text-align:center>x</td>
            <td style=text-align:center>y</td>
            <td style=text-align:center>sin(theta)</th>
            <td style=text-align:center>cos(theta)</th>
            <td style=text-align:center>radius</td>
            <td style=text-align:center>relative speed</td>
            <td style=text-align:center>relative angular speed</td>
            <td style=text-align:center>gaze</td>
        </tr>
         <tr>
            <td style=text-align:center>0</td>
            <td style=text-align:center>1</td>
            <td style=text-align:center>2</td>
            <td style=text-align:center>3</td>
            <td style=text-align:center>4</td>
            <td style=text-align:center>5</td>
            <td style=text-align:center>6</td>
            <td style=text-align:center>7</td>
            <td style=text-align:center>8</td>
            <td style=text-align:center>9</td>
            <td style=text-align:center>10</td>
            <td style=text-align:center>11</td>
            <td style=text-align:center>12</td>
            <td style=text-align:center>13</td>
        </tr>
    </table>
    Details of the field values:
    
    * One hot encodings of the object.

        The one hot encodings are as follows:
        * human:  ```[0, 1, 0, 0, 0, 0]```
        * table: ```[0, 0, 1, 0, 0, 0]```
        * laptop: ```[0, 0, 0, 1, 0, 0]```
        * plant: ```[0, 0, 0, 0, 1, 0]```
        * wall: ```[0, 0, 0, 0, 0, 1]```

    * x, y coordinates relative to the robot. For rectangular shaped objects the coordinates would correspond to the center of geometry.

    * theta : The orientation with respect to the robot

    * radius: Radius of the object. Rectangular objects will contain the radius of the circle that circumscribes the rectangle

    * relative translational speed is the magnitude of relative velocity of the entity with respect to the robot

    * relative angular speed is calculated by the difference in the angles across two consecutive time steps and dividing by the time-step

    * gaze value: for humans, it is 1 if the robot lies in the line of sight of humans, otherwise 0. For entities other than humans, the gaze value is 0. Line of sight of the humans is decided by whether the robot lies from -gaze_angle/2 to +gaze_angle/2 in the human frame. Gaze angle can be changed by changing the `gaze_angle` parameter in the config file.

    The observation vector of the all entities of the same type would be concatenated into a single vector and that would be placed in the corresponding key in the dictionary. For example, let's say there are 4 humans, then the four vectors of shape (14,) would be concatenated to (56,) and the `"humans"` key in the observation dictionary would contain the vector of size (56, ). Individual observations can be accessed by simply reshaping the observation to (-1, 14).

    For walls, each wall is segmented into smaller walls of size `wall_segment_size` (can be found the config). Observations from each segment are returned in `obs["walls"]`
3. The observation space of the environment can be obtained by calling `env.observation_space`.  

## Action Space
The action space for holonomic robot consists of three components, vx, vy, and va. Here the X axis is the robot's heading direction. For differential drive robots, the component vy would be 0. You can control the type of the robot using the config file's `robot_type` parameter. All the three components take in a value between -1 and 1, which will be later mapped to the corresponding speed by using the maxima set in the config file. If you want to use a discrete action space, you could use the [`DiscreteActions`](https://github.com/gnns4hri/SocNavGym#wrappers) wrapper.

## Info Dict
The environment also returns meaningful metrics at every step in an episode. The following table describes each metric that is returned in the info dict.

<table  style=text-align:left>
    <tr>
        <th style=text-align:left>Metric</th>
        <th style=text-align:left>Description</th>
    </tr>
    <tr>
        <td style=text-align:left> "OUT_OF_MAP" </td>
        <td style=text-align:left> Boolean value that indicates whether the robot went out of the map</td>
    </tr>
    <tr>
        <td style=text-align:left> "COLLISION_HUMAN" </td>
        <td style=text-align:left> Boolean value that indicates whether the robot collided with a human</td>
    </tr>
    <tr>
        <td style=text-align:left> "COLLISION_OBJECT" </td>
        <td style=text-align:left> Boolean value that indicates whether the robot collided with an object</td>
    </tr>
    <tr>
        <td style=text-align:left> "COLLISION_WALL" </td>
        <td style=text-align:left> Boolean value that indicates whether the robot collided with a wall</td>
    </tr>
    <tr>
        <td style=text-align:left> "COLLISION" </td>
        <td style=text-align:left> Boolean value that indicates whether the robot collided with any entity</td>
    </tr>
    <tr>
        <td style=text-align:left> "SUCCESS" </td>
        <td style=text-align:left> Boolean value that indicates whether the robot reached the goal</td>
    </tr>
    <tr>
        <td style=text-align:left> "TIMEOUT" </td>
        <td style=text-align:left> Boolean value that indicates whether the episode has terminated due maximum steps</td>
    </tr>
    <tr>
        <td style=text-align:left> "FAILURE_TO_PROGRESS" </td>
        <td style=text-align:left> The number of timesteps that the robot failed to reduce the distance to goal</td>
    </tr>
    <tr>
        <td style=text-align:left> "STALLED_TIME" </td>
        <td style=text-align:left> The number of timesteps that the robot's velocity is 0</td>
    </tr>
    <tr>
        <td style=text-align:left> "TIME_TO_REACH_GOAL" </td>
        <td style=text-align:left> Number of time steps taken by the robot to reach its goal</td>
    </tr>
    <tr>
        <td style=text-align:left> "STL" </td>
        <td style=text-align:left> Success weighted by time length</td>
    </tr>
    <tr>
        <td style=text-align:left> "SPL" </td>
        <td style=text-align:left> Success weighted by path length</td>
    </tr>
    <tr>
        <td style=text-align:left> "PATH_LENGTH" </td>
        <td style=text-align:left> Total path length covered by the robot</td>
    </tr>
    <tr>
        <td style=text-align:left> "V_MIN" </td>
        <td style=text-align:left> Minimum velocity that the robot has achieved</td>
    </tr>
    <tr>
        <td style=text-align:left> "V_AVG" </td>
        <td style=text-align:left> Average velocity of the robot</td>
    </tr>
    <tr>
        <td style=text-align:left> "V_MAX" </td>
        <td style=text-align:left> Maximum velocity that robot has achieved</td>
    </tr>
    <tr>
        <td style=text-align:left> "A_MIN" </td>
        <td style=text-align:left> Minimum acceleration that the robot has achieved</td>
    </tr>
    <tr>
        <td style=text-align:left> "A_AVG" </td>
        <td style=text-align:left> Average acceleration of the robot</td>
    </tr>
    <tr>
        <td style=text-align:left> "A_MAX" </td>
        <td style=text-align:left> Maximum acceleration that robot has achieved</td>
    </tr>
    <tr>
        <td style=text-align:left> "JERK_MIN" </td>
        <td style=text-align:left> Minimum jerk that the robot has achieved</td>
    </tr>
    <tr>
        <td style=text-align:left> "JERK_AVG" </td>
        <td style=text-align:left> Average jerk of the robot</td>
    </tr>
    <tr>
        <td style=text-align:left> "JERK_MAX" </td>
        <td style=text-align:left> Maximum jerk that robot has achieved</td>
    </tr>
    <tr>
        <td style=text-align:left> "TIME_TO_COLLISION" </td>
        <td style=text-align:left>  Minimum time to collision with a human agent at any point in time in the trajectory, should all robots and humans move in a linear trajectory</td>
    </tr>
    <tr>
        <td style=text-align:left> "MINIMUM_DISTANCE_TO_HUMAN" </td>
        <td style=text-align:left>  Minimum distance to any human</td>
    </tr>
    <tr>
        <td style=text-align:left> "PERSONAL_SPACE_COMPLIANCE" </td>
        <td style=text-align:left>  Percentage of steps that the robot is not within the personal space (0.45m) of any human</td>
    </tr>
    <tr>
        <td style=text-align:left> "MINIMUM_OBSTACLE_DISTANCE" </td>
        <td style=text-align:left>  Minimum distance to any object </td>
    </tr>
    <tr>
        <td style=text-align:left> "AVERAGE_OBSTACLE_DISTANCE" </td>
        <td style=text-align:left>  Average distance to any object </td>
    </tr>
</table>


Some additional metrics that are also provided are :
<table style=text-align:left>
    <tr>
        <th style=text-align:left>Metric</th>
        <th style=text-align:left>Description</th>
    </tr>
    <tr>
        <td style=text-align:left> "DISCOMFORT_SNGNN" </td>
        <td style=text-align:left>  SNGNN_value (More about SNGNN in the section below) </td>
    </tr>
    <tr>
        <td style=text-align:left> "DISCOMFORT_DSRNN" </td>
        <td style=text-align:left>  DSRNN reward value (More about DSRNN reward function in the section below) </td>
    </tr>
    <tr>
        <td style=text-align:left> "sngnn_reward" </td>
        <td style=text-align:left>  SNGNN_value - 1 </td>
    </tr>
    <tr>
        <td style=text-align:left> "distance_reward" </td>
        <td style=text-align:left>  Value of the distance reward </td>
    </tr>
</table>

Note that the above 4 values are returned correctly if the reward function parameter in the config file is `"dsrnn"` or `"sngnn"`. If a custom reward function is written, then the user is required to fill the above values otherwise 0s would be returned for them. For more information refer to [Writing Custom Reward Functions](https://github.com/gnns4hri/SocNavGym/tree/main#writing-custom-reward-functions).
    
Lastly, information about the interactions is returned as an adjacency list. There are two types of interactions, `"human-human"` and `"human-laptop"`. For every interaction between human `i` and human `j` (`i` and `j` are the based on the order in which the human's observations appear in the observation dictionary. So to extract the `i`<sup>th</sup> human's observation, you could just do `obs["humans"].reshape(-1, 14)[i]`), the tuple `(i, j)` and `(j, i)` would be present in `info["interactions"]["human-human"]`, and similarly for an interaction between the `i`<sup>th</sup> human and the `j`<sup>th</sup> laptop, the tuple `(i, j)` would be present in `info["interactions"]["human-laptop"]`. Again, `j` is based on the order in which the laptops appear in the observation. To make it more clear, let's consider an example. Consider 4 humans, 1 table and two laptops in the environment and no walls. Also, two among the 4 humans are interacting with each other, and one other human is interacting with a laptop. For this scenario, the observation returned would be like this:
```python
obs_dict = {
    "humans": [obs_human0, obs_human1, obs_human2, obs_human3],  # human observations stacked in a 1D array
    "tables": [obs_table0],  # table observation
    "laptops": [obs_laptop0, obs_laptop1]  # laptop observations stacked in a 1D array.
}
```
Let's say the humans with observations `obs_human1` and `obs_human2` are the ones who are interacting. Also, the human whose observation is `obs_human3` interacts with the laptop which has an observation `obs_laptop1`. For such a case, the info dict for interactions would look like this:
```python
info = {
    "interactions": {
        "human-human": [(1, 2), (2, 1)],
        "human-laptop": [(3, 1)]
    }
    ...  # rest of the info dict
}
```

## Reward Function
The environment provides implementation of the [SNGNN](https://arxiv.org/abs/2102.08863) reward function, and the [DSRNN](https://arxiv.org/abs/2011.04820) reward function. If you want to use these reward functions, the config passed to the environment should have the value corresponding to the field `reward_file` as `"sngnn"` or `"dsrnn"` respectively. 


The environment also allows users to provide custom reward functions. Follow the guide below to create your own reward function.

## Writing Custom Reward Functions

1. Create a new python file in which you **have to** create a class named `Reward`. It **must** inherit from `RewardAPI` class. To do this, do the following
    ```python
    from socnavgym.envs.rewards import RewardAPI

    class Reward(RewardAPI):
        ...
    ```
2. Overwrite the function `compute_reward` with the custom reward function. The input of the `compute_reward` function is the action of the current timestep, the previous entity observations and the current entity observations. The previous and current observations are given as a dictionary with key as the id of the entity, and the value is an instance of the `EntityObs` namedtuple defined in [this](https://github.com/gnns4hri/SocNavGym/blob/main/socnavgym/envs/socnavenv_v1.py#L19) file. It contains the fields : id, x, y, theta, sin_theta, cos_theta for each entity in the environment. Note that all these values are in the robot's frame of reference. 
3. If need be, you can also access the lists of humans, plants, interactions etc, that the environment maintains by referencing the `self.env` variable. An example of this can be found in the [`dsrnn_reward.py`](https://github.com/gnns4hri/SocNavGym/blob/main/socnavgym/envs/rewards/dsrnn_reward.py#L24) file
4. The `RewardAPI` class provides four helper functions - `check_collision`, `check_timeout`, `check_reached` and `check_out_of_map`. These functions are boolean functions that check if the robot has collided with any enitity, whether the maximum episode length has been reached, whether the robot has reached the goal, or if the robot has moved out of the map respectively. The last case can occur only when the envirnoment is configured to have no walls.
5. The `RewardAPI` class also has a helper function defined to compute the SNGNN reward function. Call `compute_sngnn_reward(actions, prev_obs, curr_obs)` to compute the SNGNN reward. Also note that if you are using the SNGNN reward function in your custom reward function, please set the variable `self.use_sngnn` to `True`.
6. You can also store any additional information that needs to be returned in the info dict of step function by storing all of it in the variable `self.info` of the `Reward` class.
7. Storing anything in a class variable will persist across the steps in an episode. After every episode, the reward class object's `__init__()` method would be invoked.
8. Provide the path to the file where you defined your custom reward function in the config file's `reward_file`.


## Config File
The behaviour of the environment is controlled using the config file. The config file needs to be passed as a parameter while doing `gym.make`. These are the following parameters and their corresponding descriptions


<table  style=text-align:center>
    <tr>
        <th style=text-align:center> </th>
        <th style=text-align:center>Parameter</th>
        <th style=text-align:center>Description</th>
    </tr>
    <tr>
        <td rowspan="2"> rendering </td>
        <td> resolution_view </td>
        <td> size of the window for rendering the environment </td>
    </tr>
    <tr>
        <td> milliseconds </td>
        <td> delay parameter for waitKey()</td>
    </tr>
    <tr>
        <td rowspan="2"> episode </td>
        <td> episode_length </td>
        <td> maximum steps in an episode</td>
    </tr>
    <tr>
        <td> time_step </td>
        <td> number of seconds that one step corresponds to</td>
    </tr>
    <tr>
        <td rowspan="3"> robot </td>
        <td> robot_radius </td>
        <td> radius of the robot</td>
    </tr>
    <tr>
        <td> goal_radius </td>
        <td> radius of the robot's goal</td>
    </tr>
    <tr>
        <td> robot_type </td>
        <td> Accepted values are "diff-drive" (for differential drive robot) and "holonomic" (for holonomic robot)</td>
    </tr>
    <tr>
        <td rowspan="6"> human </td>
        <td> human_diameter </td>
        <td> diameter of the human</td>
    </tr>
    <tr>
        <td> human_goal_radius </td>
        <td> radius of human's goal</td>
    </tr>
    <tr>
        <td> human_policy </td>
        <td> policy of the human. Can be "random", "sfm", or "orca". If "random" is chosen, then one of "orca" or "sfm" would be randomly chosen</td>
    </tr>
    <tr>
        <td> gaze_angle </td>
        <td> gaze value (in the observation) for humans would be set to 1 when the robot lies between -gaze_angle/2 and +gaze_angle/2</td>
    </tr>
    <tr>
        <td> fov_angle </td>
        <td> the frame of view for humans</td>
    </tr>
    <tr>
        <td> prob_to_avoid_robot </td>
        <td> the probability that the human would consider the robot in its policy</td>
    </tr>
    <tr>
        <td rowspan="2"> laptop </td>
        <td> laptop_width </td>
        <td> width of laptops</td>
    </tr>
    <tr>
        <td> laptop_length </td>
        <td> length of laptops</td>
    </tr>
    <tr>
        <td rowspan="1"> plant </td>
        <td> plant_radius </td>
        <td> radius of plant<s/td>
    </tr>
     <tr>
        <td rowspan="2"> table </td>
        <td> table_width </td>
        <td> width of tables</td>
    </tr>
    <tr>
        <td> table_length </td>
        <td> length of tables</td>
    </tr>
    <tr>
        <td rowspan="1"> wall </td>
        <td> wall_thickness </td>
        <td> thickness of walls</td>
    </tr>
    <tr>
        <td rowspan="3"> human-human-interaction </td>
        <td> interaction_radius </td>
        <td> radius of the human-crowd</td>
    </tr>
    <tr>
        <td> interaction_goal_radius </td>
        <td> radius of the human-crowd's goal</td>
    </tr>
    <tr>
        <td> noise_varaince </td>
        <td> a random noise of normal(0, noise_variance) is applied to the humans' speed to break uniformity</td>
    </tr>
    <tr>
        <td rowspan="1"> human-laptop-interaction </td>
        <td> human_laptop_distance </td>
        <td> distance between human and laptop</td>
    </tr>
    <tr>
        <td rowspan="43"> env </td>
        <td> margin </td>
        <td> margin for the env </td>
    </tr>
    <tr>
        <td> max_advance_human </td>
        <td> maximum speed for humans </td>
    </tr>
    <tr>
        <td> max_advance_robot </td>
        <td> maximum linear speed for the robot </td>
    </tr>
    <tr>
        <td> max_rotation </td>
        <td> maximum rotational speed for robot </td>
    </tr>
    <tr>
        <td> wall_segment_size </td>
        <td> size of the wall segment, used when segmenting the wall </td>
    </tr>
    <tr>
        <td> speed_threshold </td>
        <td> speed below which would be considered 0 (for humans) </td>
    </tr>
    <tr>
        <td> crowd_dispersal_probability </td>
        <td> probability of crowd dispersal </td>
    </tr>
    <tr>
        <td> human_laptop_dispersal_probability </td>
        <td> probability to disperse a human-laptop-interaction </td>
    </tr>
    <tr>
        <td> crowd_formation_probability </td>
        <td> probability of crowd formation </td>
    </tr>
    <tr>
        <td> human_laptop_formation_probability </td>
        <td> probability to form a human-laptop-interaction </td>
    </tr>
    <tr>
        <td> reward_file </td>
        <td> Path to custom-reward file. If you want to use the in-built SNGNN reward function or the DSRNN reward function, set the value to "sngnn" or "dsrnn" respectively </td>
    </tr>
    <tr>
        <td> cuda_device </td>
        <td> cuda device to use (in case of multiple cuda devices). If cpu or only one cuda device, keep it as 0 </td>
    </tr>
    <tr>
        <td> min_static_humans </td>
        <td> minimum no. of static humans in the environment</td>
    </tr>
    <tr>
        <td> max_static_humans </td>
        <td> maximum no. of static humans in the environment</td>
    </tr>
    <tr>
        <td> min_dynamic_humans </td>
        <td> minimum no. of dynamic humans in the environment</td>
    </tr>
    <tr>
        <td> max_dynamic_humans </td>
        <td> maximum no. of dynamic humans in the environment</td>
    </tr>
    <tr>
        <td> min_tables </td>
        <td> minimum no. of tables in the environment</td>
    </tr>
    <tr>
        <td> max_tables </td>
        <td> maximum no. of tables in the environment</td>
    </tr>
    <tr>
        <td> min_plants </td>
        <td> minimum no. of plants in the environment</td>
    </tr>
    <tr>
        <td> max_plants </td>
        <td> maximum no. of plants in the environment</td>
    </tr>
    <tr>
        <td> min_laptops </td>
        <td> minimum no. of laptops in the environment</td>
    </tr>
    <tr>
        <td> max_laptops </td>
        <td> maximum no. of laptops in the environment</td>
    </tr>
    <tr>
        <td> min_h_h_dynamic_interactions </td>
        <td> minimum no. of dynamic human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_dynamic_interactions </td>
        <td> maximum no. of dynamic human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_h_dynamic_interactions_non_dispersing </td>
        <td> minimum no. of dynamic human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_dynamic_interactions_non_dispersing </td>
        <td> maximum no. of dynamic human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_h_static_interactions </td>
        <td> minimum no. of static human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_static_interactions </td>
        <td> maximum no. of static human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_h_static_interactions_non_dispersing </td>
        <td> minimum no. of static human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_static_interactions_non_dispersing </td>
        <td> maximum no. of static human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_human_in_h_h_interactions </td>
        <td> minimum no. of humans in a human-human interaction </td>
    </tr>
    <tr>
        <td> max_human_in_h_h_interactions </td>
        <td> maximum no. of humans in a human-human interaction </td>
    </tr>
    <tr>
        <td> min_h_l_interactions </td>
        <td> minimum no. of human-laptop interactions in the env. Note that these crowds can disperse if the parameter human_laptop_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_l_interactions </td>
        <td> maximum no. of human-laptop interactions in the env. Note that these crowds can disperse if the parameter human_laptop_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_l_interactions_non_dispersing </td>
        <td> minimum no. of human-laptop interactions in the env. Note that these crowds never disperse, even if the parameter human_laptop_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_l_interactions_non_dispersing </td>
        <td> maximum no. of human-laptop interactions in the env. Note that these crowds never disperse, even if the parameter human_laptop_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> get_padded_observations </td>
        <td> flag value that indicates whether you require padded observations or not. You can change it using env.set_padded_observations(True/False) </td>
    </tr>
    <tr>
        <td> set_shape </td>
        <td> Sets the shape of the environment. Accepted values are "random", "square", "rectangle", "L" or "no-walls"  </td>
    </tr>
    <tr>
        <td> add_corridors </td>
        <td> True or False, whether there should be corridors in the environment</td>
    </tr>
    <tr>
        <td> min_map_x </td>
        <td> minimum size of map along x direction </td>
    </tr>
    <tr>
        <td> max_map_x </td>
        <td> maximum size of map along x direction </td>
    </tr>
    <tr>
        <td> min_map_y </td>
        <td> minimum size of map along y direction </td>
    </tr>
    <tr>
        <td> max_map_y </td>
        <td> maximum size of map along y direction </td>
    </tr>
    
</table>

## Wrappers
Gym wrappers are convenient to have changes in the observation-space / action-space. SocNavGym implements 4 wrappers. 

The following are the wrappers implemented by SocNavGym:
1. `DiscreteActions` : To change the environment from a continuous action space environment to a discrete action space environment. The action space consists of 7 discrete actions. They are :
    * Turn anti-clockwise (0)
    * Turn clock-wise (1)
    * Turn anti-clockwise and moving forward (2)
    * Turning clockwise and moving forward (3)
    * Move forward (4)
    * Move backward (5)
    * Stay still (6)

    As an example, to make the robot move forward throughout the episode, just do the following:
    ```python
    import socnavgym
    from socnavgym.wrappers import DiscreteActions

    env = gym.make("SocNavGym-v1", config="environment_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = DiscreteActions(env)  # creates an env with discrete action space

    # simulate an episode with random actions
    done = False
    env.reset()
    while not done:
        obs, rew, terminated, truncated, info = env.step(4)  # 4 is for moving forward 
        done = terminated or truncated
        env.render()

    ```

2. `NoisyObservations` : This wrapper can be used to add noise to the observations so as to emulate real world sensor noise. The parameters that the wrapper takes in are `mean`, `std_dev`. Apart from this, there is also a parameter called `apply_noise_to` which defaults to `[robot", "humans", "tables", "laptops", "plants", "walls"]`, meaning all enitity types. If you want to apply noise to only a few entity types, then pass a list with only those entity types to this parameter. The noise value can be controlled using the `mean` and the `std_dev` parameters. Basically, a Gaussian noise with `mean` and `std_dev` is added to the observations of all the entities whose entity type is listed in the parameter `apply_noise_to`.
    As an example, to add a small noise with 0 mean and 0.1 std dev to all entity types do the following:
    ```python
    import socnavgym
    from socnavgym.wrappers import NoisyObservations

    env = gym.make("SocNavGym-v1", config="environment_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = NoisyObservations(env, mean=0, std_dev=0.1)

    # simulate an episode with random actions
    done = False
    env.reset()
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())  # obs would now be a noisy observation

        done = terminated or truncated
        env.render()

    ```

3. `PartialObservations` : This wrapper is used to return observations that are present in the frame of view of the robot, and also that lies within the range. Naturally, the wrapper takes in two parameters `fov_angle` and the `range`. 
    An example of using the `PartialObservations` wrapper:

    ```python
    import socnavgym
    from socnavgym.wrappers import PartialObservations
    from math import pi

    env = gym.make("SocNavGym-v1", config="environment_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = PartialObservations(env, fov_angle=2*pi/3, range=1)  # creates a robot with a 120 degreee frame of view, and the sensor range is 1m.

    # simulate an episode with random actions
    env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        env.render()

    ```
4. `WorldFrameObservations` : Returns all the observations in the world frame. The observation space of the `"robot"` would look like this:
    <table  style=text-align:center>
            <tr>
                <th colspan="6"  style=text-align:center>Encoding</th>
                <th colspan="2" style=text-align:center> Robot Goal coordinates</th>
                <th colspan="2" style=text-align:center> Robot coordinates</th>
                <th colspan="2" style=text-align:center> Angular Details</th>
                <th colspan="3" style=text-align:center>Velocities Speeds</th>
                <th style=text-align:center>Radius</th>
            </tr>
            <tr>
                <td style=text-align:center>enc0</td>
                <td style=text-align:center>enc1</td>
                <td style=text-align:center>enc2</td>
                <td style=text-align:center>enc3</td>
                <td style=text-align:center>enc4</td>
                <td style=text-align:center>enc5</td>
                <td style=text-align:center>goal_x</td>
                <td style=text-align:center>goal_y</td>
                <td style=text-align:center>x</td>
                <td style=text-align:center>y</td>
                <td style=text-align:center>sin(theta)</th>
                <td style=text-align:center>cos(theta)</th>
                <td style=text-align:center>vel_x</th>
                <td style=text-align:center>vel_y</th>
                <td style=text-align:center>vel_a</th>
                <td style=text-align:center>radius</td>
            </tr>
            <tr>
                <td style=text-align:center>0</td>
                <td style=text-align:center>1</td>
                <td style=text-align:center>2</td>
                <td style=text-align:center>3</td>
                <td style=text-align:center>4</td>
                <td style=text-align:center>5</td>
                <td style=text-align:center>6</td>
                <td style=text-align:center>7</td>
                <td style=text-align:center>8</td>
                <td style=text-align:center>9</td>
                <td style=text-align:center>10</td>
                <td style=text-align:center>11</td>
                <td style=text-align:center>12</td>
                <td style=text-align:center>13</td>
                <td style=text-align:center>14</td>
                <td style=text-align:center>15</td>
            </tr>
        </table>
    The other enitity observations would remain the same, the only difference being that the positions and velocities would be in the world frame of reference and not in the robot's frame of reference. 
    
    An example of using the `WorldFrameObservations` wrapper:

    ```python
    import socnavgym
    from socnavgym.wrappers import WorldFrameObservations
    from math import pi

    env = gym.make("SocNavGym-v1", config="environment_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = WorldFrameObservations(env) 

    # simulate an episode with random actions
    env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())  # obs contains observations that are in the world frame 
        done = terminated or truncated
        env.render()

    ```

## Training Agents
The script to train the agents is `stable_dqn.py`. This is an implementation of DuelingDQN using StableBaselines3. We use [Comet ML](https://www.comet.com/site/) for logging, so please create an account before proceeding. It is completely free of cost. Run the following commands to reproduce our results on the experiments mentioned in the paper:

1. Experiment 1 (Using DSRNN Reward)
    ```bash
    python3 stable_dqn.py -e="./environment_configs/exp1_no_sngnn.yaml" -r="dsrnn_exp1" -s="dsrnn_exp1" -d=False -p=<project_name> -a=<api_key>
    ```

2. Experiment 1 (Using SNGNN Reward)
    ```bash
    python3 stable_dqn.py -e="./environment_configs/exp1_with_sngnn.yaml" -r="sngnn_exp1" -s="sngnn_exp1" -d=False -p=<project_name> -a=<api_key>
    ```

3. Experiment 2 (Using DSRNN Reward)
    ```bash
    python3 stable_dqn.py -e="./environment_configs/exp2_no_sngnn.yaml" -r="dsrnn_exp2" -s="dsrnn_exp2" -d=False -p=<project_name> -a=<api_key>
    ```

4. Experiment 2 (Using SNGNN Reward)
    ```bash
    python3 stable_dqn.py -e="./environment_configs/exp2_with_sngnn.yaml" -r="sngnn_exp2" -s="sngnn_exp2" -d=False -p=<project_name> -a=<api_key>
    ```

5. Experiment 3 (Using DSRNN Reward)
    ```bash
    python3 stable_dqn.py -e="./environment_configs/exp3_no_sngnn.yaml" -r="dsrnn_exp3" -s="dsrnn_exp3" -d=False -p=<project_name> -a=<api_key>
    ```

6. Experiment 3 (Using SNGNN Reward)
    ```bash
    python3 stable_dqn.py -e="./environment_configs/exp3_with_sngnn.yaml" -r="sngnn_exp3" -s="sngnn_exp3" -d=False -p=<project_name> -a=<api_key>
    ```
In general, the `stable_dqn` script can be used as follows:
```bash
usage: python3 stable_dqn.py [-h] -e ENV_CONFIG -r RUN_NAME -s SAVE_PATH -p
                     PROJECT_NAME -a API_KEY [-d USE_DEEP_NET] [-g GPU]

optional arguments:
  -h, --help            show this help message and exit
  -e ENV_CONFIG, --env_config ENV_CONFIG
                        path to environment config
  -r RUN_NAME, --run_name RUN_NAME
                        name of comet_ml run
  -s SAVE_PATH, --save_path SAVE_PATH
                        path to save the model
  -p PROJECT_NAME, --project_name PROJECT_NAME
                        project name in comet ml
  -a API_KEY, --api_key API_KEY
                        api key to your comet ml profile
  -d USE_DEEP_NET, --use_deep_net USE_DEEP_NET
                        True or False, based on whether you want a transformer
                        based feature extractor
  -g GPU, --gpu GPU     gpu id to use

```

## Evaluating Agents
The evaluation script for the Dueling DQN agent using StableBaselines3 can be found in `sb3_eval.py`. 
```bash
usage: python3 sb3_eval.py [-h] -n NUM_EPISODES -w WEIGHT_PATH -c CONFIG

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_EPISODES, --num_episodes NUM_EPISODES
                        number of episodes
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        path to weight file
  -c CONFIG, --config CONFIG
                        path to config file
```

## Manually Controlling the Robot
You can control the robot using a joystick and also record observations, actions and rewards. To do this, run the `manual_control_js.py`.
```bash
usage: python3 manual_control_js.py [-h] -n NUM_EPISODES [-j JOYSTICK_ID] [-c CONFIG] [-r RECORD] [-s START]

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_EPISODES, --num_episodes NUM_EPISODES
                        number of episodes
  -j JOYSTICK_ID, --joystick_id JOYSTICK_ID
                        Joystick identifier
  -c CONFIG, --config CONFIG
                        Environment config file
  -r RECORD, --record RECORD
                        Whether you want to record the observations, and actions or not
  -s START, --start START
                        starting episode number

```
## Tutorials
1. [Installation tutorial](https://colab.research.google.com/drive/1vX1mxQJ82MQpHLg43Y2BArPYeDRsimse?usp=sharing)
2. [Training a Deep RL agent on SocNavGym](https://colab.research.google.com/drive/1NNt4k8Ouzs9vNj20AXsWfzwWXW8MG5QM?usp=sharing)