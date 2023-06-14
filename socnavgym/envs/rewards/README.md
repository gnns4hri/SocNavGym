# Writing Custom Reward Functions

## What should be done
1. Create a new python file in which you **have to** create a class named `Reward`. It **must** inherit from `RewardAPI` class defined in the file named `reward_api.py`. Refer to `sngnn_reward.py` or `dsrnn_reward.py` if you have any doubts. Place the file in `socnavenv/envs/rewards` directory.
2. Include the following two imports as is at the beginning of your python file containing the Reward class:

    ```python
    import socnavenv
    from socnavenv.envs.rewards.reward_api import RewardAPI
    ```
3. Overwrite the function `compute_reward` with the custom reward function. The input of the `compute_reward` function is the action of the current timestep, the previous entity observations and the current entity observations. The previous and current observations are given as a dictionary with key as the id of the entity, and the value is an instance of the `EntityObs` namedtuple defined in the file `socanvenv/envs/socnavenv_v1.py`. It contains the fields : id, x, y, theta, sin_theta, cos_theta for each entity in the environment. Note that all these values are in the robot's frame of reference. 
4. If need be, you can also access the lists of humans, plants, interactions etc, that the environment maintains by referencing the `self.env` variable. An example of this can be found in the `dsrnn_reward.py` file
5. The `RewardAPI` class provides helper functions such as `check_collision`, `check_timeout`, `check_reached` and `check_out_of_map`. These functions are boolean functions that check if the robot has collided with any enitity, whether the maximum episode length has been reached, whether the robot has reached the goal, or if the robot has moved out of the map respectively. The last case can occur only when the envirnoment is configured to have no walls.
6. The `RewardAPI` class also has a helper function defined to compute the SNGNN reward function. Call `compute_sngnn_reward(actions, prev_obs, curr_obs)` to compute the SNGNN reward. Also note that if you are using the SNGNN reward function in your custom reward function, please set the variable `self.use_sngnn` to `True`.
7. You can also store any additional information that needs to be returned in the info dict of step function by storing all of it in the variable `self.info` of the `Reward` class.
8. Storing anything in a class variable will persist across the steps in an episode. There will be a new instantiation of the Reward class object every episode.
