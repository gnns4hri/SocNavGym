"""
Hindsight Experience Replay (HER) wrapper for SocNavGym.

This wrapper modifies the environment to support goal-relabeling for HER.
It works by storing the original goals and allowing the replay buffer
to sample alternative goals for past experiences.
"""

import sys
import numpy as np
from typing import Dict, Any, Tuple, NamedTuple, Optional
import copy
import gymnasium as gym

import torch
from stable_baselines3.common.buffers import ReplayBufferSamples
import socnavgym
MINIMUM_TRANSITIONS = 3


def print_obs_transition(transition, text="transition_obs_pose"):
    obs = transition["obs"]
    print(f"{text}: {obs[0]}, {obs[1]}, {np.atan2(obs[3],obs[4])}  (a={np.atan2(obs[1], obs[0])} d={np.sqrt(obs[0]*obs[0] + obs[1]*obs[1])})")

class HERGoalEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds HER support to SocNavGym environment.
    
    This wrapper:
    1. Stores original goals for each episod
    2. Provides methods for goal relabeling
    3. Maintains compatibility with the original environment
    """

    replay_buffer = None
    active = True
    
    def __init__(self, env, her_config: Dict[str, Any]):
        """
        Initialize HER wrapper.
        
        Args:
            env: The base SocNavGym environment
            her_config: HER configuration dictionary
        """
        super().__init__(env)
        self.her_config = her_config
        
        # Store original observation space
        self.original_obs_space = self.env.observation_space
        
        # Episode tracking
        self.episode_transitions = []

        self.original_env = env
        self.base_env = env
        while hasattr(self.base_env, 'env'):
            self.base_env = self.base_env.env

        self.THIS_IS_THE_HER_WRAPPER = True


    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and store original goals.
        
        Returns:
            observation: Original observation
            info: Additional info including original goals
        """
        # Reset the base environment
        obs, info = self.env.reset(**kwargs)
        
        if self.her_config['enabled'] and HERGoalEnvWrapper.active is True:
            self.do_the_trick()

        # Clear episode transitions
        self.episode_transitions = []
        
        return obs, info

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Step the environment
        internal_state_in_step = self.base_env.robot
        # print("wrapper step, internal state:", internal_state_in_step)
        obs, reward, terminated, truncated, info = self.original_env.step(action)
        
        # Store transition for potential HER relabeling
        transition = {
            "obs": copy.deepcopy(obs),
            'action': copy.deepcopy(action),
            'reward': copy.deepcopy(reward),
            'next_obs': None,  # Will be filled later
            'done': terminated or truncated,
            'robot_internal_state': copy.deepcopy(internal_state_in_step),
            'info': copy.deepcopy(info)
        }
        self.episode_transitions.append(transition)
        return obs, reward, terminated, truncated, info
    


    def relabel_with_absolute_goal(self, transition: Dict[str, Any], goal_absolute_x_y_a: np.ndarray, last=False) -> Dict[str, Any]:
        """
        Relabel a transition with a new goal.
        
        Args:
            transition: Original transition
            goal_absolute_x_y_a: New absolute goal to use for relabeling.
            
        Returns:
            Relabeled transition with new goal and recalculated reward
        """

        def get_relative_frame_coordinates(robot, goal_absolute_x_y_a):
            def transformation_matrix(robot):
                assert(robot.x is not None and robot.y is not None and robot.orientation is not None), \
                    "Robot coordinates or orientation are None type"
                tm = np.zeros((3,3), dtype=np.float32)
                ang = robot.orientation
                tm[0,0] = np.cos(ang);  tm[0,1] = -np.sin(ang);  tm[0,2] = robot.x; \
                tm[1,0] = np.sin(ang);  tm[1,1] = +np.cos(ang);  tm[1,2] = robot.y; \
                                                                 tm[2,2] = 1
                return np.linalg.inv(tm)

            goal_homogeneous_coordinates = np.array([[goal_absolute_x_y_a[0], goal_absolute_x_y_a[1], 1]])
            coords_from_robot = (transformation_matrix(robot)@goal_homogeneous_coordinates.T).T
            return coords_from_robot[0, 0:2]

        assert(len(goal_absolute_x_y_a) == 3), f"goal_absolute_x_y_a's size must be 3 {len(goal_absolute_x_y_a)}"
        robot = transition["robot_internal_state"]
        new_goal_relative_coords = get_relative_frame_coordinates(robot, goal_absolute_x_y_a)
        rel_angle = goal_absolute_x_y_a[2] - robot.orientation

        # Create a copy of the transition
        relabeled = copy.deepcopy(transition)

        relabeled["obs"][0] = new_goal_relative_coords[0]
        relabeled["obs"][1] = new_goal_relative_coords[1]
        relabeled["obs"][3] = np.sin(rel_angle)
        relabeled["obs"][4] = np.cos(rel_angle)

        # Recalculate reward based on new goal
        relabeled['reward'] = self._calculate_her_reward(relabeled, last)
        # print("r", relabeled['reward'])
        
        # Mark as relabeled
        relabeled['relabeled'] = True
        
        return relabeled
    
    def _calculate_her_reward(self, relabeled: Dict[str, Any], last=False) -> float:
        """
        Calculate reward for HER relabeled goal.
        
        Args:
            relabeled: Relabeled transition containing the relabeled observation (including the goal)
            goal: New goal position
            
        Returns:
            Recalculated reward
        """
        def _check_out_of_map(env_robot):
            env = self.base_env
            return (env_robot.x > env.MAP_X) or (env_robot.x < -env.MAP_X) or (env_robot.y > env.MAP_Y) or (env_robot.y < -env.MAP_Y)

        def _check_reached_goal(robot):
            distance_to_goal = np.sqrt( (robot[0]**2) + (robot[1]**2) )
            angular_distance_to_goal = np.abs(np.atan2(robot[3], robot[4]))
            return distance_to_goal < self.base_env.GOAL_THRESHOLD and angular_distance_to_goal < self.base_env.GOAL_ORIENTATION_THRESHOLD

        def _check_collision(humans):
            for human in humans:
                distance_to_human = np.sqrt( (human[0]**2) + (human[1]**2) )
                if distance_to_human < 0.35:
                    return True
            return False

        def _check_timeout():
            return self.base_env.ticks > self.base_env.EPISODE_LENGTH

        obs = relabeled["obs"]
        robot = relabeled["obs"][0:8]
        humans = relabeled["obs"][7:].reshape(-1,8)
        robot_int = relabeled["robot_internal_state"]

        if _check_out_of_map(relabeled["robot_internal_state"]):
            return -5.
        elif _check_reached_goal(robot):
            return 10.
        elif _check_collision(humans):
            return -10.
        elif _check_timeout():
            return -8.

        return 0
    
    

    def do_the_trick(self):
        """
        Sample batch with HER transitions mixed in.
        """
        if not self.episode_transitions or len(self.episode_transitions) <= 2:
            return


        # Fill next observations
        for i, transition in enumerate(self.episode_transitions[:-1]): # Skip the last transition (no next state)
            self.episode_transitions[i]['next_obs'] = self.episode_transitions[i + 1]["obs"]
        self.episode_transitions[-1]['next_obs'] = self.episode_transitions[-1]["obs"]

        # Sample from any state in the episode and take it as the fake goal
        episode_indices = list(range(len(self.episode_transitions)))
        sample_idx = np.random.choice(episode_indices) + MINIMUM_TRANSITIONS
        sample_idx = min(len(episode_indices)-1, sample_idx)
        xv = self.episode_transitions[sample_idx]["robot_internal_state"].x
        yv = self.episode_transitions[sample_idx]["robot_internal_state"].y
        av = self.episode_transitions[sample_idx]["robot_internal_state"].orientation

        # Draw noise for the fake goal, so that it is not suspiciously perfect
        offset_in_range = (np.random.random()-0.5)*2 * self.base_env.GOAL_THRESHOLD
        angle_wrt_goal  = (np.random.random()-0.5)*2 * np.pi
        x_offset = offset_in_range*np.cos(angle_wrt_goal)
        y_offset = offset_in_range*np.sin(angle_wrt_goal)
        a_offset = (np.random.random()-0.5)*2 * self.base_env.GOAL_ORIENTATION_THRESHOLD

        # Add the noise to the fake goal
        xv += x_offset
        yv += y_offset
        av += a_offset
        sample_goal = [xv, yv, av]  # Goal in robot frame


        for i in range(0, sample_idx+1):
            transition = self.episode_transitions[i]
            internal = transition["robot_internal_state"]

        for i in range(0, sample_idx+1):
            relabeled = self.relabel_with_absolute_goal(self.episode_transitions[i], sample_goal, i==sample_idx)
            HERGoalEnvWrapper.replay_buffer.add(
                obs=torch.tensor(relabeled["obs"]).unsqueeze(0),
                next_obs=torch.tensor(relabeled["next_obs"]).unsqueeze(0),
                action=torch.tensor(relabeled['action']).unsqueeze(0),
                reward=torch.tensor([relabeled['reward']]),
                done=torch.tensor([relabeled['done']]),
                infos=[relabeled["info"]]
            )
        
