"""
Hindsight Experience Replay (HER) wrapper for SocNavGym.

This wrapper modifies the environment to support goal-relabeling for HER.
It works by storing the original goals and allowing the replay buffer
to sample alternative goals for past experiences.
"""

import numpy as np
from typing import Dict, Any, Tuple, NamedTuple, Optional
import copy
import gymnasium as gym

import torch
# Use ReplayBufferSamples from stable-baselines3
from stable_baselines3.common.buffers import ReplayBufferSamples


class HERGoalEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds HER support to SocNavGym environment.
    
    This wrapper:
    1. Stores original goals for each episod
    2. Provides methods for goal relabeling
    3. Maintains compatibility with the original environment
    """
    
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
        
        # Clear episode transitions
        self.episode_transitions = []
        
        return obs, info

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Step the environment
        obs, reward, terminated, truncated, info = self.original_env.step(action)
        
        # Store transition for potential HER relabeling
        transition = {
            "obs": copy.deepcopy(obs),
            'action': copy.deepcopy(action),
            'reward': reward,
            'next_obs': None,  # Will be filled later
            'done': terminated or truncated,
            'robot_internal_state': self.base_env.robot
        }
        self.episode_transitions.append(transition)

        return obs, reward, terminated, truncated, info
    

    def relabel_with_absolute_goal(self, transition: Dict[str, Any], goal_absolute_x_y_a: np.ndarray) -> Dict[str, Any]:
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
        relabeled['reward'] = self._calculate_her_reward(relabeled)
        
        # Mark as relabeled
        relabeled['relabeled'] = True
        
        return relabeled
    
    def _calculate_her_reward(self, relabeled: Dict[str, Any]) -> float:
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
            return (env.MAP_X/2 < env_robot.x) or (env_robot.x < -env.MAP_X/2) or (env.MAP_Y/2 < env_robot.y) or (env_robot.y < -env.MAP_Y/2)

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

        if _check_out_of_map(relabeled["robot_internal_state"]):
            return -5.
        elif _check_reached_goal(robot):
            return 10.
        elif _check_collision(humans):
            return -10.
        elif _check_timeout():
            return -8.

        return 0
    
    
    def __getattr__(self, name: str):
        """Delegate other attribute accesses to the base environment."""
        return getattr(self.original_env, name)


class HERReplayBufferWrapper:
    """
    Wrapper for replay buffer that adds HER sampling.
    """

    def __init__(self, base_buffer, her_wrapper: HERGoalEnvWrapper, her_config: Dict[str, Any]):
        self.base_buffer = base_buffer
        self.her_wrapper = her_wrapper
        self.her_config = her_config
        
        
    def add(self, *args, **kwargs):
        return self.base_buffer.add(*args, **kwargs)


    def __getattr__(self, name: str):
        """Delegate other attribute accesses to the base buffer."""
        return getattr(self.base_buffer, name)


    def sample(self, batch_size: int, env):
        """
        Sample batch with HER transitions mixed in.
        """
        def sample_her_transitions(n_samples: int) -> list:
            """
            Sample relabeled transitions using HER

            Args:
                n_samples: Number of samples to generate per transition
                
            Returns:
                List of relabeled transitions
            """
            if not self.her_wrapper.episode_transitions or len(self.her_wrapper.episode_transitions) < 2:
                return None
            her_transitions = []
            
            # Fill next observations
            for i, transition in enumerate(self.her_wrapper.episode_transitions[:-1]): # Skip the last transition (no next state)
                self.her_wrapper.episode_transitions[i]['next_obs'] = self.her_wrapper.episode_transitions[i + 1]["obs"]

            # Sample from any state in the episode
            episode_indices = list(range(len(self.her_wrapper.episode_transitions)))
            # print("sample_her_transitions", n_samples)
            for _ in range(n_samples):
                sample_idx = np.random.choice(episode_indices)
                xv = self.her_wrapper.episode_transitions[sample_idx]["obs"][0]
                yv = self.her_wrapper.episode_transitions[sample_idx]["obs"][1]
                sv = self.her_wrapper.episode_transitions[sample_idx]["obs"][3]
                cv = self.her_wrapper.episode_transitions[sample_idx]["obs"][4]
                av = np.atan2(sv, cv)
                sample_goal = [xv, yv, av]  # Goal in robot frame
                relabeled = self.her_wrapper.relabel_with_absolute_goal(transition, sample_goal)
                her_transitions.append(relabeled)
                        
            return her_transitions

        def _convert_her_to_replay_buffer_samples(her_transitions):
            """Convert HER transitions to ReplayBufferSamples format."""
            
            # Build observations by stacking all observation components
            observations_list = []
            next_observations_list = []
            
            for transition in her_transitions:
                obs = transition["obs"]
                next_obs = transition.get('next_obs', obs)  # Use same obs if next_obs not available

                observations_list.append(obs)
                next_observations_list.append(next_obs)
            
            # Stack all arrays and convert to tensors
            observations = torch.stack([torch.tensor(obs) for obs in observations_list])
            next_observations = torch.stack([torch.tensor(obs) for obs in next_observations_list])
            actions = torch.stack([torch.tensor(t['action'], dtype=torch.float32) for t in her_transitions[:batch_size]])
            rewards = torch.stack([torch.tensor(t['reward'], dtype=torch.float32) for t in her_transitions[:batch_size]])
            dones = torch.stack([torch.tensor(t['done'], dtype=torch.float32) for t in her_transitions[:batch_size]])
            
            
            # Ensure observations have the same shape as expected by the regular batch
            # If observations are 1D but should be 2D, reshape them
            if observations.ndim == 1:
                observations = observations.reshape(-1, 1)
            if next_observations.ndim == 1:
                next_observations = next_observations.reshape(-1, 1)
            
            # Create ReplayBufferSamples named tuple
            return ReplayBufferSamples(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
                dones=dones,
                discounts=None
            )

        def _combine_replay_buffer_samples(regular_batch, her_batch):
            """
                Combine two ReplayBufferSamples objects.

                This function is supposed to concatenate them, making sure they
                are correct in the target replay buffer format.
            """

            
            def ensure_tensor(array):
                """Ensure array is a torch tensor, convert if needed."""
                if isinstance(array, torch.Tensor):
                    return array
                else:
                    return torch.tensor(array, dtype=torch.float32)
            
            def concatenate_tensors(tensor1, tensor2):
                """Concatenate two tensors, handling different devices."""
                # Ensure both tensors are on the same device
                if tensor1.device != tensor2.device:
                    tensor2 = tensor2.to(tensor1.device)
                
                return torch.cat([tensor1, tensor2], dim=0)
            
            
            # Ensure all arrays are tensors
            regular_obs = ensure_tensor(regular_batch.observations)
            her_obs = ensure_tensor(her_batch.observations)
            regular_next_obs = ensure_tensor(regular_batch.next_observations)
            her_next_obs = ensure_tensor(her_batch.next_observations)
            regular_actions = ensure_tensor(regular_batch.actions)
            her_actions = ensure_tensor(her_batch.actions)
            regular_rewards = ensure_tensor(regular_batch.rewards)
            her_rewards = ensure_tensor(her_batch.rewards)
            regular_dones = ensure_tensor(regular_batch.dones)
            her_dones = ensure_tensor(her_batch.dones)
            
            # Concatenate all arrays using torch operations
            observations=concatenate_tensors(regular_obs, her_obs)
            next_observations=concatenate_tensors(regular_next_obs, her_next_obs)
            actions=concatenate_tensors(regular_actions, her_actions)



            her_rewards = her_rewards.unsqueeze(dim=1)
            her_dones = her_dones.unsqueeze(dim=1)
            rewards=concatenate_tensors(regular_rewards, her_rewards)
            dones=concatenate_tensors(regular_dones, her_dones)

            combined = ReplayBufferSamples(
                observations=concatenate_tensors(regular_obs, her_obs),
                next_observations=concatenate_tensors(regular_next_obs, her_next_obs),
                actions=concatenate_tensors(regular_actions, her_actions),
                rewards=concatenate_tensors(regular_rewards, her_rewards),
                dones=concatenate_tensors(regular_dones, her_dones),
                discounts=regular_batch.discounts  # Keep discounts from regular batch (should already be tensor)
            )

            return combined


        # print(f"SAMPLE\n{batch_size=}")
        # If HER is not enabled, delegate the task to the base_buffer instance
        if not self.her_config.get('enabled', False):
            regular_batch = self.base_buffer.sample(batch_size, env=env)
            # print(regular_batch)
            # print(type(regular_batch))
            # print("HER disabled, returning", len(regular_batch[0]), "samples directly from the environment")
            return regular_batch

        # Sample HER transitions
        her_ratio = self.her_config.get('ratio', 0.25)
        n_her_samples = int(batch_size * her_ratio)
        n_regular_samples = batch_size - n_her_samples
        her_transitions = sample_her_transitions(n_samples=n_her_samples)

        # If we don't have HER transitions, return the original, otherwise mix them with regular samples
        if her_transitions is None or not her_transitions or n_her_samples <= 0:
            regular_batch = self.base_buffer.sample(batch_size, env=env)
            # print("We don't yet have transitions, returning", len(regular_batch[0]), "samples directly from the environment")
            return regular_batch
        else:
            # print("transitions", len(her_transitions[0]), her_transitions[0].keys())
            # Sample fewer regular transitions
            regular_batch = self.base_buffer.sample(n_regular_samples, env=env)
          
            # Convert HER transitions to ReplayBufferSamples format
            her_batch = _convert_her_to_replay_buffer_samples(her_transitions)

            # print(f"{len(regular_batch[0])=}, {len(her_batch[0])=}")

            # Combine batches
            ret = _combine_replay_buffer_samples(regular_batch, her_batch)
            # print("ret:", len(ret[0]))
            return ret
        
