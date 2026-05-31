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
        
        # Add goal information to observations
        self._extend_observation_space()
        
        # Episode tracking
        self.episode_transitions = []

        self.original_env = env
        self.base_env = env
        while hasattr(self.base_env, 'env'):
            self.base_env = self.base_env.env

    def _extend_observation_space(self):
        """Extend observation space to include goal information."""
        # For SocNavGym, we'll add goal information to the robot observation
        # The original observation structure is preserved
        pass
    
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    
    def _update_goal_in_observation(self, obs: Dict[str, Any], goal: np.ndarray) -> Dict[str, Any]:
        """
        Update goal information in observation.
        
        Args:
            obs: Original observation
            goal: Goal pose must be in the frame of reference of the robot [x, y, a]

        Returns:
            Observation with added goal information
        """
        # For SocNavGym v2, we can add goal info to the robot observation
        # The robot observation already contains goal information in the elements 0, 1, 3, and 4,
        # but we need to overwrite it
        obs_with_goal = copy.deepcopy(obs)
        obs_with_goal['robot'][0] = goal[0]
        obs_with_goal['robot'][1] = goal[1]
        obs_with_goal['robot'][3] = sin(goal[2])
        obs_with_goal['robot'][4] = cos(goal[2])
       
        return obs_with_goal
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Step the environment and store transition for potential HER relabeling.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Step the environment
        obs, reward, terminated, truncated, info = self.original_env.step(action)
        
        

        # Store transition for potential HER relabeling
        transition = {
            'obs': copy.deepcopy(obs),
            'action': copy.deepcopy(action),
            'reward': reward,
            'next_obs': None,  # Will be filled later
            'done': terminated or truncated,
            'robot_internal_state': self.base_env.robot
        }
        self.episode_transitions.append(transition)
        
        return obs, reward, terminated, truncated, info
    
    def relabel_goal(self, transition: Dict[str, Any], x_y_a: np.ndarray) -> Dict[str, Any]:
        """
        Relabel a transition with a new goal.
        
        Args:
            transition: Original transition
            x_y_a: New goal to use for relabeling. In this case, it must be in 
            
        Returns:
            Relabeled transition with new goal and recalculated reward
        """

        def get_relative_frame_coordinates(robot, x_y_a):
            def transformation_matrix(robot):
                assert(robot.x is not None and robot.y is not None and robot.orientation is not None), \
                    "Robot coordinates or orientation are None type"
                tm = np.zeros((3,3), dtype=np.float32)
                ang = robot.orientation
                tm[0,0] = np.cos(ang);  tm[0,1] = -np.sin(ang);  tm[0,2] = robot.x; \
                tm[1,0] = np.sin(ang);  tm[1,1] = +np.cos(ang);  tm[1,2] = robot.y; \
                                                                 tm[2,2] = 1
                return np.linalg.inv(tm)

            goal_homogeneous_coordinates = np.array([[x_y_a[0], x_y_a[1], 1]])
            coords_from_robot = (transformation_matrix(robot)@goal_homogeneous_coordinates.T).T
            return coords_from_robot[:, 0:2]

        assert(len(x_y_a) == 3), f"x_y_a's size must be 3 {len(x_y_a)}"
        robot = transition["robot_internal_state"]
        new_goal_relative_coords = get_relative_frame_coordinates(robot, x_y_a)[0]
        rel_angle = x_y_a[2] - robot.orientation
        s_c = np.array([np.sin(rel_angle), np.cos(rel_angle)])


        # Create a copy of the transition
        relabeled = copy.deepcopy(transition)

        relabeled['obs'][0] = new_goal_relative_coords[0]
        relabeled['obs'][1] = new_goal_relative_coords[1]
        relabeled['obs'][3] = s_c[0]
        relabeled['obs'][4] = s_c[1]

        # Recalculate reward based on new goal
        # For navigation tasks, reward is typically distance-based
        relabeled['reward'] = self._calculate_her_reward(relabeled, x_y_a)
        
        # Mark as relabeled
        relabeled['relabeled'] = True
        
        return relabeled
    
    def _calculate_her_reward(self, relabeled: Dict[str, Any], goal: np.ndarray) -> float:
        """
        Calculate reward for HER relabeled goal.
        
        Args:
            obs: Observation containing robot position
            goal: New goal position
            
        Returns:
            Recalculated reward
        """
        obs = relabeled["obs"]

        # Extract robot data from observation
        # In SocNavGym v2, robot observation contains goal coordinates in robot frame
        robot_obs = obs[0:8]
        
        # The first 2 elements are goal coordinates in robot frame
        # For simple distance-based reward:
        distance_to_goal = np.linalg.norm(robot_obs[:2])
        
        # Simple reward: negative distance (encourages getting closer to goal)
        reward = 0
        
        # Add bonus for reaching goal
        # Access GOAL_THRESHOLD through the base environment
        if distance_to_goal < self.base_env.GOAL_THRESHOLD:
            reward += 10.0  # Bonus for reaching goal
            
        return reward
    
    def sample_her_transitions(self, strategies: str = None, n_samples: int = 4) -> list:
        """
        Sample relabeled transitions using HER strategies.
        
        Args:
            strategies: HER strategies, a list containing 'final' and/or 'episode'.
            n_samples: Number of samples to generate per transition
            
        Returns:
            List of relabeled transitions
        """
        if not self.episode_transitions or len(self.episode_transitions) < 2:
            return []

        if strategies is None:
            strategies = ["final", "episode"]
        her_transitions = []
        

        # Fill next observations
        for i, transition in enumerate(self.episode_transitions[:-1]): # Skip the last transition (no next state)
            self.episode_transitions[i]['next_obs'] = self.episode_transitions[i + 1]['obs']


        if "final" in strategies:
            # Get the final state of the episode
            final_state = self.episode_transitions[-1]['next_obs']
            # Use the final state goal
            if final_state is not None:
                n_samples -= 1  # We decrease the number of samples to generate if we generate one from the final one
                final_goal = final_state['robot'][:2]  # Goal in robot frame
                final_angle = np.atan2(*final_state['robot'][3:5])
                x_y_a = [final_goal[0], final_goal[1], final_angle]
                relabeled = self.relabel_goal(transition, x_y_a)
                her_transitions.append(relabeled)

        if "episode" in strategies:
            # Sample from any state in the episode
            episode_indices = list(range(len(self.episode_transitions)))
            for _ in range(n_samples):
                sample_idx = np.random.choice(episode_indices)
                xv = self.episode_transitions[sample_idx]['obs'][0]
                yv = self.episode_transitions[sample_idx]['obs'][1]
                sv = self.episode_transitions[sample_idx]['obs'][3]
                cv = self.episode_transitions[sample_idx]['obs'][4]
                av = np.atan2(sv, cv)
                sample_goal = [xv, yv, av]  # Goal in robot frame
                relabeled = self.relabel_goal(transition, sample_goal)
                her_transitions.append(relabeled)
                    
        return her_transitions
    
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
        """Add transition to buffer."""
        return self.base_buffer.add(*args, **kwargs)
        
    def sample(self, batch_size: int, env=None):
        """
        Sample batch with HER transitions mixed in.
        """
        # Sample regular transitions
        regular_batch = self.base_buffer.sample(batch_size, env=env)
        
        # If HER is not enabled, just return regular batch
        if not self.her_config.get('enabled', False):
            return regular_batch
            
        # Sample HER transitions
        her_ratio = self.her_config.get('ratio', 0.8)
        n_her_samples = int(batch_size * her_ratio)
        n_regular_samples = batch_size - n_her_samples
        
        her_transitions = self.her_wrapper.sample_her_transitions(
            strategies=self.her_config.get('strategies', ["final", "episode"]),
            n_samples=self.her_config.get('n_sampled_goal', 4)
        )
        
        # If we have HER transitions, mix them with regular samples
        if her_transitions and n_her_samples > 0:
            # Sample fewer regular transitions
            regular_batch = self.base_buffer.sample(n_regular_samples, env=env)
            

            
            # Debug: Print regular batch information
            print(f"DEBUG: Regular batch observations shape: {regular_batch.observations.shape}")
            print(f"DEBUG: Regular batch observations dtype: {regular_batch.observations.dtype}")
            if regular_batch.observations.size()[0] > 0:
                print(f"DEBUG: Regular batch first observation sample: {regular_batch.observations[0][:5]}..." if regular_batch.observations.ndim > 1 else f"DEBUG: Regular batch first observation: {regular_batch.observations[0]}")
            
            # Convert regular batch to numpy arrays to handle CUDA tensors
            regular_batch = self._convert_batch_to_numpy(regular_batch)
            
            # Convert HER transitions to ReplayBufferSamples format
            her_batch = self._convert_her_to_replay_buffer_samples(her_transitions, n_her_samples)
            
            # Combine batches
            return self._combine_replay_buffer_samples(regular_batch, her_batch)
                
        return regular_batch
        
    def _convert_her_to_batch(self, her_transitions, batch_size):
        """Convert HER transitions to batch format."""
        import torch
        
        def to_numpy(array):
            """Convert array to numpy, handling torch tensors."""
            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            return array
        
        # This is a simplified version - actual implementation would need
        # to match the exact format of your replay buffer
        batch = {}
        batch['observations'] = np.stack([to_numpy(t['obs']) for t in her_transitions[:batch_size]])
        batch['actions'] = np.stack([to_numpy(t['action']) for t in her_transitions[:batch_size]])
        batch['rewards'] = np.stack([to_numpy(t['reward']) for t in her_transitions[:batch_size]])
        batch['done'] = np.stack([to_numpy(t["done"]) for t in her_transitions[:batch_size]])
        
        return batch
        
    def _convert_her_to_replay_buffer_samples(self, her_transitions, batch_size):
        """Convert HER transitions to ReplayBufferSamples format."""
        import torch
        
        def to_numpy(array):
            """Convert array to numpy, handling torch tensors."""
            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            return array
        
        # Extract observations, actions, rewards, etc. from HER transitions
        batch_size = min(batch_size, len(her_transitions))
        
        # Build observations by stacking all observation components
        observations_list = []
        next_observations_list = []
        
        for transition in her_transitions[:batch_size]:
            # Convert observation dict to flat array or use as-is
            obs = transition['obs']
            next_obs = transition.get('next_obs', obs)  # Use same obs if next_obs not available
            
            # Debug: Print HER transition information
            print(f"DEBUG: HER transition obs type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"DEBUG: HER transition obs keys: {list(obs.keys())}")
            
            # # The issue is that HER transitions store raw observations, but regular batch uses processed observations
            # # We need to add goal information to match the regular batch format
            # if isinstance(obs, dict):
            #     # Add goal information to match the format used by regular batch
            #     # The regular batch uses observations with goals added via _update_goal_in_observation
            #     goal_to_use = transition.get('robot_internal_state', transition['original_goal'])
            #     obs_with_goal = self.her_wrapper._update_goal_in_observation(obs, goal_to_use)
                
            #     # Debug: Print observation processing information
            #     print(f"DEBUG: HER obs_with_goal type: {type(obs_with_goal)}")
            #     if hasattr(obs_with_goal, 'shape'):
            #         print(f"DEBUG: HER obs_with_goal shape: {obs_with_goal.shape}")
                
            #     observations_list.append(to_numpy(obs_with_goal))
                
            #     # Do the same for next_obs
            #     if isinstance(next_obs, dict):
            #         next_obs_with_goal = self.her_wrapper._update_goal_in_observation(next_obs, goal_to_use)
            #         next_observations_list.append(to_numpy(next_obs_with_goal))
            #     else:
            #         next_observations_list.append(to_numpy(next_obs))
            # else:
            #     # For now, we'll assume observations are already in the right format
            observations_list.append(to_numpy(obs))
            next_observations_list.append(to_numpy(next_obs))
        
        # Stack all arrays
        observations = np.stack(observations_list)
        next_observations = np.stack(next_observations_list)
        actions = np.stack([to_numpy(t['action']) for t in her_transitions[:batch_size]])
        rewards = np.stack([to_numpy(t['reward']) for t in her_transitions[:batch_size]])
        dones = np.stack([to_numpy(t['done']) for t in her_transitions[:batch_size]])
        
        # Debug: Print HER batch information
        print(f"DEBUG: HER batch observations shape: {observations.shape}")
        print(f"DEBUG: HER batch observations dtype: {observations.dtype}")
        if observations.size > 0:
            print(f"DEBUG: HER batch first observation sample: {observations[0][:5]}..." if observations.ndim > 1 else f"DEBUG: HER batch first observation: {observations[0]}")
        print(f"DEBUG: HER batch actions shape: {actions.shape}")
        print(f"DEBUG: HER batch rewards shape: {rewards.shape}")
        print(f"DEBUG: HER batch dones shape: {dones.shape}")
        
        # Ensure observations have the same shape as expected by the regular batch
        # If observations are 1D but should be 2D, reshape them
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        if next_observations.ndim == 1:
            next_observations = next_observations.reshape(-1, 1)
        
        # Create ReplayBufferSamples named tuple
        return ReplayBufferSamples(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            discounts=None
        )
    
    def _combine_replay_buffer_samples(self, regular_batch, her_batch):
        """Combine two ReplayBufferSamples objects."""
        import torch


        def to_numpy(array):
            """Convert array to numpy, handling torch tensors."""
            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            return array
        
        # Convert all arrays to numpy (handling potential CUDA tensors)
        regular_obs = to_numpy(regular_batch.observations)
        her_obs = to_numpy(her_batch.observations)
        regular_next_obs = to_numpy(regular_batch.next_observations)
        her_next_obs = to_numpy(her_batch.next_observations)
        regular_actions = to_numpy(regular_batch.actions)
        her_actions = to_numpy(her_batch.actions)
        regular_rewards = to_numpy(regular_batch.rewards)
        her_rewards = to_numpy(her_batch.rewards)
        regular_dones = to_numpy(regular_batch.dones)
        her_dones = to_numpy(her_batch.dones)
        
        # Debug: Print shapes before concatenation
        print(f"DEBUG: Before concatenation - regular_obs shape: {regular_obs.shape} {type(regular_obs)}")
        print(f"DEBUG: Before concatenation - her_obs shape: {her_obs.shape} {type(her_obs)}")
        print(f"DEBUG: Before concatenation - regular_actions shape: {regular_actions.shape}")
        print(f"DEBUG: Before concatenation - her_actions shape: {her_actions.shape}")
        print(f"DEBUG: Before concatenation - regular_dones shape: {regular_dones.shape}")
        print(f"DEBUG: Before concatenation - her_dones shape: {her_dones.shape}")
        
        # Ensure both arrays have the same number of dimensions
        if regular_obs.ndim != her_obs.ndim:
            # Try to reshape the arrays to match dimensions
            if regular_obs.ndim == 2 and her_obs.ndim == 1:
                her_obs = her_obs.reshape(1, -1) if her_obs.size > 0 else her_obs.reshape(0, 0)
            elif regular_obs.ndim == 1 and her_obs.ndim == 2:
                regular_obs = regular_obs.reshape(1, -1) if regular_obs.size > 0 else regular_obs.reshape(0, 0)
            else:
                # More complex case - try to expand dimensions
                if her_obs.ndim < regular_obs.ndim:
                    her_obs = np.expand_dims(her_obs, axis=0)
                else:
                    regular_obs = np.expand_dims(regular_obs, axis=0)
        


        her_rewards = np.expand_dims(her_rewards, axis=0)
        her_dones = np.expand_dims(her_dones, axis=0)

        print(f"{regular_obs=}")
        print(f"{her_obs=}")
        print(f"{regular_obs.shape=}")
        print(f"{her_obs.shape=}")

        print(f"{regular_rewards.shape=}")
        print(f"{her_rewards.shape=}")

        # Concatenate all arrays
        combined = ReplayBufferSamples(
            observations=np.concatenate([regular_obs, her_obs], axis=0),
            next_observations=np.concatenate([regular_next_obs, her_next_obs], axis=0),
            actions=np.concatenate([regular_actions, her_actions], axis=0),
            rewards=np.concatenate([regular_rewards, her_rewards], axis=0),
            dones=np.concatenate([regular_dones, her_dones], axis=0),
            discounts=to_numpy(regular_batch.discounts) if regular_batch.discounts is not None else None  # Keep discounts from regular batch
        )

        # Debug: Print final combined batch information
        print(f"DEBUG: Combined batch observations shape: {combined.observations.shape}")
        print(f"DEBUG: Combined batch actions shape: {combined.actions.shape}")
        print(f"DEBUG: Combined batch rewards shape: {combined.rewards.shape}")
        print(f"DEBUG: Combined batch dones shape: {combined.dones.shape}")

        return combined
    
    def _convert_batch_to_numpy(self, batch):
        """Convert a ReplayBufferSamples batch to numpy arrays."""
        import torch
        
        def to_numpy(array):
            """Convert array to numpy, handling torch tensors."""
            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            return array
        
        return ReplayBufferSamples(
            observations=to_numpy(batch.observations),
            next_observations=to_numpy(batch.next_observations),
            actions=to_numpy(batch.actions),
            rewards=to_numpy(batch.rewards),
            dones=to_numpy(batch.dones),
            discounts=to_numpy(batch.discounts) if batch.discounts is not None else None
        )
    
    def __getattr__(self, name: str):
        """Delegate other attribute accesses to the base buffer."""
        return getattr(self.base_buffer, name)