"""
Hindsight Experience Replay (HER) wrapper for SocNavGym.

This wrapper modifies the environment to support goal-relabeling for HER.
It works by storing the original goals and allowing the replay buffer
to sample alternative goals for past experiences.
"""

import numpy as np
from typing import Dict, Any, Tuple
import copy


class HERGoalEnvWrapper:
    """
    Wrapper that adds HER support to SocNavGym environment.
    
    This wrapper:
    1. Stores original goals for each episode
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
        self.env = env
        self.her_config = her_config
        
        # Store original observation space
        self.original_obs_space = self.env.observation_space
        
        # Add goal information to observations
        self._extend_observation_space()
        
        # Episode tracking
        self.current_episode_goals = None
        self.episode_transitions = []
        
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
        
        # Store the original goal for this episode
        # Access robot through the unwraped environment
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        self.current_episode_goals = {
            'robot_goal': np.array([base_env.robot.goal_x, base_env.robot.goal_y]),
            'original_obs': copy.deepcopy(obs)
        }
        
        # Clear episode transitions
        self.episode_transitions = []
        
        # Add goal info to observation
        obs_with_goal = self._add_goal_to_obs(obs, self.current_episode_goals['robot_goal'])
        
        return obs_with_goal, info
    
    def _add_goal_to_obs(self, obs: Dict[str, Any], goal: np.ndarray) -> Dict[str, Any]:
        """
        Add goal information to observation.
        
        Args:
            obs: Original observation
            goal: Goal position [x, y]
            
        Returns:
            Observation with added goal information
        """
        # For SocNavGym v2, we can add goal info to the robot observation
        # The robot observation already contains goal information in the first 2 elements
        # So we don't need to modify it, but we'll track it separately
        obs_with_goal = copy.deepcopy(obs)
        
        # Store goal in info if needed
        if 'goal' not in obs_with_goal:
            obs_with_goal['goal'] = goal
            
        return obs_with_goal
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Step the environment and store transition for potential HER relabeling.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Step the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store transition for potential HER relabeling
        if self.current_episode_goals is not None:
            transition = {
                'obs': copy.deepcopy(obs),
                'action': copy.deepcopy(action),
                'reward': reward,
                'next_obs': None,  # Will be filled later
                'done': terminated or truncated,
                'original_goal': copy.deepcopy(self.current_episode_goals['robot_goal'])
            }
            self.episode_transitions.append(transition)
        
        # Add goal info to observation
        obs_with_goal = self._add_goal_to_obs(obs, self.current_episode_goals['robot_goal'])
        
        return obs_with_goal, reward, terminated, truncated, info
    
    def relabel_goal(self, transition: Dict[str, Any], new_goal: np.ndarray) -> Dict[str, Any]:
        """
        Relabel a transition with a new goal.
        
        Args:
            transition: Original transition
            new_goal: New goal to use for relabeling
            
        Returns:
            Relabeled transition with new goal and recalculated reward
        """
        # Create a copy of the transition
        relabeled = copy.deepcopy(transition)
        
        # Update the goal in the observation
        if 'goal' in relabeled['obs']:
            relabeled['obs']['goal'] = new_goal
            
        # Recalculate reward based on new goal
        # For navigation tasks, reward is typically distance-based
        relabeled['reward'] = self._calculate_her_reward(relabeled['obs'], new_goal)
        
        # Mark as relabeled
        relabeled['relabeled'] = True
        relabeled['new_goal'] = new_goal
        
        return relabeled
    
    def _calculate_her_reward(self, obs: Dict[str, Any], goal: np.ndarray) -> float:
        """
        Calculate reward for HER relabeled goal.
        
        Args:
            obs: Observation containing robot position
            goal: New goal position
            
        Returns:
            Recalculated reward
        """
        # Extract robot position from observation
        # In SocNavGym v2, robot observation contains goal coordinates in robot frame
        robot_obs = obs['robot']
        
        # The first 2 elements after one-hot encoding are goal coordinates in robot frame
        # For simple distance-based reward:
        distance_to_goal = np.linalg.norm(robot_obs[6:8])  # x, y coordinates in robot frame
        
        # Simple reward: negative distance (encourages getting closer to goal)
        reward = -distance_to_goal
        
        # Add bonus for reaching goal
        # Access GOAL_THRESHOLD through the base environment
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        if distance_to_goal < base_env.GOAL_THRESHOLD:
            reward += 10.0  # Bonus for reaching goal
            
        return reward
    
    def sample_her_transitions(self, strategy: str = "future", n_samples: int = 4) -> list:
        """
        Sample relabeled transitions using HER strategy.
        
        Args:
            strategy: HER strategy ('future', 'final', 'episode', 'random')
            n_samples: Number of samples to generate per transition
            
        Returns:
            List of relabeled transitions
        """
        her_transitions = []
        
        if not self.episode_transitions or len(self.episode_transitions) < 2:
            return her_transitions
            
        # Get the final state of the episode
        final_state = self.episode_transitions[-1]['next_obs']
        
        for i, transition in enumerate(self.episode_transitions):
            # Skip the last transition (no next state)
            if i == len(self.episode_transitions) - 1:
                continue
                
            # Get the next observation
            transition['next_obs'] = self.episode_transitions[i + 1]['obs']
            
            # Sample new goals based on strategy
            if strategy == "future":
                # Sample from future states in the same episode
                future_indices = list(range(i + 1, len(self.episode_transitions)))
                if future_indices:
                    for _ in range(min(n_samples, len(future_indices))):
                        future_idx = np.random.choice(future_indices)
                        future_goal = self.episode_transitions[future_idx]['obs']['robot'][6:8]  # Goal in robot frame
                        relabeled = self.relabel_goal(transition, future_goal)
                        her_transitions.append(relabeled)
                        
            elif strategy == "final":
                # Use the final state goal
                if final_state is not None:
                    final_goal = final_state['robot'][6:8]  # Goal in robot frame
                    relabeled = self.relabel_goal(transition, final_goal)
                    her_transitions.append(relabeled)
                    
            elif strategy == "episode":
                # Sample from any state in the episode
                episode_indices = list(range(len(self.episode_transitions)))
                for _ in range(n_samples):
                    sample_idx = np.random.choice(episode_indices)
                    sample_goal = self.episode_transitions[sample_idx]['obs']['robot'][6:8]  # Goal in robot frame
                    relabeled = self.relabel_goal(transition, sample_goal)
                    her_transitions.append(relabeled)
                    
            elif strategy == "random":
                # Sample random goals
                for _ in range(n_samples):
                    # Sample random goal within environment bounds
                    random_goal = np.array([
                        np.random.uniform(-self.env.MAP_X/2, self.env.MAP_X/2),
                        np.random.uniform(-self.env.MAP_Y/2, self.env.MAP_Y/2)
                    ])
                    relabeled = self.relabel_goal(transition, random_goal)
                    her_transitions.append(relabeled)
                    
        return her_transitions
    
    def __getattr__(self, name: str):
        """Delegate other attribute accesses to the base environment."""
        return getattr(self.env, name)


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
        
        # Sample HER transitions if enabled
        if self.her_config.get('enabled', False):
            her_ratio = self.her_config.get('ratio', 0.8)
            n_her_samples = int(batch_size * her_ratio)
            n_regular_samples = batch_size - n_her_samples
            
            # Sample HER transitions
            her_transitions = self.her_wrapper.sample_her_transitions(
                strategy=self.her_config.get('strategy', 'future'),
                n_samples=self.her_config.get('n_sampled_goal', 4)
            )
            
            # If we have HER transitions, mix them with regular samples
            if her_transitions and n_her_samples > 0:
                # Sample fewer regular transitions
                regular_batch = self.base_buffer.sample(n_regular_samples, env=env)
                
                # Convert HER transitions to the same format as regular samples
                her_batch = self._convert_her_to_batch(her_transitions, n_her_samples)
                
                # Combine batches
                batch = self._combine_batches(regular_batch, her_batch)
                return batch
                
        return regular_batch
        
    def _convert_her_to_batch(self, her_transitions, batch_size):
        """Convert HER transitions to batch format."""
        # This is a simplified version - actual implementation would need
        # to match the exact format of your replay buffer
        batch = {}
        for key in her_transitions[0]['obs'].keys():
            if key == 'robot':
                # Robot observation includes goal coordinates
                batch[f'robot_{key}'] = np.stack([t['obs']['robot'] for t in her_transitions[:batch_size]])
            else:
                batch[key] = np.stack([t['obs'][key] for t in her_transitions[:batch_size]])
        
        # Add actions, rewards, etc.
        batch['actions'] = np.stack([t['action'] for t in her_transitions[:batch_size]])
        batch['rewards'] = np.stack([t['reward'] for t in her_transitions[:batch_size]])
        batch['dones'] = np.stack([t['done'] for t in her_transitions[:batch_size]])
        
        return batch
        
    def _combine_batches(self, regular_batch, her_batch):
        """Combine regular and HER batches."""
        combined = {}
        
        # Combine observations
        for key in regular_batch.keys():
            if key.startswith('robot_'):
                # Handle robot observations specially
                combined_key = key.replace('robot_', '')
                if combined_key in her_batch:
                    combined[key] = np.concatenate([regular_batch[key], her_batch[key]], axis=0)
                else:
                    combined[key] = regular_batch[key]
            elif key in her_batch:
                combined[key] = np.concatenate([regular_batch[key], her_batch[key]], axis=0)
            else:
                combined[key] = regular_batch[key]
                
        # Combine other data
        for suffix in ['actions', 'rewards', 'dones']:
            if suffix in her_batch:
                combined[suffix] = np.concatenate([regular_batch[suffix], her_batch[suffix]], axis=0)
                
        return combined
    
    def __getattr__(self, name: str):
        """Delegate other attribute accesses to the base buffer."""
        return getattr(self.base_buffer, name)