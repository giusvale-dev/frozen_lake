import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import math


class FrozenLakeNonDeterministic:
    
    def __init__(self, epsilon: float, discount_factor: float, learning_rate: float, render: bool, episodes_number: int):
        
        if epsilon < 0 or epsilon > 1:
            raise Exception("Epsilon must be between 0 and 1")
        
        if discount_factor < 0 or discount_factor > 1:
            raise Exception("Discount factor must be between 0 and 1")
        
        if learning_rate < 0 or learning_rate > 1:
            raise Exception("Learning rate must be between 0 and 1")
        
        if episodes_number <= 0:
            raise Exception("episodes_number must be > 0")
        
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.render = render
        self.episodes_number = episodes_number
        self.learning_rate = learning_rate

    def training(self):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human" if self.render else None)
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Zeroize the Q-Table (4x4x4)
        rewards_per_episodes = np.zeros(self.episodes_number)
        epsilon_min = 0.01
        k = 0.00005
        epsilon_start = self.epsilon
        for i in range(self.episodes_number):
            state = env.reset()[0]
            terminated = False      # True when agent goes in hole or reached goal
            truncated = False       # True when actions > 200 (value given by documentation)
            while(not terminated and not truncated):

                action = epsilon_greedy(env, self.epsilon, q, state) # Choose an action with epsilon-greedy strategy
            
                new_state,reward,terminated,truncated,_ = env.step(action) # Execute and observe the action
            
                # Update the Q-value using the Q-learning rule
                q[state, action] = q[state, action] + self.learning_rate * (
                    reward + self.discount_factor * np.max(q[new_state, :]) - q[state, action]
                )
            
                # Set the new state
                state = new_state

            # Keep trace of rewards for episode
            if reward == 1:
                rewards_per_episodes[i] = 1
            
            # Update epsilon
            self.epsilon = max(epsilon_min, epsilon_start * math.exp(-k * (i+1)))
            
        # Derive policy from Q-Values
        policy = {s: np.argmax(q[s]) for s in range (env.observation_space.n)}

        # Episode ends, close the environment
        env.close()
        
        #Reset epsilon to original value
        self.epsilon = epsilon_start

        return q, rewards_per_episodes, policy
    
    def run_agent(self, policy):
        
        """
        Runs the agent using the learned policy.

        Args:
            policy (dict): A dictionary mapping states to optimal actions.

        Returns:
            tuple:
                total_reward (int): Total rewards.
                win_counts: List of binary values indicating success in each episode.
                loss_counts: List of binary values indicating failure in each episode.
        """
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human" if self.render else None)

        total_reward = 0
        win_counts = []
        loss_counts = []
        for i in range(self.episodes_number):
            
            state = env.reset()[0]
            terminated = False
            truncated = False
            
            while(not terminated and not truncated):
                action = policy[state]
                new_state, reward, terminated, _, _ = env.step(action) # execute the action
                state = new_state
            
            total_reward+=reward

            if reward > 0:
                win_counts.append(1)
                loss_counts.append(0)
            else:
                win_counts.append(0)
                loss_counts.append(1)

        # Episode ends, close the environment
        env.close()
        return total_reward, win_counts, loss_counts

class FrozenLakeDeterministic:
    """
    Class to implement Q-learning for deterministic version of Frozen Lake
    """
    def __init__(self, epsilon: float, discount_factor: float, render: bool, episodes_number: int):
        
        """
        Initializes the FrozenLakeDeterministic object.

        Args:
            epsilon (float): Exploration probability, must be between 0 and 1.
            discount_factor (float): Discount factor for future rewards, must be between 0 and 1.
            render (bool): If True, show the environment UI.
            episodes_number (int): Number of episodes for training or evaluation.

        Raises:
            Exception: If epsilon is not between 0 and 1.
            Exception: If discount_factor is not between 0 and 1.
            Exception: If episodes_number is less than or equal to 0.
        """

        if epsilon < 0 or epsilon > 1:
            raise Exception("Epsilon must be between 0 and 1")
        
        if discount_factor < 0 or discount_factor > 1:
            raise Exception("Discount factor must be between 0 and 1")
        
        if episodes_number <= 0:
            raise Exception("episodes_number must be > 0")
        
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.render = render
        self.episodes_number = episodes_number
        
    def training(self):
        
        """
        Trains the agent.

        Returns:
            tuple:
                Q: The Q-table containing state-action values.
                rewards_per_episodes: Array of rewards.
                policy (dict): The optimal policy derived from the Q-values, mapping states to actions.
        """

        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=True if self.render else None)
            
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards_per_episodes = np.zeros(self.episodes_number)
        epsilon_min = 0.01
        k = 0.00001
        epsilon_start = self.epsilon
        for i in range(self.episodes_number):
            state = env.reset()[0]
            terminated = False      # True when agent goes in hole or reached goal
            truncated = False       # True when actions > 200 (value given by documentation)
            while(not terminated and not truncated):

                action = epsilon_greedy(env, self.epsilon, Q, state) # Choose an action with epsilon-greedy strategy

                new_state,reward,terminated,truncated,_ = env.step(action) # Execute and observe the action

                # Update the Q-value using the Q-learning rule
                Q[state, action] = reward + self.discount_factor * np.max(Q[new_state,:])

                # Set the new state
                state = new_state

            # Keep trace of rewards for episode
            if reward == 1:
                rewards_per_episodes[i] = 1

            # Update epsilon
            self.epsilon = max(epsilon_min, epsilon_start * math.exp(-k * (i+1)))

        # Derive policy from Q-Values
        policy = {s: np.argmax(Q[s]) for s in range (env.observation_space.n)}
        env.close()
        
        #Reset epsilon to original value
        self.epsilon = epsilon_start

        return Q, rewards_per_episodes, policy
    
    def run_agent(self, policy):
        """
        Runs the agent using the learned policy.

        Args:
            policy (dict): A dictionary mapping states to optimal actions.

        Returns:
            tuple:
                total_rewards: List of rewards obtained in each episode.
                win_counts: List of binary values indicating success in each episode.
                loss_counts: List of binary values indicating failure in each episode.
        """
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human" if self.render else None)

        total_rewards = []
        win_counts = []
        loss_counts = []
        for i in range(self.episodes_number):
            state = env.reset()[0]
            terminated = False
            truncated = False
            episode_reward = 0

            while(not terminated and not truncated):
                action = policy[state]
                new_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                state = new_state

            total_rewards.append(episode_reward)

            if episode_reward > 0:
                win_counts.append(1)
                loss_counts.append(0)
            else:
                win_counts.append(0)
                loss_counts.append(1)

        env.close()
        return total_rewards, win_counts, loss_counts
    
def epsilon_greedy(env, epsilon, q_table, state):
    """
    Perform epsilon-greedy action selection for a given state in a reinforcement learning environment.

    Parameters:
        env (gym.Env): The environment, which provides the action space.
        epsilon (float): The probability of choosing a random action (exploration). Should be in the range [0, 1].
        q_table (numpy.ndarray): A Q-table where q_table[state, action] gives the estimated value of taking the 
                                  action in the given state.
        state (int): The current state of the agent, represented as an index corresponding to rows in q_table.

    Returns:
        int: The selected action to execute in the environment.

    Description:
        The epsilon-greedy algorithm balances exploration and exploitation:
        - With probability `epsilon`, it selects a random action from the environment's action space.
        - With probability `1 - epsilon`, it selects the action with the highest Q-value for the current state.
    """
    random_number = np.random.default_rng().random()
    if random_number < epsilon:
        action = env.action_space.sample() # Explore: random action
    else:
        action = np.argmax(q_table[state,:]) # Exploit: best action
    return action