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

    def training(self, map_name="4x4"):
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=True, render_mode="human" if self.render else None)
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Zeroize the Q-Table
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
    
    def run_agent(self, policy, map_name="4x4"):
        
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=True, render_mode="human" if self.render else None)

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
    
    def __init__(self, epsilon: float, discount_factor: float, render: bool, episodes_number: int):
        
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
        
    def training(self, map_name="4x4"):
        
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False, render_mode=True if self.render else None)
            
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
    
    def run_agent(self, policy, map_name="4x4"):
        
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False, render_mode="human" if self.render else None)

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
    
    random_number = np.random.default_rng().random()
    if random_number < epsilon:
        action = env.action_space.sample() # Explore: random action
    else:
        action = np.argmax(q_table[state,:]) # Exploit: best action
    return action