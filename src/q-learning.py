import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import random
from typing import List
import math


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

def decay_rate(init_epsilon: float, epsilon_min: float, total_episodes: int) -> float:
    if epsilon_min >= init_epsilon:
        raise Exception("epsilon_min must be < init_epsilon")
    if total_episodes <= 0 or epsilon_min <= 0 or init_epsilon <=0:
        raise Exception("The input parameters must be >= 0")
    # Compute the decay rate
    k = (init_epsilon - epsilon_min) / total_episodes
    return k

def decrement_epsilon(epsilon: float, epsilon_min: float, decay_rate:float, episode_step: int) -> float:
    return max(epsilon_min, epsilon - (decay_rate * episode_step))


def deterministic_training(discount_factor, epsilon, episodes):
    
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=None)
    q = np.zeros((env.observation_space.n, env.action_space.n)) # Zeroize the Q-Table (8x8x4)
    rewards_per_episodes = np.zeros(episodes)

   # k = decay_rate(epsilon, 0.01, episodes)
    learning_rate = 0.1

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63
        terminated = False      # True when agent goes in hole or reached goal
        truncated = False       # True when actions > 200 (value given by documentation)
        while(not terminated and not truncated):

            action = epsilon_greedy(env, epsilon, q, state)
            
            new_state,reward,terminated,truncated,_ = env.step(action) # execute the action
           
            #q[state, action] = reward + discount_factor * np.max(q[new_state,:])
            # Update the Q-value using the Q-learning rule
            q[state, action] = q[state, action] + learning_rate * (
                reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
            )
            # Set the new state
            state = new_state

        # Keep trace of rewards for episode
        if reward == 1:
            print("YESSSSSSSSS")
            rewards_per_episodes[i] = 1
        
        # Linear decay
    #    epsilon = decrement_epsilon(epsilon, 0.01, k, i)
    
    # Episode ends, close the environment
    env.close()
    return q, rewards_per_episodes

def q_learning_deterministic(number_episodes, epsilon=0.95, discount_factor=1, learning_rate = 0.1):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=None)
    q = np.zeros((env.observation_space.n, env.action_space.n)) # Zeroize the Q-Table (8x8x4)
    rewards_per_episodes = np.zeros(number_episodes)
    for i in range(number_episodes):
        state = env.reset()[0]
        terminated = False      # True when agent goes in hole or reached goal
        truncated = False       # True when actions > 200 (value given by documentation)
        while(not terminated and not truncated):

            action = epsilon_greedy(env, epsilon, q, state) # Choose an action with epsilon-greedy strategy
        
            new_state,reward,terminated,truncated,_ = env.step(action) # Execute and observe the action
           
           # Update the Q-value using the Q-learning rule
           # q[state, action] = reward + discount_factor * np.max(q[new_state,:])
            q[state, action] = q[state, action] + learning_rate * (
                reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
            )
         
            # Set the new state
            state = new_state

        # Keep trace of rewards for episode
        if reward == 1:
            rewards_per_episodes[i] = 1
        
    # Derive policy from Q-Values
    policy = {s: np.argmax(q[s]) for s in range (env.observation_space.n)}

    # Episode ends, close the environment
    env.close()
    
    return q, rewards_per_episodes, policy

def use_deterministic_learning(policy, number_episodes):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="human")
    total_reward = 0
    for i in range(number_episodes):
        
        state = env.reset()[0]  # states: 0 to 63
        terminated = False      # True when agent goes in hole or reached goal
        truncated = False       # True when actions > 200 (value given by documentation)
        
        while(not terminated and not truncated):
            action = policy[state]
            new_state, reward, terminated, truncated,_ = env.step(action) # execute the action
            state = new_state
        
        total_reward+=reward
            
    # Episode ends, close the environment
    env.close()
    return total_reward


def cumulative_rewards_chart(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name="cumulative_rewards.png"):
    
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    for i in range(len(values)):
        cumulative_sum = np.cumsum(values[i])
        plt.plot(cumulative_sum, label=categories[i], alpha = 0.6)
    
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)


def print_qtable(qtable):
    for row in qtable:
        print(" ".join(map(str, row)))

def bar_chart(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name):
    
    plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_name + ".png")

def epsilon_analysis():

    categories = ["\u03B5=1", "\u03B5=0.99", "\u03B5=0.95", "\u03B5=0.5"]
    values = []
    reward_list = []
    
    q, rewards, policy = q_learning_deterministic(10000, 1, 1, 0.01)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.99, 1, 0.01)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.95, 1, 0.01)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.5, 1, 0.01)
    reward_list.append(rewards)

    cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) \u03B3=1 \u03B1=0.01", file_name="epsilon_deterministic_analysis")

def learning_rate_analysis():

    categories = ["\u03B1=0.5", "\u03B1=0.3", "\u03B1=0.1", "\u03B1=0.01"]
    reward_list = []
    
    q, rewards, policy = q_learning_deterministic(10000, 0.95, 1, 0.5)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.95, 1, 0.3)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.95, 1, 0.1)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.95, 1, 0.01)
    reward_list.append(rewards)

    cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) \u03B5=0.95 \u03B3=1", file_name="learning_rate_deterministic_analysis")

def discount_factor_analysis():
    categories = ["\u03B3=1", "\u03B3=0.99", "\u03B3=0.95", "\u03B3=0.7"]
    reward_list = []
    
    q, rewards, policy = q_learning_deterministic(10000, 0.95, 1, 0.01)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.95, 0.99, 0.01)
    reward_list.append(rewards)

    q, rewards, policy = q_learning_deterministic(10000, 0.95, 0.95, 0.01)
    reward_list.append(rewards)
    
    q, rewards, policy = q_learning_deterministic(10000, 0.95, 0.7, 0.01)
    reward_list.append(rewards)

    cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) \u03B5=0.95 \u03B1=0.01", file_name="discount_factor_deterministic_analysis")

def heatmap(qtable, file_name:str, title:str, xlabel:str, ylabel:str):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(qtable, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_name + ".png")

if __name__ == '__main__':
    #epsilon_analysis()
    #learning_rate_analysis()
    #discount_factor_analysis()
    q, rewards, policy = q_learning_deterministic(10000)
    print(use_deterministic_learning(policy, 10))
    #heatmap(q, "qtable", "Q-Table", "Actions", "States")
        
