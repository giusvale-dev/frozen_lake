import gymnasium as gym
import numpy as np

from typing import List
import math

from plot import cumulative_rewards_chart


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

        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=True if self.render else None)
            
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
            epsilon = max(epsilon_min, epsilon_start * math.exp(-k * (i+1)))

        # Derive policy from Q-Values
        policy = {s: np.argmax(Q[s]) for s in range (env.observation_space.n)}
        env.close()

        return Q, rewards_per_episodes, policy
    
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
        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="human" if self.render else None)

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
    
    def discount_factor_analysis(self, q):
        pass
    def epsilon_analysis(self, q):
        categories = ["\u03B5=1", "\u03B5=0.99", "\u03B5=0.95", "\u03B5=0.5"]
#     reward_list = []
    
#     if deterministic:
#         q, rewards, policy = q_learning_deterministic(15000, 1, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.99, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.5, 1, 0.1)
#         reward_list.append(rewards)

#         cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Deterministic Environment \u03B3=1 \u03B1=0.1", file_name="epsilon_deterministic_analysis.png")


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

def epsilon_analysis():

    categories = ["\u03B5=1", "\u03B5=0.99", "\u03B5=0.95", "\u03B5=0.5"]
    reward_list = []
    
    fld1 = FrozenLakeDeterministic(epsilon=1, discount_factor=0.99, render=False, episodes_number=10000)
    q, rewards, policy = fld1.training()
    reward_list.append(rewards)

    q, rewards, policy = fld1.training()
    reward_list.append(rewards)

    fld1 = FrozenLakeDeterministic(epsilon=0.95, discount_factor=0.99, render=False, episodes_number=10000)
    q, rewards, policy = fld1.training()
    reward_list.append(rewards)

    fld1 = FrozenLakeDeterministic(epsilon=0.5, discount_factor=0.99, render=False, episodes_number=10000)
    q, rewards, policy = fld1.training()
    reward_list.append(rewards)

    cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Deterministic Environment \u03B3=1 \u03B1=0.1", file_name="epsilon_deterministic_analysis.png")

def discount_factor_analysis():
    categories = ["\u03B3=1", "\u03B3=0.99", "\u03B3=0.95", "\u03B3=0.7"]
    reward_list = []

    fld1 = FrozenLakeDeterministic(epsilon=0.95, discount_factor=1, render=False, episodes_number=10000)
    q1, rewards1, policy1 = fld1.training()
    reward_list.append(rewards1)

    fld2 = FrozenLakeDeterministic(epsilon=0.95, discount_factor=0.99, render=False, episodes_number=10000)
    q2, rewards2, policy2 = fld2.training()
    reward_list.append(rewards2)

    fld3 = FrozenLakeDeterministic(epsilon=0.95, discount_factor=0.95, render=False, episodes_number=10000)
    q3, rewards3, policy3 = fld3.training()
    reward_list.append(rewards3)

    fld4 = FrozenLakeDeterministic(epsilon=0.95, discount_factor=0.7, render=False, episodes_number=10000)
    q4, rewards4, policy4 = fld4.training()
    reward_list.append(rewards4)

    cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Deterministic Environment \u03B5=0.95", file_name="discount_factor_deterministic_analysis")

if __name__ == '__main__':
   
  #  fld = FrozenLakeDeterministic(epsilon=0.95, discount_factor=0.99, render=False, episodes_number=10000)
  #  q, rewards, policy = fld.training()
    discount_factor_analysis()
    epsilon_analysis()
    
 
 
    
  