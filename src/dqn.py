import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Network Layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # First fully connected layer from input states and hidden layer
        self.out = nn.Linear(h1_nodes, out_actions) # Second fully connected layer from the hidden nodes and the output actions

    def forward(self, x):
        x = F.relu(self.fc1(x)) # ReLU activation
        x = self.out(x)         
        return x

# Memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQN():
    
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=1, network_sync_rate=10, replay_memory_size=1000, mini_batch_size=32):

        self.alpha = alpha        
        self.gamma = gamma         
        self.network_sync_rate = network_sync_rate
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.epsilon = epsilon

        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error
        self.optimizer = None                # NN Optimizer. Initialize later.

    # Train the FrozeLake environment
    def train(self, episodes, render=False, is_slippery=False, map_name="4x4"):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. (Adam)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        # list to plot the rewards per episode during the trainig
        rewards_per_episode = np.zeros(episodes)

        # list to plot the epsilon decay during the learning process
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy.
        step_count=0

        # Track the loss values
        losses_values = []
            
        for i in range(episodes):
            state = env.reset()[0]
            
            # End episode variables
            terminated = False    
            truncated = False

            while(not terminated and not truncated):

                # Choose action with epsilon-greedy strategy
                if random.random() < self.epsilon:
                    # select random action
                    action = env.action_space.sample()
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experiences has been collected and if at least 1 reward has been collected
            # If it is achieved optimize the policy
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                losses_values.append(self.optimize(mini_batch, policy_dqn, target_dqn))        

                # Decay epsilon
                self.epsilon = max(self.epsilon - 1/episodes, 0)
                epsilon_history.append(self.epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        return policy_dqn, rewards_per_episode, epsilon_history, losses_values
        
        
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                
                # When in a terminated state, target q value is set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.gamma * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    '''
    Converts a state (int) to a tensor (an array of number states) representation.
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def run_agent(self, episodes, policy, is_slippery=False, render=False, map_name="4x4"):
        
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode='human' if render else None)
        
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(policy.state_dict())
        policy_dqn.eval()

        total_rewards = []
        win_counts = []
        loss_counts = []

        for i in range(episodes):
            
            state = env.reset()[0]
            terminated = False
            truncated = False
            episode_reward = 0

            while(not terminated and not truncated):  
                
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                
                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)

            if episode_reward > 0:
                win_counts.append(1)
                loss_counts.append(0)
            else:
                win_counts.append(0)
                loss_counts.append(1)

        env.close()
        return total_rewards, win_counts, loss_counts