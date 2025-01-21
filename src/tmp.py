
# def q_learning_deterministic(number_episodes, epsilon=0.95, discount_factor=0.99, learning_rate = 0.1):
#     env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=None)
#     q = np.zeros((env.observation_space.n, env.action_space.n)) # Zeroize the Q-Table (8x8x4)
#     rewards_per_episodes = np.zeros(number_episodes)
#     epsilon_min = 0.01
#     k = 0.00001
#     epsilon_start = epsilon
#     for i in range(number_episodes):
#         state = env.reset()[0]
#         terminated = False      # True when agent goes in hole or reached goal
#         truncated = False       # True when actions > 200 (value given by documentation)
#         while(not terminated and not truncated):

#             action = epsilon_greedy(env, epsilon, q, state) # Choose an action with epsilon-greedy strategy

#             new_state,reward,terminated,truncated,_ = env.step(action) # Execute and observe the action
           
#             # Update the Q-value using the Q-learning rule
#             q[state, action] = reward + discount_factor * np.max(q[new_state,:])
            
#             # Set the new state
#             state = new_state

#         # Keep trace of rewards for episode
#         if reward == 1:
#             rewards_per_episodes[i] = 1
        
#         # Update epsilon
#         epsilon = max(epsilon_min, epsilon_start * math.exp(-k * (i+1)))
                

#     # Derive policy from Q-Values
#     policy = {s: np.argmax(q[s]) for s in range (env.observation_space.n)}

#     # Episode ends, close the environment
#     env.close()
    
#     return q, rewards_per_episodes, policy



# def q_learning_non_deterministic(number_episodes, epsilon=0.95, discount_factor=1, learning_rate = 0.1):
#     env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode=None)
#     q = np.zeros((env.observation_space.n, env.action_space.n)) # Zeroize the Q-Table (8x8x4)
#     rewards_per_episodes = np.zeros(number_episodes)
#     for i in range(number_episodes):
#         state = env.reset()[0]
#         terminated = False      # True when agent goes in hole or reached goal
#         truncated = False       # True when actions > 200 (value given by documentation)
#         while(not terminated and not truncated):

#             action = epsilon_greedy(env, epsilon, q, state) # Choose an action with epsilon-greedy strategy
        
#             new_state,reward,terminated,truncated,_ = env.step(action) # Execute and observe the action
           
#            # Update the Q-value using the Q-learning rule
#             q[state, action] = q[state, action] + learning_rate * (
#                 reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
#             )
         
#             # Set the new state
#             state = new_state

#         # Keep trace of rewards for episode
#         if reward == 1:
#             rewards_per_episodes[i] = 1
        
#     # Derive policy from Q-Values
#     policy = {s: np.argmax(q[s]) for s in range (env.observation_space.n)}

#     # Episode ends, close the environment
#     env.close()
    
#     return q, rewards_per_episodes, policy

# def run_agent(policy, number_episodes: int, deterministic: bool, render: bool):
#     env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False if deterministic else True, render_mode="human" if render else None)
#     total_reward = 0
#     win_counts = []
#     loss_counts = []
#     for i in range(number_episodes):
        
#         state = env.reset()[0]  # states: 0 to 63
#         terminated = False      # True when agent goes in hole or reached goal
#         truncated = False
#         win = False

#         while(not terminated and not truncated):
#             action = policy[state]
#             new_state, reward, terminated, _, _ = env.step(action) # execute the action
#             state = new_state
        
#         total_reward+=reward

#         if reward > 0:
#             win_counts.append(1)
#             loss_counts.append(0)
#         else:
#             win_counts.append(0)
#             loss_counts.append(1)

            
#     # Episode ends, close the environment
#     env.close()
#     return total_reward, win_counts, loss_counts


# def cumulative_rewards_chart(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name="cumulative_rewards.png"):
    
#     plt.figure(figsize=(10, 6))
#     plt.title(title, fontsize=16)
#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel(ylabel, fontsize=14)

#     for i in range(len(values)):
#         cumulative_sum = np.cumsum(values[i])
#         plt.plot(cumulative_sum, label=categories[i], alpha = 0.6)
    
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(file_name)


# def print_qtable(qtable):
#     for row in qtable:
#         print(" ".join(map(str, row)))

# def bar_chart(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name):
    
#     plt.bar(categories, values, color='skyblue', edgecolor='black')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.savefig(file_name + ".png")

# def epsilon_analysis(deterministic: bool):

#     categories = ["\u03B5=1", "\u03B5=0.99", "\u03B5=0.95", "\u03B5=0.5"]
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
#     else:
#         q, rewards, policy = q_learning_non_deterministic(15000, 1, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_non_deterministic(15000, 0.99, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_non_deterministic(15000, 0.95, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_non_deterministic(15000, 0.5, 1, 0.1)
#         reward_list.append(rewards)

#         cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Non-Deterministic Environment \u03B3=1 \u03B1=0.1", file_name="epsilon_non_deterministic_analysis.png")


# def learning_rate_analysis(deterministic: bool):

#     categories = ["\u03B1=0.5", "\u03B1=0.3", "\u03B1=0.1", "\u03B1=0.01"]
#     reward_list = []
#     if deterministic:

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.5)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.3)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.01)
#         reward_list.append(rewards)

#         cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Deterministic Enironment \u03B5=0.95 \u03B3=1", file_name="learning_rate_deterministic_analysis")
#     else:
#         q, rewards, policy = q_learning_non_deterministic(15000, 0.95, 1, 0.5)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_non_deterministic(15000, 0.95, 1, 0.3)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_non_deterministic(15000, 0.95, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_non_deterministic(15000, 0.95, 1, 0.01)
#         reward_list.append(rewards)

#         cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Non-Deterministic Environment \u03B5=0.95 \u03B3=1", file_name="learning_rate_non_deterministic_analysis")


# def discount_factor_analysis(deterministic: bool):
#     categories = ["\u03B3=1", "\u03B3=0.99", "\u03B3=0.95", "\u03B3=0.7"]
#     reward_list = []
    
#     if deterministic:
    
#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 0.99, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 0.95, 0.1)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 0.7, 0.1)
#         reward_list.append(rewards)

#         cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Deterministic Environment \u03B5=0.95 \u03B1=0.1", file_name="discount_factor_deterministic_analysis")
    
#     else:
        
#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 1, 0.01)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 0.99, 0.01)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 0.95, 0.01)
#         reward_list.append(rewards)

#         q, rewards, policy = q_learning_deterministic(15000, 0.95, 0.7, 0.01)
#         reward_list.append(rewards)
#         cumulative_rewards_chart(values=reward_list, categories=categories, xlabel="Episodes", ylabel="Cumulative Reward ", title="Cumulative Reward Over Time (Episodes) Non-Deterministic Environment \u03B5=0.95 \u03B1=0.1", file_name="discount_factor_non_deterministic_analysis")

# def heatmap(qtable, file_name:str, title:str, xlabel:str, ylabel:str):
    
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(qtable, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.savefig(file_name + ".png")

# def plot_win_loss_log(wins, losses):
    
    
#     cumulative_wins = np.cumsum(wins)
#     cumulative_losses = np.cumsum(losses)
    
#     # Plot log of wins and losses (add 1 to avoid log(0) issue)
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.log(cumulative_wins + 1), label='Log(Wins)', color='green', linestyle='-', linewidth=2)
#     plt.plot(np.log(cumulative_losses + 1), label='Log(Losses)', color='red', linestyle='-', linewidth=2)
    
#     plt.xlabel('Episodes', fontsize=14)
#     plt.ylabel('Log(Count)', fontsize=14)
#     plt.title('Log of Wins and Losses Over Episodes', fontsize=16)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.savefig("win_loss_log.png")
