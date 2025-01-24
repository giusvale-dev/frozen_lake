import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from q_learning import FrozenLakeDeterministic
from q_learning import FrozenLakeNonDeterministic

  
alpha_label = '\u03B1'
gamma_label = '\u03B3'
epsilon_label = '\u03B5'      

def log_gamma_analysis(deterministic: bool, iterations: int):
    
    gamma_values = [0.1, 0.5, 0.98]
    epsilon = 0.98 # Intially the agent prefers explotations
    
    plt.figure(figsize=(12, 8))
     
    if deterministic:
        plt.title(f'Logarithmic Win/Loss Ratio During Training ({epsilon_label}={epsilon})')
        for gamma in gamma_values:
            fl = FrozenLakeDeterministic(epsilon=epsilon, discount_factor=gamma, render=False, episodes_number=iterations)
            Q, rewards_per_episode, policy = fl.training()
            cumulative_episodes = np.arange(1, len(rewards_per_episode) + 1)
            cumulative_wins = np.cumsum(rewards_per_episode)
            cumulative_losses = cumulative_episodes - cumulative_wins
            
            # Avoid division by zero
            cumulative_losses = np.maximum(cumulative_losses, 1)

            # Compute log(win/loss ratio)
            log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)
            log_win_loss_ratio = np.nan_to_num(log_win_loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)

            # Plot            
            plt.plot(log_win_loss_ratio, label=f'"{gamma_label}={gamma}"')
          
    else:
        alpha = 0.1  # The agent learn in conservative way 
        plt.title(f'Logarithmic Win/Loss Ratio During Training ({epsilon_label}={epsilon}, {alpha_label}={alpha})')
        for gamma in gamma_values:
            fl = FrozenLakeNonDeterministic(epsilon=epsilon, learning_rate=alpha, discount_factor=gamma, render=False, episodes_number=iterations)
            Q, rewards_per_episode, policy = fl.training()
            cumulative_episodes = np.arange(1, len(rewards_per_episode) + 1)
            cumulative_wins = np.cumsum(rewards_per_episode)
            cumulative_losses = cumulative_episodes - cumulative_wins
            
            # Avoid division by zero
            cumulative_losses = np.maximum(cumulative_losses, 1)

            # Compute log(win/loss ratio)
            log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)
            log_win_loss_ratio = np.nan_to_num(log_win_loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)

            # Plot            
            plt.plot(log_win_loss_ratio, label=f'"{gamma_label}={gamma}"')
        
    plt.xlabel('Episodes')
    plt.ylabel('Log(Win/Loss Ratio)')
    plt.legend(loc='best')
    plt.grid(True)
    max_abs_log_ratio = max(abs(log_win_loss_ratio))
    plt.ylim(-max_abs_log_ratio, max_abs_log_ratio)
    
    if deterministic:
        plt.savefig("log_gamma_analysis.png")
    else:
        plt.savefig("log_gamma_analysis_nd.png")

def log_epsilon_analysis(deterministic: bool, iterations: int):

    epsilon_values = [0.1, 0.5, 0.98]
    gamma = 0.98 # The agent prefers the future rewards
    
    plt.figure(figsize=(12, 8))
     
    if deterministic:
        plt.title(f'Logarithmic Win/Loss Ratio During Training ({gamma_label}={gamma})')
        for epsilon in epsilon_values:
            fl = FrozenLakeDeterministic(epsilon=epsilon, discount_factor=gamma, render=False, episodes_number=iterations)
            Q, rewards_per_episode, policy = fl.training()
            cumulative_episodes = np.arange(1, len(rewards_per_episode) + 1)
            cumulative_wins = np.cumsum(rewards_per_episode)
            cumulative_losses = cumulative_episodes - cumulative_wins
            
            # Avoid division by zero
            cumulative_losses = np.maximum(cumulative_losses, 1)

            # Compute log(win/loss ratio)
            log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)
            log_win_loss_ratio = np.nan_to_num(log_win_loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)

            # Plot            
            plt.plot(log_win_loss_ratio, label=f'"{epsilon_label}={epsilon}"')

          
    else:
        alpha = 0.1  # The agent learn in conservative way 
        plt.title(f'Logarithmic Win/Loss Ratio During Training ({gamma_label}={gamma}, {alpha_label}={alpha})')
        for epsilon in epsilon_values:
            fl = FrozenLakeNonDeterministic(epsilon=epsilon, learning_rate=alpha, discount_factor=gamma, render=False, episodes_number=iterations)
            Q, rewards_per_episode, policy = fl.training()
            cumulative_episodes = np.arange(1, len(rewards_per_episode) + 1)
            cumulative_wins = np.cumsum(rewards_per_episode)
            cumulative_losses = cumulative_episodes - cumulative_wins
            
            # Avoid division by zero
            cumulative_losses = np.maximum(cumulative_losses, 1)

            # Compute log(win/loss ratio)
            log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)
            log_win_loss_ratio = np.nan_to_num(log_win_loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)

            # Plot            
            plt.plot(log_win_loss_ratio, label=f'"{epsilon_label}={epsilon}"')
        
    plt.xlabel('Episodes')
    plt.ylabel('Log(Win/Loss Ratio)')
    plt.legend(loc='best')
    plt.grid(True)
    max_abs_log_ratio = max(abs(log_win_loss_ratio))
    plt.ylim(-max_abs_log_ratio, max_abs_log_ratio)
    
    if deterministic:
        plt.savefig("log_epsilon_analysis.png")
    else:
        plt.savefig("log_epsilon_analysis_nd.png")

def log_alpha_analysis(iterations: int):

    alpha_values = [0.1, 0.5, 0.98]
    gamma = 0.98 # The agent prefers the future rewards
    epsilon = 0.98
    
    plt.figure(figsize=(12, 8))
    
    plt.title(f'Logarithmic Win/Loss Ratio During Training ({gamma_label}={gamma}, {epsilon_label}={epsilon})')
    for alpha in alpha_values:
        fl = FrozenLakeNonDeterministic(epsilon=epsilon, learning_rate=alpha, discount_factor=gamma, render=False, episodes_number=iterations)
        Q, rewards_per_episode, policy = fl.training()
        cumulative_episodes = np.arange(1, len(rewards_per_episode) + 1)
        cumulative_wins = np.cumsum(rewards_per_episode)
        cumulative_losses = cumulative_episodes - cumulative_wins
        
        # Avoid division by zero
        cumulative_losses = np.maximum(cumulative_losses, 1)
        # Compute log(win/loss ratio)
        log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)
        log_win_loss_ratio = np.nan_to_num(log_win_loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        # Plot            
        plt.plot(log_win_loss_ratio, label=f'"{alpha_label}={alpha}"')
    
    plt.xlabel('Episodes')
    plt.ylabel('Log(Win/Loss Ratio)')
    plt.legend(loc='best')
    plt.grid(True)
    max_abs_log_ratio = max(abs(log_win_loss_ratio))
    plt.ylim(-max_abs_log_ratio, max_abs_log_ratio)
    
    plt.savefig("log_alpha_analysis_nd.png")


def learned_policy_deterministic_comparison(agents: List[FrozenLakeDeterministic]):
 
    plt.figure(figsize=(12, 8))
    plt.title(f'Logarithmic Win/Loss Ratio During Training)')
    for agent in agents:
        Q, rewards_per_episode, policy = agent.training()
        cumulative_episodes = np.arange(1, len(rewards_per_episode) + 1)
        cumulative_wins = np.cumsum(rewards_per_episode)
        cumulative_losses = cumulative_episodes - cumulative_wins

        # Avoid division by zero
        cumulative_losses = np.maximum(cumulative_losses, 1)
        # Compute log(win/loss ratio)
        log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)
        #log_win_loss_ratio = np.nan_to_num(log_win_loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        # Plot            
        plt.plot(log_win_loss_ratio, label=f'"{epsilon_label}={agent.epsilon}, {gamma_label}={agent.discount_factor}"')
    
    plt.xlabel('Episodes')
    plt.ylabel('Log(Win/Loss Ratio)')
    plt.legend(loc='best')
    plt.grid(True)
    max_abs_log_ratio = max(abs(log_win_loss_ratio))
    plt.ylim(-max_abs_log_ratio, max_abs_log_ratio)
    
    plt.savefig("learned_policy_deterministic_comparison.png")


def plot_cumulative_rewards(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name="cumulative_rewards.png"):
    
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

def cumulative_sum(data):
    return np.cumsum(data)

def plot_learning_curve_with_fixed_gamma_and_alpha(gamma: float, alpha: float, epsilon_values: List, episodes_number: int):
    
    alpha_label = '\u03B1'
    gamma_label = '\u03B3'

    plt.figure(figsize=(12, 8))

    for epsilon in epsilon_values:
        # Create the environment and agent with the specific epsilon and fixed gamma
        agent = FrozenLakeNonDeterministic(
            epsilon=epsilon,
            discount_factor=gamma,
            render=False,
            learning_rate=alpha,
            episodes_number=episodes_number
        )

        # Train the agent
        Q, rewards_per_episodes, policy = agent.training()

        # Plot the learning curve (smoothed rewards)
        plt.plot(np.cumsum(rewards_per_episodes), label=f"epsilon={epsilon}")
       
 
    plt.title(f'Learning Curves for Different Epsilon Values ({gamma_label}={gamma}, {alpha_label}={alpha})')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig("plot_learning_curve_with_fixed_gamma_and_alpha.png")

def heatmap(qtable, file_name:str, title:str, xlabel:str, ylabel:str):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(qtable, annot=True, cmap="YlGnBu", fmt=".4f", cbar=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_name + ".png")

def success_rate_log(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name="success_rate_log.png"):
    alpha_label = '\u03B1'
    gamma_label = '\u03B3'

    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    for i in range(len(values)):
        sum = np.cumsum(values[i])
        if sum[len(sum) - 1] > 0:
            plt.plot(np.log10(sum/len(values[i])), label=categories[i], alpha = 0.3)
        else:
            sum = [-10] * len(values[i])       
            plt.plot(sum, label=categories[i], alpha = 0.3)
    
    plt.ylim(-20, 5)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)