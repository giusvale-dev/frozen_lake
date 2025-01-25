from q_learning import FrozenLakeDeterministic, FrozenLakeNonDeterministic
from plot import heatmap, learned_policy_deterministic_comparison, plot_cumulative_rewards, success_rate_log

from dqn import FrozenLakeDQN

import numpy as np
import matplotlib.pyplot as plt

gamma_label = '\u03B3'
epsilon_label = '\u03B5'
alpha_label = '\u03B1'

def frozen_lake_non_deterministic_comparison(iterations):
            
    categories = []
    values = []
    policy_rewards = []
    
    fld = FrozenLakeNonDeterministic(epsilon=0.98, discount_factor=0.98, learning_rate=0.1, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    categories.append(f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)

    heatmap(Q, "heatmap_nd1", f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}", "Actions", "States")

    fld = FrozenLakeNonDeterministic(epsilon=0.98, discount_factor=0.98, learning_rate=0.5, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    categories.append(f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)
    
    heatmap(Q, "heatmap_nd2", f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}", "Actions", "States")

    fld = FrozenLakeNonDeterministic(epsilon=0.98, discount_factor=0.98, learning_rate=0.9, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    categories.append(f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)
    
    heatmap(Q, "heatmap_nd3", f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}", "Actions", "States")

    plot_cumulative_rewards(categories=categories, values=values, title="Non Deterministic Frozen Lake (Cumulative Rewards Comparison)", xlabel="Episodes", ylabel="Cumulative Rewards", file_name="cumulative_rewards_non_deterministic_comparison")
    success_rate_log(categories=categories, values=policy_rewards, title = "Success rate log", xlabel="Episodes", ylabel="Log(wins/episodes)", file_name="success_rate_log_nd")


def frozen_lake_deterministic_comparison(iterations):
            
    categories = []
    values = []
    policy_rewards = []
    
    fld = FrozenLakeDeterministic(epsilon=0.98, discount_factor=0.98, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    categories.append(f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}")
    values.append(rewards)
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)

    heatmap(Q, "heatmap1", f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}", "Actions", "States")

    fld = FrozenLakeDeterministic(epsilon=0.98, discount_factor=0.1, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    categories.append(f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}")
    values.append(rewards)
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)

    heatmap(Q, "heatmap2", f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}", "Actions", "States")
    
    fld = FrozenLakeDeterministic(epsilon=0.1, discount_factor=0.98, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    categories.append(f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}")
    values.append(rewards)
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)

    heatmap(Q, "heatmap3", f"{epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}", "Actions", "States")
    
    plot_cumulative_rewards(categories=categories, values=values, title="Deterministic Frozen Lake (Cumulative Rewards Comparison)", xlabel="Episodes", ylabel="Cumulative Rewards", file_name="cumulative_rewards_deterministic_comparison")
    success_rate_log(categories=categories, values=policy_rewards, title = "Success rate log", xlabel="Episodes", ylabel="Log(wins/episodes)", file_name="success_rate_log")

def frozen_lake_dqn_vs_q_learning(iterations):
    
    categories = []
    values = []
    policy_rewards = []
    
    fld = FrozenLakeNonDeterministic(epsilon=1, discount_factor=0.98, learning_rate=0.001, render=False, episodes_number=iterations)
    Q, rewards, policy = fld.training()
    
    categories.append(f"Q-Learning Agent {epsilon_label}={fld.epsilon}, {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    
    _, win_counts, _ = fld.run_agent(policy)
    policy_rewards.append(win_counts)
    
    
    dqn = FrozenLakeDQN(alpha=0.001, gamma = fld.discount_factor, epsilon=fld.epsilon)
    policyDqn, rewards, _, _  = dqn.train(iterations, False, True, "4x4")
    categories.append(f"DQN Agent {gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    _, win_counts, _ = dqn.run_agent(iterations, policyDqn, True, False, "4x4")
    policy_rewards.append(win_counts)

    plot_cumulative_rewards(categories=categories, values=values, title="Cumulative curve DQN vs Q-Learning (Non-determinisitc Environment)", file_name="cumulative_rewards_dqn_vs_q_learning", xlabel="Episodes", ylabel="Cumulative Rewards")
    success_rate_log(categories=categories, values=policy_rewards, title = "Success rate log DQN vs Q-Learning", xlabel="Episodes", ylabel="Log(wins/episodes)", file_name="success_rate_log_dqn_vs_q_learning")

def train_and_run_q_learning_deterministic():
    fld = FrozenLakeDeterministic(epsilon=0.98, discount_factor=0.98, render=False, episodes_number=10000)
    Q, rewards, policy = fld.training()
    fld.render = True
    fld.episodes_number = 10
    _, win_counts, _ = fld.run_agent(policy)

def train_and_run_q_learning_non_deterministic():
    fld = FrozenLakeNonDeterministic(learning_rate = 0.1, epsilon=0.98, discount_factor=0.98, render=False, episodes_number=10000)
    Q, rewards, policy = fld.training()
    fld.render = True
    fld.episodes_number = 10
    _, win_counts, _ = fld.run_agent(policy)

def train_and_run_dqn_determinisitic():
    dqn = FrozenLakeDQN()
    policyDqn, rewards, _, _  = dqn.train(10000, False, False, "4x4")
    dqn.run_agent(10, policyDqn, False, True, "4x4")

def train_and_run_dqn_non_determinisitic():
    dqn = FrozenLakeDQN()
    policyDqn, rewards, _, _  = dqn.train(10000, False, True, "4x4")
    dqn.run_agent(10, policyDqn, True, True, "4x4")

def show_menu():
    print("\nMain Menu")
    print("1. Q-Learning Analysis")
    print("2. DQN vs Q-Learning Analysis")
    print("3. Train (10000 episodes) and run (10 episodes) a Q-Learning agent (Deterministic)")
    print("4. Train (10000 episodes) and run (10 episodes) a Q-Learning agent (Non-Deterministic)")
    print("5. Train (10000 episodes) and run (10 episodes) a DQN agent (Deterministic)")
    print("6. Train (10000 episodes) and run (10 episodes) a DQN agent (Non-Deterministic)")
    print("7. Exit")

def main():
    while True:
        show_menu()
        choice = input("Choose an action (1-7): ")
        if choice == '1':
            frozen_lake_non_deterministic_comparison(10000)
            frozen_lake_deterministic_comparison(10000)
        elif choice == '2':
            frozen_lake_dqn_vs_q_learning(10000)
        elif choice == '3':
            train_and_run_q_learning_deterministic()
        elif choice == '4':
            train_and_run_q_learning_non_deterministic()
        elif choice == '5':
            train_and_run_dqn_determinisitic()
        elif choice == '6':
            train_and_run_dqn_non_determinisitic()
        elif choice == '7':
            exit()
        else:
            print("Invalid choice. Please try again.")

        
    
   

    

   



    
if __name__ == "__main__":
    main()