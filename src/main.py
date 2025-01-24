from q_learning import FrozenLakeDeterministic, FrozenLakeNonDeterministic
from plot import heatmap, log_epsilon_analysis, log_gamma_analysis, log_alpha_analysis, learned_policy_deterministic_comparison, plot_cumulative_rewards, success_rate_log

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

def frozen_lake_dqn_deterministic_comparison(iterations):
    
    categories = []
    values = []
    policy_rewards = []
    
    
    fld = FrozenLakeDQN(discount_factor=0.98)
    policy, rewards  = fld.train(iterations, False, True)
    categories.append(f"{gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    # _, win_counts, _ = fld.run_agent(policy)
    # policy_rewards.append(win_counts)

    fld = FrozenLakeDQN(discount_factor=0.1)
    policy, rewards  = fld.train(iterations, False, True)
    categories.append(f"{gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    # _, win_counts, _ = fld.run_agent(policy)
    # policy_rewards.append(win_counts)
    
    fld = FrozenLakeDQN(discount_factor=0.5)
    policy, rewards  = fld.train(iterations, False, True)
    categories.append(f"{gamma_label}={fld.discount_factor}, {alpha_label}={fld.learning_rate}")
    values.append(rewards)
    # _, win_counts, _ = fld.run_agent(policy)
    # policy_rewards.append(win_counts)

    plot_cumulative_rewards(categories=categories, values=values, title="Deterministic DQN Frozen Lake (Cumulative Rewards Comparison)", xlabel="Episodes", ylabel="Cumulative Rewards", file_name="cumulative_rewards_deterministic_comparison_dqn")
    #success_rate_log(categories=categories, values=policy_rewards, title = "Success rate log DQN", xlabel="Episodes", ylabel="Log(wins/episodes)", file_name="success_rate_log_dqn")

def frozen_lake_dqn_non_deterministic_comparison(iterations):
    pass

def frozen_lake_loss_analysis():
    pass

def main():
    
    
    #frozen_lake_deterministic_comparison(10500)

    #frozen_lake_non_deterministic_comparison(15000)
    #frozen_lake_dqn_deterministic_comparison(15000)

    dqn = FrozenLakeDQN()
    policy, rewards_per_episodes, epsilon =  dqn.train(1000, False, False)
    print(epsilon)
    dqn.test(10, policy, False)

    

    # print(f"win rate = {100 * fl.run_agent(policy=policy)[0]/fl.episodes_number}")



    
if __name__ == "__main__":
    main()