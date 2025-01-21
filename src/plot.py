import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def cumulative_rewards_chart(categories: List, values: List, title: str, xlabel: str, ylabel: str, file_name):
    
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