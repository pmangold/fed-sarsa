import pickle
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
palette = sns.color_palette("colorblind")

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.size": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(1, 1, figsize=(4,3)) 
ax.plot([0, 1], [0, 1])
ax.set_xlabel("test")
fig.savefig("test.pdf", bbox_inches="tight")

ENV_NAME = "cartpole"


folder_path_centralized = f"results/{ENV_NAME}"
folder_path = f"results/federated_{ENV_NAME}"
os.makedirs(os.path.join("plots/", folder_path), exist_ok=True)


def plot_results(env_name: str, result_path: str, label: str,
                 federated=False, ax=None, alpha_comm=0, **kwargs):
    # 1. Path Setup
    if federated:
        file_path = os.path.join(folder_path, result_path)
    else:
        file_path = os.path.join(folder_path_centralized, result_path)
    
    if not os.path.exists(file_path):
        print(f"Error: No results found at {file_path}")
        return

    # 2. Load Data
    with open(file_path, "rb") as f:
        data = pickle.load(f)


        
    
    raw_rewards = data["all_rewards"]
    hparams = data["hyperparameters"]
    
    if federated:
        print(data)
        num_episodes = hparams["num_episodes"]
        num_reps = hparams["num_reps"]
        num_agents = hparams["num_agents"]

        # Reshape to (episodes, repetitions, agents)
        # Then average over agents to get the performance per repetition
        repro_data = raw_rewards.reshape((num_episodes, num_reps, num_agents))
        all_rewards = np.mean(repro_data, axis=2) # Shape: (num_episodes, num_reps)
    else:
        all_rewards = raw_rewards # Shape: (num_episodes, num_envs)
        
    num_episodes, num_envs = all_rewards.shape
    comm_iter = data["hyperparameters"].get("comm_iter", 1)


    def moving_average(x, window=5):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    mean_rewards = np.asarray(jnp.mean(all_rewards, axis=1))
    std_rewards  = np.asarray(jnp.std(all_rewards, axis=1))
    min_rewards  = np.asarray(jnp.min(all_rewards, axis=1))
    max_rewards  = np.asarray(jnp.max(all_rewards, axis=1))

    window = 1

    mean_s = moving_average(mean_rewards, window)
    std_s  = moving_average(std_rewards, window)
    min_s  = moving_average(min_rewards, window)
    max_s  = moving_average(max_rewards, window)

    episodes_s = np.arange(len(mean_s))

    num_points = 50
    idx = np.linspace(0, len(mean_s) - 1, num_points).astype(int)
    
    episodes_ds = episodes_s[idx]
    mean_ds = mean_s[idx]
    std_ds  = std_s[idx]
    min_ds  = min_s[idx]
    max_ds  = max_s[idx]

    lower = np.maximum(mean_ds - std_ds, min_ds)
    upper = np.minimum(mean_ds + std_ds, max_ds)

    ax.plot(episodes_ds, mean_ds, lw=2, label=label, **kwargs)
    ax.fill_between(episodes_ds, lower, upper, alpha=0.3, **kwargs)



    if federated:
        for comm_ep in range(comm_iter - 1, num_episodes, comm_iter):
            ax.axvline(x=comm_ep, color='red', linestyle='--', alpha=alpha_comm, lw=1)

            
    ax.set_xlim(0, num_episodes)
    ax.set_xticks(np.linspace(0, num_episodes, 6))
    

if __name__ == "__main__":

    results_cent_path = "results_env100_ep500.pickle"
    


    # num_agents = 100
    num_reps = 10
    num_loc = [1, 10, 100]

    for num_agents in [2, 10, 50, 100]:

        num_envs = num_reps * num_agents
        
        fig, ax = plt.subplots(1, 1, figsize=(4,3))

        label = 'Centralized'
        plot_results(ENV_NAME, result_path=results_cent_path, label=label, federated=False, color=palette[0], ax=ax)

        
        for i, loc in enumerate(num_loc):

            results_path = "results_env" + str(num_envs) + "_ag" + str(num_agents) + "_loc" + str(loc) + "_ep500.pickle"
    
        
            label = 'H=' + str(loc)
            plot_results(ENV_NAME, result_path=results_path, label=label, federated=True, color=palette[1+i], ax=ax)


        if num_agents == 100:
            ax.legend(loc='lower right', fontsize=16)

        # 8. Save Plot
        plot_save_path = os.path.join("plots/", folder_path, "training_plot" + str(num_agents) + ".pdf")
        plt.savefig(plot_save_path, bbox_inches="tight")
        plot_save_path = os.path.join("plots/", folder_path, "training_plot" + str(num_agents) + ".png")
        plt.savefig(plot_save_path, bbox_inches="tight")
        print(f"Plot saved to: {plot_save_path}")


