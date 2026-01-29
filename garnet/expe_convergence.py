from garnet import GarnetEnv
from algos import sarsa, sarsa_det
from algos import constant, decreasing
from fed_algos import fed_sarsa_det, fed_sarsa
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import create_average_environment

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


import seaborn as sns
palette = sns.color_palette("colorblind")

S, A, B = 10, 3, 2
temperature = 2
alpha = 0.01
num_logs=1000

env = GarnetEnv(num_states=S, num_actions=A, branching_factor=B, seed=42)
env_list = [GarnetEnv(num_states=S, num_actions=A, branching_factor=B,
                      seed=1, generation_seed=0),
            GarnetEnv(num_states=S, num_actions=A, branching_factor=B,
                      seed=2,  generation_seed=1)]

print(np.max(env_list[1].rewards - env_list[0].rewards))

steps = 1000000
steps_det = 100000

N_list = [2, 10, 20, 50, 100]
H_list = [1, 100, 10000]

theta0 = np.zeros(S*A)
avg_env_2 = create_average_environment([env_list[0], env_list[1]])
theta_lim_fed = fed_sarsa_det(theta0, [env_list[0], env_list[1]], steps_det, H=1, alpha=0.1, gamma=0.9, temperature=temperature, seed=1, steptype=constant, verbose=True)
theta_lim_avg = sarsa_det(theta0, avg_env_2, steps_det, alpha=0.1, gamma=0.9, temperature=temperature, seed=1, steptype=constant, verbose=True)

print(theta_lim_fed)

print(theta_lim_avg)

seeds = np.arange(10)

print(theta_lim_fed)

rng = np.random.default_rng(seed=42)
theta0 = theta_lim_fed + rng.uniform(-10/np.sqrt(S*A), 10/np.sqrt(S*A), size=theta_lim_fed.shape)


for N in N_list:
    envs = [ GarnetEnv(num_states=S, num_actions=A, branching_factor=B,
                       seed=i, generation_seed=i%2)
             for i in range(N) ]    
    
    MSE_avg = {}
    MSE_fed = {}
    all_thetas = {}

    for H in H_list:
        print("----", N, H, "----")
        MSE_avg[H] = np.zeros((len(seeds), num_logs))
        MSE_fed[H] = np.zeros((len(seeds), num_logs))
        all_thetas[H] = np.zeros((len(seeds), num_logs, S*A))

        def run_one_seed(seed):
            _, _, thetas, _ = fed_sarsa(
                theta0, envs, steps, H=H, alpha=alpha,
                gamma=0.9, temperature=temperature,
                seed=seed, steptype=constant, verbose=False,
                num_logs=num_logs
            )
            mse_fed = np.linalg.norm(thetas - theta_lim_fed, axis=1) ** 2
            mse_avg = np.linalg.norm(thetas - theta_lim_avg, axis=1) ** 2
            return seed, mse_fed, mse_avg, thetas
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_one_seed, seed) for seed in seeds]
            for future in as_completed(futures):
                seed, mse_fed, mse_avg, thetas = future.result()
                MSE_fed[H][seed] = mse_fed
                MSE_avg[H][seed] = mse_avg
                all_thetas[H][seed] = thetas


    os.makedirs("results/S_" + str(S) + "/" + str(N) + "_" + str(alpha) + "/", exist_ok=True)
    with open("results/S_" + str(S) + "/" + str(N) + "_" + str(alpha) + "/expe_local_step.pickle", "wb") as f:
        pickle.dump({
            "theta_lim_avg": theta_lim_avg,
            "theta_lim_fed": theta_lim_fed,
            "thetas": all_thetas,
            "seeds": seeds,
            "H_list": H_list
        }, f)
        
   
