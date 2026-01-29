import numpy as np
from tqdm import tqdm

from utils import *


def sarsa_det(theta0, env, steps, alpha=0.05, gamma=0.99, temperature=1.0, seed=42, steptype=constant, verbose=False):
    rng = np.random.default_rng(seed=seed)
    num_states, num_actions = env.num_states, env.num_actions
    theta = theta0.copy()


    update_size = np.inf
    last_update_size = -np.inf
    
    for t in tqdm(range(steps)):
        print(last_update_size, update_size, alpha)

        if update_size < last_update_size:
            alpha *= 1.1
        else:
            alpha /= 2

        if update_size < 1e-14:
            break

        if alpha < 1e-50:
            return -1

        update = det_sarsa_update(env, theta, temperature, gamma, num_states, num_actions)

        last_update_size = update_size
        update_size = np.linalg.norm(update)
        
        theta += alpha * update
        
        if t % (steps//100) == 0:
            if verbose:
                print(update_size, alpha)
                print(theta)

    return theta


def sarsa(theta0, env, steps, alpha=0.05, gamma=0.99, temperature=1.0, seed=42, steptype=constant, num_logs=10000):
    rng = np.random.default_rng(seed=seed)
    num_states, num_actions = env.num_states, env.num_actions
    theta = theta0.copy()

    log_times = np.linspace(0, steps, num_logs, dtype=int)
    current_log = 1

    theta_hist = np.zeros((len(log_times), num_states * num_actions))

    theta_hist[0] = theta.copy()

    
    state = env.reset()
    action = get_next_action(env, theta, state, temperature, rng, num_states, num_actions)
    phi = env.features[state, action]
    
    rewards = []

    for t in tqdm(range(steps)):
        next_state, reward = env.step(action)
        next_action = get_next_action(env, theta, next_state, temperature, rng, num_states, num_actions)
        rewards.append(reward)

        phi_next = env.features[next_state, next_action]

        td_target = reward + gamma * (theta @ phi_next)
        td_error = td_target - (theta @ phi)
        theta += steptype(t, alpha) * td_error * phi

        state, action = next_state, next_action

        phi = phi_next

        if t >= log_times[current_log]:
            theta_hist[current_log] = theta.copy()
            current_log += 1

    return rewards, theta, theta_hist, log_times
