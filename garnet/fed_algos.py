import numpy as np
from tqdm import tqdm

from utils import *




def one_hot_featurize(state, action, size_feature, rng, num_actions):
    vec = np.zeros(size_feature)
    vec[state * num_actions + action] = 1.0
    return vec

def random_featurize(state, action, size_feature, rng, num_actions):
    vec = np.zeros(num_states * num_actions)
    vec[state * num_actions + action] = 1.0
    return vec

def q_value(env, theta, state, action, num_states, num_actions):
    return theta @ env.features[state][action]

def softmax(q_vals, temperature):
    q_vals = np.array(q_vals) / temperature
    exp_q = np.exp(q_vals - np.max(q_vals)) 
    return exp_q / np.sum(exp_q)



def compute_stationary_distribution(env, theta, temperature, num_states, num_actions, tol=1e-12, max_iter=10000):
    S, A = env.num_states, env.num_actions

    P_pi = np.zeros((S, S))

    for s in range(S):
        
        qs = [q_value(env, theta, s, a, num_states, num_actions) for a in range(num_actions)]
        policy = softmax(qs, temperature)
        
        for a in range(A):
            prob_a = policy[a]
            next_states = env.transitions[s][a]
            probs = env.probs[s][a]
            for idx, s_next in enumerate(next_states):
                P_pi[s, s_next] += prob_a * probs[idx]

    mu = np.ones(S) / S
    for _ in range(max_iter):
        mu_next = mu @ P_pi
        if np.linalg.norm(mu_next - mu, 1) < tol:
            break
        mu = mu_next

    return mu  


def get_next_action(env, theta, state, temperature, rng, num_states, num_actions):
    
    qs = [q_value(env, theta, state, a, num_states, num_actions) for a in range(num_actions)]
    probs = softmax(qs, temperature)

    return rng.choice(num_actions, p=probs)

def det_sarsa_update(env, theta, temperature, gamma, num_states, num_actions):
    
    update = np.zeros(len(theta))
    mu = compute_stationary_distribution(env, theta, temperature, num_states, num_actions)
        
    for state in range(num_states):
        p_s = mu[state]

        qs = [q_value(env, theta, state, a, num_states, num_actions) for a in range(num_actions)]
        policy = softmax(qs, temperature)

        for action in range(num_actions):

            pi_sa = policy[action]
            
            for index_next_state, next_state in enumerate(env.transitions[state][action]):

                next_qs = [q_value(env, theta, next_state, a,
                                   num_states, num_actions) for a in range(num_actions)]
                next_policy = softmax(next_qs, temperature)

                p_next_state = env.probs[state][action][index_next_state]
                for next_action in range(num_actions):
                    pi_sa_next = next_policy[next_action]
                    
                    phi = env.features[state, action]
                    phi_next = env.features[next_state, next_action]
                        
                    reward = env.rewards[state, action]
                    td_target = reward + gamma * (theta @ phi_next)
                    td_error = td_target - (theta @ phi)

                    update += p_s*pi_sa*p_next_state*pi_sa_next * td_error * phi


    return update



def create_average_environment(envs):
    new_env = deepcopy(envs[0])

    num_states = envs[0].num_states
    num_actions = envs[0].num_actions
    
    transitions = [ [ [] for a in range(num_actions)] for s in range(num_states)]
    probs = [ [ [] for a in range(num_actions)] for s in range(num_states)]
    rewards = np.zeros(envs[0].rewards.shape)

    for env in envs:
        # update rewards
        rewards += env.rewards

        # update transitions
        for s in range(num_states):
            for a in range(num_actions):
                for i, next_s in enumerate(env.transitions[s][a]):
                    try:
                        idx = transitions[s][a].index(next_s)
                        probs[s][a][idx] += env.probs[s][a][idx]
                    except:
                        transitions[s][a].append(next_s)
                        probs[s][a].append(env.probs[s][a][i])

    # normalized
    rewards /= len(envs)
    for s in range(num_states):
        for a in range(num_actions):
            probs[s][a] = np.array(probs[s][a])
            probs[s][a] /= np.sum(probs[s][a])

    new_env.transitions = transitions
    new_env.probs = probs
    new_env.rewards = rewards
    
    return new_env


def fed_sarsa_det(theta0, envs, steps, H, alpha=0.05, gamma=0.99, temperature=1.0, seed=42, steptype=constant, verbose=False):
    rng = np.random.default_rng(seed=seed)
    num_agents = len(envs)
    num_states, num_actions = envs[0].num_states, envs[0].num_actions
    theta = np.array([theta0.copy() for _ in range(num_agents)])

    update_size = np.inf
    last_update_size = -np.inf
    last_avg = 0
    
    for t in tqdm(range(steps)):
        print(last_update_size, update_size, alpha)

        for c in range(num_agents):
            update = det_sarsa_update(envs[c], theta[c], temperature, gamma, num_states, num_actions)

            theta[c] += alpha * update
            
        if t % H == 0:
            avg_theta = np.mean(theta, axis=0)
            theta = np.zeros(theta.shape) + avg_theta[None, :]

            last_update_size = update_size
            update_size = np.linalg.norm(avg_theta - last_avg) / alpha
            last_avg = avg_theta.copy()

            # if update_size < last_update_size:
            #     alpha *= (1 + 0.1 * np.log(H))
            # else:
            #     alpha /= 2
                
            # if alpha < 1e-50:
            #     return -1


            
            if update_size < 1e-14:
                break         


        if t % (steps//100) == 0:
            if verbose:
                print(update_size, alpha)
                print(avg_theta)

    return np.mean(theta, axis=0)


def fed_sarsa(theta0, envs, steps, H, alpha=0.05, gamma=0.99, temperature=1.0, seed=42, steptype=constant, verbose=False, num_logs=10000):
    rng = np.random.default_rng(seed=seed)
    num_agents = len(envs)
    num_states, num_actions = envs[0].num_states, envs[0].num_actions
    theta = np.array([theta0.copy() for _ in range(num_agents)])


    log_times = np.linspace(0, steps, num_logs, dtype=int)
    current_log = 1
    
   
    theta_hist = np.zeros((len(log_times), num_states * num_actions))
    theta_hist[0] = np.mean(theta, axis=0).copy()
    
    state = [env.reset() for env in envs]
    action = [get_next_action(envs[c], theta[c], state[c], temperature,
                              rng, num_states, num_actions)
              for c in range(num_agents)]

    phi = [envs[c].features[state[c], action[c]] for c in range(num_agents)]
    
    rewards = []

    for t in tqdm(range(steps)):

        rewards.append(0)

        for c in range(num_agents):
            next_state, reward = envs[c].step(action[c])
            next_action = get_next_action(envs[c], theta[c], next_state,
                                          temperature, rng,
                                          num_states, num_actions)
            rewards[-1] += reward

            phi_next = envs[c].features[next_state, next_action]

            td_target = reward + gamma * (theta[c] @ phi_next)
            td_error = td_target - (theta[c] @ phi[c])
            theta[c] += steptype(t, alpha) * td_error * phi[c]
            state[c] = next_state
            action[c] = next_action
            phi[c] = phi_next.copy()

        if t % H == 0:
           avg_theta = np.mean(theta, axis=0)
           for c in range(num_agents):
               theta[c] = avg_theta.copy()

        if t >= log_times[current_log]:
            theta_hist[current_log] = np.mean(theta, axis=0)
            current_log += 1
            
    return rewards, theta, theta_hist, log_times
