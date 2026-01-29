import os
import pickle

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import gymnax
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Import SARSA
# ---------------------------------------------------------
from sarsa_fct import sarsa 


# ---------------------------------------------------------
# 0. Config
# ---------------------------------------------------------
ENV_NAME = "CartPole-v1"

num_envs = 100
num_episodes = 500
batch_size = 32
buffer_size = 5000
gamma = 0.99
tau = 1.0
epsilon_decay = 0.99
min_epsilon = 0.001
learning_rate = 1e-3
seed = 42

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
max_length = 500
env, env_params = gymnax.make("CartPole-v1")

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


model = QNetwork(action_dim=env.num_actions)
optimizer = optax.adam(learning_rate)

# ---------------------------------------------------------
# Run SARSA
# ---------------------------------------------------------
rng = jax.random.PRNGKey(seed)

results = sarsa(
    rng=rng,
    env=env,
    env_params=env_params,
    model=model,
    optimizer=optimizer,
    num_envs=num_envs,
    num_episodes=num_episodes,
    max_length=max_length,
    buffer_size=buffer_size,
    batch_size=batch_size,
    gamma=gamma,
    tau=tau,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon,
)

all_rewards = results["all_rewards"]


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
path = f"results/{ENV_NAME}"
os.makedirs(path, exist_ok=True)

results_data = {
    "env_name": ENV_NAME,
    "all_rewards": jax.device_get(all_rewards),
    "hyperparameters": {
        "num_envs": num_envs,
        "num_episodes": num_episodes,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "learning_rate": learning_rate,
    },
}

file_path = os.path.join(path, "results.pickle")
with open(file_path, "wb") as f:
    pickle.dump(results_data, f)

print(f"Results saved to {file_path}")
