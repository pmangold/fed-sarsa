import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import gymnax
import matplotlib.pyplot as plt
from functools import partial
from typing import NamedTuple
import pickle
import os

import argparse

from garnet_env import Garnet

# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Run FedSARSA on a Gymnax environment")
parser.add_argument(
    "--env", type=str, default="cartpole",
    help="Name of the environment (e.g., cartpole, pendulum, mountaincar)"
)
parser.add_argument(
    "--rep", type=int, default=100,
    help="Number of repetitions"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility"
)
parser.add_argument(
    "--agent", type=int, default=10,
    help="Number of agents"
)
parser.add_argument(
    "--local", type=int, default=1,
    help="Number of local steps"
)
args = parser.parse_args()

ENV_NAME = args.env
SEED = args.seed
LOCAL = args.local
AGENTS = args.agent
REP = args.rep


path = f"results/federated_{ENV_NAME}"
os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
num_reps = REP
num_agents = AGENTS
num_envs = num_reps * num_agents
batch_size = 32
buffer_size = 5000
gamma = 0.99
tau = 1  # Target network soft update rate
epsilon_decay = 0.99
min_epsilon = 0.001
learning_rate = 1e-3
comm_iter = LOCAL  # communication interval (episodes)


num_episodes = 500
max_length = 500
env, env_params = gymnax.make("CartPole-v1")


# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------
class QNetwork(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

model = QNetwork(action_dim=env.num_actions)
optimizer = optax.adam(learning_rate=learning_rate)

    
# ---------------------------------------------------------
# 2. Replay Buffer (Vectorized State)
# ---------------------------------------------------------
class ReplayBufferState(NamedTuple):
    ptr: jnp.ndarray
    full: jnp.ndarray
    obs_buf: jnp.ndarray
    next_obs_buf: jnp.ndarray
    actions: jnp.ndarray
    next_actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

def init_buffer(size, obs_dim, num_envs):
    return ReplayBufferState(
        ptr=jnp.zeros(num_envs, dtype=jnp.int32),
        full=jnp.zeros(num_envs, dtype=jnp.bool_),
        obs_buf=jnp.zeros((num_envs, size, obs_dim), dtype=jnp.float32),
        next_obs_buf=jnp.zeros((num_envs, size, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros((num_envs, size), dtype=jnp.int32),
        next_actions=jnp.zeros((num_envs, size), dtype=jnp.int32),
        rewards=jnp.zeros((num_envs, size), dtype=jnp.float32),
        dones=jnp.zeros((num_envs, size), dtype=jnp.float32),
    )

# Single-agent logic to be vmapped
def _buffer_add_single(buffer, obs, action, reward, next_obs, next_action, done):
    idx = buffer.ptr
    size = buffer.obs_buf.shape[0]
    return buffer._replace(
        obs_buf=buffer.obs_buf.at[idx].set(obs),
        next_obs_buf=buffer.next_obs_buf.at[idx].set(next_obs),
        actions=buffer.actions.at[idx].set(action),
        next_actions=buffer.next_actions.at[idx].set(next_action),
        rewards=buffer.rewards.at[idx].set(reward),
        dones=buffer.dones.at[idx].set(done),
        ptr=(idx + 1) % size,
        full=buffer.full | ((idx + 1) % size == 0)
    )

def _buffer_sample_single(buffer, rng, batch_size):
    max_idx = jax.lax.cond(buffer.full, lambda: buffer.obs_buf.shape[0], lambda: buffer.ptr)
    idxs = jax.random.randint(rng, (batch_size,), 0, max_idx)
    return (
        buffer.obs_buf[idxs], buffer.actions[idxs], buffer.rewards[idxs],
        buffer.next_obs_buf[idxs], buffer.next_actions[idxs], buffer.dones[idxs]
    )

# ---------------------------------------------------------
# 3. Vectorized Core Functions
# ---------------------------------------------------------
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0))
def v_buffer_add(buffer, obs, action, reward, next_obs, next_action, done):
    return _buffer_add_single(buffer, obs, action, reward, next_obs, next_action, done)

@partial(jax.vmap, in_axes=(0, 0, None))
def v_buffer_sample(buffer, rng, batch_size):
    return _buffer_sample_single(buffer, rng, batch_size)

@partial(jax.jit, static_argnums=(4,))
@partial(jax.vmap, in_axes=(0, 0, None, 0, None))
def v_select_action(params, obs, epsilon, rng, model):
    q_values = model.apply({'params': params}, obs)
    rng_eps, rng_act = jax.random.split(rng)
    is_random = jax.random.uniform(rng_eps, ()) < epsilon
    return jnp.where(is_random, jax.random.randint(rng_act, (), 0, q_values.shape[-1]), jnp.argmax(q_values, axis=-1))

@partial(jax.jit, static_argnums=(9,))
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
def v_train_step(params, target_params, opt_state, obs, actions, rewards, next_obs, next_actions, done_mask, model, gamma):
    def loss_fn(p):
        q_sa = jnp.take_along_axis(model.apply({'params': p}, obs), actions[:, None], axis=-1).squeeze()
        q_next = jnp.take_along_axis(model.apply({'params': target_params}, next_obs), next_actions[:, None], axis=-1).squeeze()
        target = rewards + (1.0 - done_mask) * gamma * q_next
        return jnp.mean((q_sa - jax.lax.stop_gradient(target))**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ---------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------
def federated_aggregate_blockwise(trees, block_size):
    """Aggregate parameters for each block of size block_size separately."""
    def avg_block(leaf):
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf
        # reshape to (num_blocks, block_size, ...)
        new_shape = (leaf.shape[0] // block_size, block_size) + leaf.shape[1:]
        leaf_reshaped = leaf.reshape(new_shape)
        # compute mean per block
        mean_block = jnp.mean(leaf_reshaped, axis=1, keepdims=True)
        # broadcast mean back
        return jnp.broadcast_to(mean_block, leaf_reshaped.shape).reshape(leaf.shape)
    return jax.tree_util.tree_map(avg_block, trees)


# ---------------------------------------------------------
# Rollout step
# ---------------------------------------------------------
def rollout_step(carry, _):
    rng, buffer, obs, action, epsilon, params, target_params, opt_state, env_state, not_done = carry
    rng, rng_step, rng_sample, rng_act = jax.random.split(rng, 4)

    # Environment step
    next_obs, next_env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0,0,0,None))(
        jax.random.split(rng_step, num_envs), env_state, action, env_params
    )

    # Next action
    next_action = v_select_action(params, next_obs, epsilon, jax.random.split(rng_act, num_envs), model)

    # Add to buffer
    new_buffer = v_buffer_add(buffer, obs, action, reward, next_obs, next_action, done)

    # Local training if enough samples
    def do_train(inputs):
        p, tp, s, b, r_batch = inputs
        batch = v_buffer_sample(b, r_batch, batch_size)
        p, s, _ = v_train_step(p, tp, s, *batch, model, gamma)
        return p, s

    params, opt_state = jax.lax.cond(
        new_buffer.ptr[0] > batch_size,
        do_train,
        lambda x: (x[0], x[2]),
        (params, target_params, opt_state, new_buffer, jax.random.split(rng_sample, num_envs))
    )

    next_not_done = not_done & (~done)
    return (rng, new_buffer, next_obs, next_action, epsilon, params, target_params, opt_state, next_env_state, next_not_done), (reward * not_done)


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
rng = jax.random.PRNGKey(SEED)
rng, rng_init = jax.random.split(rng)
init_obs, _ = env.reset(rng_init, env_params)

params = jax.vmap(lambda r: model.init(r, init_obs)['params'])(jax.random.split(rng_init, num_envs))
target_params = params
opt_state = jax.vmap(optimizer.init)(params)
buffer = init_buffer(buffer_size, init_obs.shape[0], num_envs)

all_block_rewards = jnp.zeros((num_episodes, num_reps, num_agents))
epsilon = 1.0


# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
for ep in range(num_episodes):
    rng, rng_reset, rng_act = jax.random.split(rng, 3)
    obs, env_state = jax.vmap(env.reset, in_axes=(0,None))(jax.random.split(rng_reset, num_envs), env_params)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    action = v_select_action(params, obs, epsilon, jax.random.split(rng_act, num_envs), model)

    carry = (rng, buffer, obs, action, epsilon, params, target_params, opt_state, env_state, jnp.ones(num_envs, dtype=jnp.bool_))
    carry, rewards = jax.lax.scan(rollout_step, carry, None, length=max_length)

    rng, buffer, _, _, _, params, target_params, opt_state, _, _ = carry

    # Federated averaging across agents every episode (or comm_iter)
    if (ep + 1) % comm_iter == 0:

        params = federated_aggregate_blockwise(params, num_agents)
        opt_state = federated_aggregate_blockwise(opt_state, num_agents)

    # Target network update
    target_params = jax.tree_util.tree_map(lambda p,tp: p*tau + tp*(1-tau), params, target_params)

    # Sum over timesteps → total reward per environment
    episode_rewards = jnp.sum(rewards, axis=0)  # shape: (total_envs,)

    # Reshape by blocks → (num_reps, num_envs)
    rewards_blocked = episode_rewards.reshape(num_reps, num_agents)
    
    # Store per-block rewards
    all_block_rewards = all_block_rewards.at[ep].set(rewards_blocked)

    # Optional: compute per-block summary
    block_rewards = jnp.mean(rewards_blocked, axis=1)  # total reward per block
    mean_reward = jnp.mean(block_rewards)
    std_rewards = jnp.std(block_rewards)
    
    if ep % 10 == 0:
        print(f"Episode {ep} | Mean Reward: {mean_reward:.2f} +- {std_rewards:.2f} | Epsilon: {epsilon:.3f}")

    

# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------
results_data = {
    "env_name": ENV_NAME,
    "all_rewards": jax.device_get(all_block_rewards),
    "hyperparameters": {
        "num_envs": num_reps * num_agents,
        "num_reps": num_reps,
        "num_agents": num_agents,
        "num_episodes": num_episodes,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "comm_iter": comm_iter
    }
}

file_name = "results_env" + str(num_envs) + "_ag" + str(num_agents) + "_loc" + str(comm_iter) + "_ep" + str(num_episodes) + ".pickle"
file_path = os.path.join(path, file_name)
with open(file_path, "wb") as f:
    pickle.dump(results_data, f)

print(f"Results successfully saved to {file_path}")
