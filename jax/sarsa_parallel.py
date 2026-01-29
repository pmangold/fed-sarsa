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
parser = argparse.ArgumentParser(description="Run SARSA on a Gymnax environment")
parser.add_argument(
    "--env", type=str, default="cartpole",
    help="Name of the environment (e.g., cartpole, pendulum, mountaincar)"
)
args = parser.parse_args()
ENV_NAME = args.env


path = f"results/{ENV_NAME}"
os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
num_envs = 100
batch_size = 32
buffer_size = 5000
gamma = 0.99
tau = 1  # Target network soft update rate
epsilon_decay = 0.99
min_epsilon = 0.001
learning_rate = 1e-3


if ENV_NAME == "cartpole":
    num_episodes = 500
    max_length = 500
    env, env_params = gymnax.make("CartPole-v1")
elif ENV_NAME == "pendulum":
    num_episodes = 500
    max_length = 200
    env, env_params = gymnax.make("Pendulum-v1")
elif ENV_NAME == "garnet":
    num_episodes = 1000
    max_length = 100
    env = Garnet(num_states=100, num_actions=4, branching_factor=3)
    env_params = env.default_params
else:
    num_episodes = 2000
    max_length = 200
    env, env_params = gymnax.make("MountainCar-v0")

print(ENV_NAME, "on", num_envs)

# ---------------------------------------------------------
# Model and Optimizer Definition
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
optimizer = optax.adam(learning_rate=learning_rate)

    
# ---------------------------------------------------------
# Replay Buffer (Vectorized State)
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
# Vectorized Core Functions
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
# Rollout & Training Loop Logic
# ---------------------------------------------------------
def rollout_step(carry, _):
    rng, buffer, obs, action, epsilon, params, target_params, opt_state, env_state, not_done = carry
    rng, rng_step, rng_sample, rng_act = jax.random.split(rng, 4)

    # 1. Environment step (Vectorized)
    next_obs, next_env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(rng_step, num_envs), env_state, action, env_params
    )

    # 2. Select next action (Vectorized)
    next_action = v_select_action(params, next_obs, epsilon, jax.random.split(rng_act, num_envs), model)

    # 3. Add to independent buffers
    new_buffer = v_buffer_add(buffer, obs, action, reward, next_obs, next_action, done)

    # 4. Independent Training
    def do_train(inputs):
        p, tp, s, b, r_batch = inputs
        batch = v_buffer_sample(b, r_batch, batch_size)
        p, s, _ = v_train_step(p, tp, s, *batch, model, gamma)
        return p, s

    params, opt_state = jax.lax.cond(
        new_buffer.ptr[0] > batch_size, # Train once buffer starts filling
        do_train,
        lambda x: (x[0], x[2]),
        (params, target_params, opt_state, new_buffer, jax.random.split(rng_sample, num_envs))
    )

    next_not_done = not_done & (~done)
    return (rng, new_buffer, next_obs, next_action, epsilon, params, target_params, opt_state, next_env_state, next_not_done), (reward * not_done)


# ---------------------------------------------------------
# Initialization & Execution
# ---------------------------------------------------------
rng = jax.random.PRNGKey(42)
rng, rng_init = jax.random.split(rng)
init_obs, _ = env.reset(rng_init, env_params)

# Vectorized Initialization
params = jax.vmap(lambda r: model.init(r, init_obs)['params'])(jax.random.split(rng_init, num_envs))
target_params = params
opt_state = jax.vmap(optimizer.init)(params)
buffer = init_buffer(buffer_size, init_obs.shape[0], num_envs)

all_rewards = jnp.zeros((num_episodes, num_envs))
epsilon = 1.0

for ep in range(num_episodes):
    rng, rng_reset, rng_act = jax.random.split(rng, 3)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rng_reset, num_envs), env_params)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    action = v_select_action(params, obs, epsilon, jax.random.split(rng_act, num_envs), model)
    
    # Run Episode via Scan
    carry = (rng, buffer, obs, action, epsilon, params, target_params, opt_state, env_state, jnp.ones(num_envs, dtype=jnp.bool_))
    carry, rewards = jax.lax.scan(rollout_step, carry, None, length=max_length)
    
    # Finalize episode state
    rng, buffer, _, _, _, params, target_params, opt_state, _, _ = carry
    target_params = jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), params, target_params)
    
    episode_rewards = jnp.sum(rewards, axis=0)
    mean_reward = jnp.mean(jnp.sum(rewards, axis=0))
    std_rewards = jnp.std(jnp.sum(rewards, axis=0))
    all_rewards = all_rewards.at[ep].set(episode_rewards)

    if ep % 10 == 0:
        print(f"Episode {ep} | Mean Reward: {mean_reward:.2f} +- {std_rewards:.2f} | Epsilon: {epsilon:.3f}")



        
results_data = {
    "env_name": ENV_NAME,
    "all_rewards": jax.device_get(all_rewards),
    "hyperparameters": {
        "num_envs": num_envs,
        "num_episodes": num_episodes,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay
    }
}

# 3. Save the pickle file
file_name = "results_env" + str(num_envs) + "_ep" + str(num_episodes) + ".pickle"
print(file_name)
file_path = os.path.join(path, file_name)
with open(file_path, "wb") as f:
    pickle.dump(results_data, f)

print(f"Results successfully saved to {file_path}")
