import numpy as np
import os
import time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
import snake_gym_env  # your custom Snake gym environment
from tqdm import trange
from collections import deque

def create_dense_q_model(input_shape, num_actions):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])

def render_trained_agent(trained_model, env_class="Snake-v0", grid_size=20):
    eval_env = gym.make(env_class, render_mode="human", grid_size=grid_size)
    obs, _ = eval_env.reset()
    obs = np.array(obs, dtype=np.float32)
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = np.argmax(trained_model(np.expand_dims(obs, axis=0), training=False).numpy())
        obs, reward, terminated, truncated, info = eval_env.step(action)
        obs = np.array(obs, dtype=np.float32)
        total_reward += reward
        steps += 1
        eval_env.render()
        time.sleep(0.05)  # Control rendering speed
        done = terminated or truncated

    print(f"Final episode reward: {total_reward}, steps: {steps}")
    eval_env.close()

def main():
    NUM_ENVS = 2048
    n_episodes = 100_000
    max_steps_per_episode = 500
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = epsilon_max - epsilon_min
    batch_size = 256
    max_memory_length = 100_000
    update_after_actions = 4
    update_target_network = 1000

    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("Snake-v0", render_mode=None)
        for _ in range(NUM_ENVS)
    ])
    num_actions = env.single_action_space.n
    input_shape = env.single_observation_space.shape

    model = create_dense_q_model(input_shape, num_actions)
    model_target = create_dense_q_model(input_shape, num_actions)
    model_target.set_weights(model.get_weights())
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    action_history = deque(maxlen=max_memory_length)
    state_history = deque(maxlen=max_memory_length)
    state_next_history = deque(maxlen=max_memory_length)
    rewards_history = deque(maxlen=max_memory_length)
    done_history = deque(maxlen=max_memory_length)
    episode_reward_history = deque(maxlen=100)
    step_count = 0
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

    @tf.function
    def train_step(state_sample, action_sample, rewards_sample, state_next_sample, done_sample):
        future_rewards = model_target(state_next_sample, training=False)
        max_future_q = tf.reduce_max(future_rewards, axis=1)
        updated_q_values = rewards_sample + gamma * max_future_q
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        masks = tf.one_hot(action_sample, num_actions)
        with tf.GradientTape() as tape:
            q_values = model(state_sample)
            q_action = tf.reduce_sum(q_values * masks, axis=1)
            loss = loss_function(updated_q_values, q_action)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    obs, _ = env.reset(seed=None)
    obs = np.array(obs, dtype=np.float32)
    running_reward = 0

    for episode_count in trange(n_episodes, desc="Training Episodes"):
        episode_rewards[:] = 0
        dones = np.zeros(NUM_ENVS, dtype=bool)
        obs, _ = env.reset(seed=None)
        obs = np.array(obs, dtype=np.float32)

        for t in range(max_steps_per_episode):
            step_count += NUM_ENVS

            rand_vals = np.random.rand(NUM_ENVS)
            greedy_actions = np.argmax(
                model(tf.convert_to_tensor(obs), training=False).numpy(), axis=1
            )
            random_actions = np.random.randint(num_actions, size=NUM_ENVS)
            actions = np.where(rand_vals < epsilon, random_actions, greedy_actions)

            epsilon -= epsilon_interval / 1_000_000
            epsilon = max(epsilon, epsilon_min)

            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            next_obs = np.array(next_obs, dtype=np.float32)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(NUM_ENVS):
                action_history.append(actions[i])
                state_history.append(obs[i])
                state_next_history.append(next_obs[i])
                rewards_history.append(rewards[i])
                done_history.append(dones[i])
                episode_rewards[i] += rewards[i]

            obs = next_obs

            if step_count % update_after_actions == 0 and len(done_history) > batch_size:
                indices = np.random.choice(len(done_history), size=batch_size, replace=False)
                state_sample = tf.convert_to_tensor(np.array([state_history[i] for i in indices]), dtype=tf.float32)
                state_next_sample = tf.convert_to_tensor(np.array([state_next_history[i] for i in indices]), dtype=tf.float32)
                rewards_sample = tf.convert_to_tensor(np.array([rewards_history[i] for i in indices]), dtype=tf.float32)
                action_sample = tf.convert_to_tensor(np.array([action_history[i] for i in indices]), dtype=tf.int32)
                done_sample = tf.convert_to_tensor(np.array([float(done_history[i]) for i in indices]), dtype=tf.float32)
                train_step(state_sample, action_sample, rewards_sample, state_next_sample, done_sample)

            if step_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                running_reward = np.mean(episode_reward_history) if episode_reward_history else 0.0
                print(f"Running reward: {running_reward:.2f} at episode {episode_count}, step count {step_count}")

            for i in range(NUM_ENVS):
                if dones[i]:
                    episode_reward_history.append(episode_rewards[i])
                    episode_rewards[i] = 0

        running_reward = np.mean(episode_reward_history) if episode_reward_history else 0.0

        if running_reward > 40:
            print(f"Solved at episode {episode_count+1}!")
            break

    env.close()

    # --- Render the trained agent once after training ---
    render_trained_agent(model, env_class="Snake-v0", grid_size=50)  # Change grid_size if needed

if __name__ == "__main__":
    main()