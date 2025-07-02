import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DML_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
import snake_gym_env  # your custom Snake gym environment
from tqdm import trange, tqdm
from collections import deque

def create_dense_q_model(input_shape, num_actions):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
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
        time.sleep(0.05)
        done = terminated or truncated

    print(f"Final episode reward: {total_reward}, steps: {steps}")
    eval_env.close()

def main():
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

    env = gym.make("Snake-v0", render_mode=None)
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

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

    running_reward = 0

    progress = trange(n_episodes, desc="Training Episodes", dynamic_ncols=True)
    for episode_count in progress:
        obs, _ = env.reset(seed=None)
        obs = np.array(obs, dtype=np.float32)
        episode_reward = 0

        for t in range(max_steps_per_episode):
            step_count += 1

            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(model(tf.convert_to_tensor(np.expand_dims(obs, axis=0)), training=False).numpy())

            epsilon -= epsilon_interval / 1_000_000
            epsilon = max(epsilon, epsilon_min)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated

            action_history.append(action)
            state_history.append(obs)
            state_next_history.append(next_obs)
            rewards_history.append(reward)
            done_history.append(done)
            episode_reward += reward

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
                tqdm.write(f"Running reward: {running_reward:.2f} at episode {episode_count}, step count {step_count}")

            if done:
                break

        episode_reward_history.append(episode_reward)
        running_reward = np.mean(episode_reward_history) if episode_reward_history else 0.0

        if running_reward > 40:
            tqdm.write(f"Solved at episode {episode_count+1}!")
            break

    env.close()

    render_trained_agent(model, env_class="Snake-v0", grid_size=50)

if __name__ == "__main__":
    main()