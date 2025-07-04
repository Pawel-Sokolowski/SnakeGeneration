import itertools
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
import snake_gym_env
from tqdm import trange, tqdm
from collections import deque
import matplotlib.pyplot as plt  # <-- ADD THIS

def create_dense_q_model(input_shape, num_actions):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(8, activation='sigmoid'),
        layers.Dense(16, activation='sigmoid'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(num_actions, activation='softmax')
    ])

def param_grid_search(param_grid):
    keys, values = zip(*param_grid.items())
    for bundle in itertools.product(*values):
        yield dict(zip(keys, bundle))

def main():
    param_grid = {
        "n_episodes": [100_000],
        "gamma": [0.95, 0.99],
        "epsilon": [1.0],
        "epsilon_min": [0.1],
        "batch_size": [128, 256, 512, 1024],
        "learning_rate": [0.00025, 0.0005, 0.005, 0.01],
    }

    for params in param_grid_search(param_grid):
        n_episodes = params["n_episodes"]
        max_steps_per_episode = 500
        gamma = params["gamma"]
        epsilon = params["epsilon"]
        epsilon_min = params["epsilon_min"]
        epsilon_max = epsilon
        epsilon_interval = epsilon_max - epsilon_min
        batch_size = params["batch_size"]
        max_memory_length = 100_000
        update_after_actions = 4
        update_target_network = 1000
        learning_rate = params["learning_rate"]

        env = gym.make("Snake-v0", render_mode=None)
        num_actions = env.action_space.n
        input_shape = env.observation_space.shape

        model = create_dense_q_model(input_shape, num_actions)
        model_target = create_dense_q_model(input_shape, num_actions)
        model_target.set_weights(model.get_weights())
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        loss_function = keras.losses.Huber()

        action_history = deque(maxlen=max_memory_length)
        state_history = deque(maxlen=max_memory_length)
        state_next_history = deque(maxlen=max_memory_length)
        rewards_history = deque(maxlen=max_memory_length)
        done_history = deque(maxlen=max_memory_length)
        episode_reward_history = deque(maxlen=100)
        all_episode_rewards = []  # <-- ADD THIS
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
        progress = trange(n_episodes, desc=f"gamma={gamma},lr={learning_rate}", dynamic_ncols=True)
        for episode_count in progress:
            obs, _ = env.reset(seed=None)
            obs = np.array(obs, dtype=np.float32)
            episode_reward = 0
            e = epsilon

            for t in range(max_steps_per_episode):
                step_count += 1

                if np.random.rand() < e:
                    action = np.random.randint(num_actions)
                else:
                    action = np.argmax(model(tf.convert_to_tensor(np.expand_dims(obs, axis=0)), training=False).numpy())

                e -= epsilon_interval / 1_000_000
                e = max(e, epsilon_min)

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
            all_episode_rewards.append(episode_reward)  # <-- ADD THIS
            running_reward = np.mean(episode_reward_history) if episode_reward_history else 0.0

            if running_reward > 40:
                tqdm.write(f"Solved at episode {episode_count+1}!")
                break

        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_name = (
            f"model_ep{n_episodes}_gamma{str(gamma).replace('.','')}_"
            f"epsilon{str(epsilon).replace('.','')}_"
            f"lr{str(learning_rate).replace('.','')}.keras"
        )
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
        print(f"Saved model to {model_path}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(all_episode_rewards, label="Episode Reward")
        if len(all_episode_rewards) > 100:
            running_avg = np.convolve(all_episode_rewards, np.ones(100) / 100, mode='valid')
            ax.plot(range(99, len(all_episode_rewards)), running_avg, label="100-episode running avg", color='orange')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"Learning Curve: gamma={gamma}, lr={learning_rate}, batch={batch_size}")
        ax.legend()
        plot_name = model_name.rsplit('.', 1)[0] + ".png"
        plot_path = os.path.join(model_dir, plot_name)
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved learning curve plot to {plot_path}")
        env.close()

if __name__ == "__main__":
    main()