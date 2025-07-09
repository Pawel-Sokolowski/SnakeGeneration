import itertools
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
import snake_gym_env
from tqdm import trange, tqdm
from collections import deque
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, running on CPU.")

# 🧠 NoisyDense layer definition
class NoisyDense(layers.Layer):
    def __init__(self, units, activation=None, sigma=0.5):
        super().__init__()
        self.units = units
        self.activation = layers.Activation(activation) if activation else None
        self.sigma = sigma

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.mu_weight = self.add_weight(name="mu_weight", shape=(in_dim, self.units),
                                         initializer="random_uniform", trainable=True)
        self.sigma_weight = self.add_weight(name="sigma_weight", shape=(in_dim, self.units),
                                            initializer=tf.keras.initializers.Constant(0.017),
                                            trainable=True)
        self.mu_bias = self.add_weight(name="mu_bias", shape=(self.units,),
                                       initializer="random_uniform", trainable=True)
        self.sigma_bias = self.add_weight(name="sigma_bias", shape=(self.units,),
                                          initializer=tf.keras.initializers.Constant(0.017),
                                          trainable=True)
    def call(self, inputs):
        eps_in = tf.random.normal([inputs.shape[-1], 1])
        eps_out = tf.random.normal([1, self.units])
        noise = tf.multiply(eps_in, eps_out)
        weight = self.mu_weight + self.sigma * self.sigma_weight * noise
        bias = self.mu_bias + self.sigma * self.sigma_bias * tf.squeeze(eps_out)
        output = tf.matmul(inputs, weight) + bias
        return self.activation(output) if self.activation else output

# 🧠 Q-network using NoisyDense
def create_noisy_conv1d_q_model(input_shape, num_actions):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = NoisyDense(128, activation='relu')(x)
    x = NoisyDense(64, activation='relu')(x)
    q_values = NoisyDense(num_actions)(x)
    return keras.Model(inputs=inputs, outputs=q_values)

def param_grid_search(param_grid):
    keys, values = zip(*param_grid.items())
    for bundle in itertools.product(*values):
        yield dict(zip(keys, bundle))

def train_model(model_dir="saved_models"):
    os.makedirs(model_dir, exist_ok=True)

    param_grid = {
        "num_episodes": [100_000],
        "gamma": [0.99],
        "batch_size": [64],
        "learning_rate": [0.001],
    }

    for params in param_grid_search(param_grid):
        print("Starting training with parameters:", params)
        train_with_params(params, model_dir)

def train_with_params(params, model_dir):
    num_episodes = params["num_episodes"]
    max_steps_per_episode = 500
    gamma = params["gamma"]
    batch_size = params["batch_size"]
    max_memory_length = 100_000
    update_after_actions = 4
    update_target_network = 1000
    learning_rate = params["learning_rate"]

    env = gym.make("Snake-v0", render_mode=None)
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape
    obs_dtype = env.observation_space.dtype

    model = create_noisy_conv1d_q_model(input_shape, num_actions)
    model_target = create_noisy_conv1d_q_model(input_shape, num_actions)
    model_target.set_weights(model.get_weights())
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    state_history = np.zeros((max_memory_length,) + input_shape, dtype=obs_dtype)
    state_next_history = np.zeros((max_memory_length,) + input_shape, dtype=obs_dtype)
    action_history = np.zeros(max_memory_length, dtype=np.int32)
    rewards_history = np.zeros(max_memory_length, dtype=np.float32)
    done_history = np.zeros(max_memory_length, dtype=np.float32)

    episode_reward_history = deque(maxlen=100)
    all_episode_rewards = []
    step_count = 0
    memory_index = 0

    progress = trange(num_episodes, desc=f"gamma={gamma},lr={learning_rate}", dynamic_ncols=True)
    save_interval = num_episodes // 10 or 1

    for episode_count in progress:
        obs, _ = env.reset(seed=None)
        obs = np.array(obs, dtype=np.float32)
        episode_reward = 0

        for t in range(max_steps_per_episode):
            step_count += 1
            action = tf.argmax(model(tf.expand_dims(obs, axis=0), training=True)[0]).numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated

            state_history[memory_index] = obs
            state_next_history[memory_index] = next_obs
            action_history[memory_index] = action
            rewards_history[memory_index] = reward
            done_history[memory_index] = done

            episode_reward += reward
            obs = next_obs
            memory_index = (memory_index + 1) % max_memory_length

            if step_count % update_after_actions == 0 and memory_index > batch_size:
                indices = np.random.choice(min(memory_index, max_memory_length), size=batch_size, replace=False)
                state_sample = tf.convert_to_tensor(state_history[indices], dtype=tf.float32)
                state_next_sample = tf.convert_to_tensor(state_next_history[indices], dtype=tf.float32)
                rewards_sample = tf.convert_to_tensor(rewards_history[indices], dtype=tf.float32)
                action_sample = tf.convert_to_tensor(action_history[indices], dtype=tf.int32)
                done_sample = tf.convert_to_tensor(done_history[indices], dtype=tf.float32)

                future_rewards = model_target(state_next_sample, training=False)
                max_future_q = tf.reduce_max(future_rewards, axis=1)
                updated_q_values = rewards_sample + gamma * max_future_q
                updated_q_values = updated_q_values * (1.0 - done_sample) - done_sample

                masks = tf.one_hot(action_sample, num_actions)
                with tf.GradientTape() as tape:
                    q_values = model(state_sample, training=True)
                    q_action = tf.reduce_sum(q_values * masks, axis=1)
                    loss = loss_function(updated_q_values, q_action)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                running_reward = np.mean(episode_reward_history) if episode_reward_history else 0.0
                tqdm.write(f"Running reward: {running_reward:.2f} at episode {episode_count}, step {step_count}")

            if done:
                break

        episode_reward_history.append(episode_reward)
        all_episode_rewards.append(episode_reward)
        running_reward = np.mean(episode_reward_history) if episode_reward_history else 0.0

        if running_reward > 200:
            tqdm.write(f"Solved at episode {episode_count+1}!")
            break

        if (episode_count + 1) % save_interval == 0:
            model_name = (
                f"model_ep{num_episodes}_gamma{str(gamma).replace('.','')}_"
                f"lr{str(learning_rate).replace('.','')}_ep{episode_count+1}.keras"
            )
            model.save(os.path.join(model_dir, model_name))

    model.save(os.path.join(model_dir, f"model_final.keras"))
    plt.figure(figsize=(10, 6))
    plt.plot(all_episode_rewards, label="Episode Reward")
    if len(all_episode_rewards) > 100:
        avg = np.convolve(all_episode_rewards, np.ones(100) / 100, mode='valid')
        plt.plot(range(99, len(all_episode_rewards)), avg, label="100-episode running avg", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Learning Curve: gamma={gamma}, lr={learning_rate}, batch={batch_size}")
    
if __name__ == "__main__":
    train_model()