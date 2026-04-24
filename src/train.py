import os
import sys
import argparse
from typing import List, Tuple, Optional, Union
import gymnasium as gym
import numpy as np
import tensorflow as tf
from src.reinforce import Reinforce
from src.reinforce_with_baseline import ReinforceWithBaseline
from src.deep_q import DeepQModel


def discount(rewards: List[float], discount_factor: float = .99) -> List[float]:
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep, which
    are calculated by summing the rewards for each future timestep, discounted
    by how far in the future it is.
    For example, in the simple case where the episode rewards are [1, 3, 5] 
    and discount_factor = .99 we would calculate:
    dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
    dr_2 = 3 + 0.99 * 5 = 7.95
    dr_3 = 5
    and thus return [8.8705, 7.95 , 5].
    Refer to the slides for more details about how/why this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    discounted = []
    running = 0.0
    for r in reversed(rewards):
        running = r + discount_factor * running
        discounted.append(running)
    discounted.reverse()
    return discounted

def generate_trajectory(env: gym.Env, model: tf.keras.Model) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()[0]
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        states.append(state)

        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        probs = model(state_tensor)[0].numpy()

        action = np.random.choice(len(probs), p=probs)
        actions.append(action)

        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result

        rewards.append(reward)
        state = next_state
    
    return states, actions, rewards


def train_reinforce_episode(env: gym.Env, model: tf.keras.Model) -> Tuple[float, tf.Tensor]:
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode, and the loss
    """

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    states, actions, rewards = generate_trajectory(env, model)
    total_reward = float(sum(rewards))

    discounted_rewards = discount(rewards)
    states_tensor = tf.convert_to_tensor(np.array(states, dtype=np.float32))
    actions_array = np.array(actions, dtype=np.int32)
    discounted_array = np.array(discounted_rewards, dtype=np.float32)

    with tf.GradientTape() as tape:
        loss = model.loss_func(states_tensor, actions_array, discounted_array)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_reward, loss

def train_deep_q_episode(env: gym.Env, model: tf.keras.Model, batch_size: int, memory: List, epsilon: float = .1) -> Tuple[float, List, tf.Tensor]:
    """
    This function should train your model for one episode.
    Each call to this function should play an episode using the epsilon-greedy algorithm.
    After each episode, the model should be trained on a batch of experiences sampled from the memory.
    Make sure to return the total reward for the episode, memory, and the loss.

    :param env: The openai gym environment
    :param model: The model
    :param batch_size: The number of experiences to sample from the memory
    :param memory: The memory to sample from (you will mutate this in place)
    :param epsilon: The epsilon value for epsilon-greedy
    :returns: The total reward for the episode, memory, and the loss

    NOTE: Make sure you convert the batch elements to the correct types (np.ndarray, tf.Tensor, etc.)
    """
    state = env.reset()[0]
    done = False
    total_reward = 0.0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            q_values = model(state_tensor)[0]
            action = int(tf.argmax(q_values).numpy())

        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done_flag = terminated or truncated
        else:
            next_state, reward, done_flag, _ = step_result

        memory.append((state, action, reward, next_state, done_flag))
        total_reward += reward
        state = next_state
        done = done_flag

    if len(memory) > 1000:
        memory = memory[-1000:]

    last_loss = tf.constant(0.0, dtype=tf.float32)
    num_batches = 10

    for _ in range(num_batches):
        if len(memory) < batch_size:
            break

        idx = np.random.choice(len(memory), size=batch_size, replace=False)
        batch = [memory[i] for i in idx]
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = zip(*batch)

        batch_states = tf.convert_to_tensor(np.array(batch_states, dtype=np.float32))
        batch_next_states = tf.convert_to_tensor(np.array(batch_next_states, dtype=np.float32))
        batch_rewards = np.array(batch_rewards, dtype=np.float32)
        batch_actions = np.array(batch_actions, dtype=np.int32)
        batch_done = np.array(batch_done, dtype=np.float32)

        with tf.GradientTape() as tape:
            loss = model.loss_func((batch_states, batch_actions, batch_rewards, batch_next_states, batch_done))
        grads = tape.gradient(loss, model.model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.model.trainable_variables))
        last_loss = loss

    model.target_model.set_weights(model.model.get_weights())
    return total_reward, memory, last_loss



def train(env: gym.Env, model: tf.keras.Model, memory: Optional[List] = None, epsilon: float = .1) -> Union[Tuple[float, tf.Tensor], Tuple[float, List, tf.Tensor]]: 
    """
    This function is now responsible for directing the training process.
    Depending on the type of model, it should call the appropriate training function.
    Think about what needs to happen if the model is a DeepQModel, and what needs to 
    happen if it is a Reinforce model.     

    :param env: The openai gym environment
    :param model: The model
    :param memory: The memory to sample from (you will initalize or mutate this)
    :param epsilon: The epsilon value for epsilon-greedy
    :returns: The total reward for the episode, and the loss
    """
    if isinstance(model, DeepQModel):
        if memory is None:
            memory = []
            while len(memory) < 50:
                state = env.reset()[0]
                done = False
                while not done and len(memory) < 50:
                    action = env.action_space.sample()
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, _ = step_result
                        done_flag = terminated or truncated
                    else:
                        next_state, reward, done_flag, _ = step_result
                    memory.append((state, action, reward, next_state, done_flag))
                    state = next_state
                    done = done_flag

        if len(memory) > 1000:
            memory = memory[-1000:]

        batch_size = 32
        reward, memory, loss = train_deep_q_episode(env, model, batch_size, memory, epsilon)
        return reward, memory, loss

    else:
        reward, loss = train_reinforce_episode(env, model)
        return reward, loss