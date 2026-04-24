import os
import numpy as np
import tensorflow as tf
from typing import List

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Reinforce(tf.keras.Model):
    def __init__(self, state_size: int, num_actions: int, lr: float = 0.001) -> None:
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        :param lr: learning rate for the optimizer
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_actions, activation="softmax")
        ])
        

    def call(self, states: tf.Tensor) -> tf.Tensor:
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        return self.model(states)

    def loss_func(self, states: tf.Tensor, actions: List[int], discounted_rewards: List[float]) -> tf.Tensor:
        """
        Computes the REINFORCE policy gradient loss.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        probs = self(states)

        episode_length = tf.shape(probs)[0]
        indices = tf.stack([tf.range(episode_length, dtype=tf.int32), actions], axis=1)
        action_probs = tf.gather_nd(probs, indices)

        log_probs = tf.math.log(action_probs + 1e-8)
        loss = -tf.reduce_sum(log_probs * discounted_rewards)
        return loss
