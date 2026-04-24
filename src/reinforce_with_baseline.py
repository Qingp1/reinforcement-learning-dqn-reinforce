import os
import numpy as np
import tensorflow as tf
from typing import List

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size: int, num_actions: int, lr: float = 0.001) -> None:
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        :param lr: learning rate for the optimizer
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_actions, activation="softmax")
        ])

        self.critic = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation=None)
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
        # TODO: implement this!
        return self.actor(states)
    
    def value_function(self, states: tf.Tensor) -> tf.Tensor:
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        values = self.critic(states)
        return tf.squeeze(values, axis=-1)
    
    def loss_func(self, states: tf.Tensor, actions: List[int], discounted_rewards: List[float]) -> tf.Tensor:
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 2, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        probs = self.call(states)
        values = self.value_function(states)

        T = tf.shape(states)[0]
        indices = tf.stack([tf.range(T, dtype=tf.int32), actions], axis=1)
        action_probs = tf.gather_nd(probs, indices)
        log_probs = tf.math.log(action_probs + 1e-8)

        advantage = discounted_rewards - values

        actor_loss = -tf.reduce_mean(tf.stop_gradient(advantage) * log_probs)
        critic_loss = tf.reduce_mean(tf.square(advantage))

        loss = actor_loss + critic_loss
        return loss
