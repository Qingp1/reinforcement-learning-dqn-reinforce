import os
import numpy as np
import tensorflow as tf
from typing import Tuple

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepQModel(tf.keras.Model):

    def __init__(self, state_size: int, num_actions: int, lr: float = 0.001) -> None:
        super(DeepQModel, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_actions)
        ])

        self.model.build(input_shape=(None, state_size))

        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.build(input_shape=(None, state_size))
        self.target_model.set_weights(self.model.get_weights())

    def call(self, states: tf.Tensor) -> tf.Tensor:
        return self.model(states)
    

    def loss_func(self, batch: Tuple[tf.Tensor, np.ndarray, tf.Tensor, tf.Tensor, np.ndarray], discount_factor: float = 0.99) -> tf.Tensor:
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = batch

        batch_states = tf.convert_to_tensor(batch_states, dtype=tf.float32)
        batch_next_states = tf.convert_to_tensor(batch_next_states, dtype=tf.float32)
        batch_rewards = tf.convert_to_tensor(batch_rewards, dtype=tf.float32)
        batch_actions = tf.convert_to_tensor(batch_actions, dtype=tf.int32)
        batch_done = tf.convert_to_tensor(batch_done, dtype=tf.float32)

        q_values = self.model(batch_states)
        q_next = self.target_model(batch_next_states)

        batch_size = tf.shape(q_values)[0]
        indices = tf.stack([tf.range(batch_size, dtype=tf.int32), batch_actions], axis=1)
        q_sa = tf.gather_nd(q_values, indices)

        max_q_next = tf.reduce_max(q_next, axis=1)
        targets = batch_rewards + (1.0 - batch_done) * discount_factor * max_q_next

        loss = tf.reduce_mean(tf.square(q_sa - targets))
        return loss
