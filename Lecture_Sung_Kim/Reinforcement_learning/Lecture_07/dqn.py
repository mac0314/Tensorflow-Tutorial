# This Python file uses the following encoding: utf-8

# https://youtu.be/Fbf9YUyDFww?list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
# RL lab 07 - 1 - DQN 1 (NIPS 2013)

import tensorflow as tf
import numpy as np


# Network - Go deep
class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h1_size=10, h2_size=15, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name="input_x")

            # First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, h1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            layer1 = tf.nn.relu(tf.matmul(self._X, W1))

            # Second layer of weights
            W2 = tf.get_variable("W2", shape=[h1_size, h2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            layer2 = tf.nn.relu(tf.matmul(layer1, W2))

            # Third layer of weights
            W3 = tf.get_variable("W3", shape=[h2_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            # Q prediction
            # linear regression이기 때문에 activation function을 하지 않아도 된다.
            self._Q_pred = tf.matmul(layer2, W3)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Q_pred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Q_pred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
