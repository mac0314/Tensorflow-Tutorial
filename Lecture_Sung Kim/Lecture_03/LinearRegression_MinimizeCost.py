# This Python file uses the following encoding: utf-8

# https://youtu.be/pHPmzTQ_e2o?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 03 - Linear Regression 의 cost 최소화의 TensorFlow 구현

import tensorflow as tf
import matplotlib.pyplot as plt

# tf Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

# Set model weights
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(X, W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2)) / (m)

# Initializing the variables
init = tf.global_variables_initializer()

# For graphs
W_val = []
cost_val = []

# Launch the graph
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    print i*0.1, sess.run(cost, feed_dict={W: i*0.1})
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))


# Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()

