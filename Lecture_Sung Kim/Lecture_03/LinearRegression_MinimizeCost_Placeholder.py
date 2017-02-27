# This Python file uses the following encoding: utf-8

# https://youtu.be/pHPmzTQ_e2o?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 03 - Linear Regression 의 cost 최소화의 TensorFlow 구현

import tensorflow as tf


x_data = [1., 2., 3.]
y_data = [1., 2., 3.]


# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
descent = W - tf.multiply(0.1, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))
update = W.assign(descent)


# Before starting, initialize the variables. We will 'run' this first
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(100):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)


# Learns best fit is W : [1]


sess.close()

