# This Python file uses the following encoding: utf-8

# https://youtu.be/4HrSxpi3IAM?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 02 - Tensorflow로 간단한 linear regression을 구현

import tensorflow as tf


x_data = [1, 2, 3]
y_data = [1, 2, 3]


# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


# Our hypothesis
hypothesis = W * x_data + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables. We will 'run' this first
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W), sess.run(b)


sess.close()

# Learns best fit is W : [1], b : [0]

