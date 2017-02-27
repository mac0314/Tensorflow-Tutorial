# This Python file uses the following encoding: utf-8

# https://youtu.be/iEaVR1N8EEk?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 04 - multi-variable linear regression을 TensorFlow에서 구현하기

import tensorflow as tf
import numpy as np



x_data = [[1., 0., 3., 0., 5.],
           [0., 2., 0., 4., 0.]]
y_data = [1, 2, 3, 4, 5]


# Try to find values for W and b that compute y_data = tf.matmul(W, x_data) + b
# (We know that W1, W2 should be 1 and b 0, but Tensorflow will
# figure that out for us.
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


# Our hypothesis
# Matrix multiplication
hypothesis = tf.matmul(W, x_data) + b


'''
# Load data in text file
# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1];

print 'x', x_data
print 'y', y_data

W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


# Our hypothesis
# Matrix multiplication
hypothesis = tf.matmul(W, x_data) + b
'''

'''
# b를 더하는 부분도 행렬에 추가했을 경우
x_data = [[1, 1, 1, 1, 1],
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]
y_data = [1, 2, 3, 4, 5]

# Try to find values for W and b that compute y_data = tf.matmul(W, x_data)
# (We know that W1, W2 should be 1 and b 0, but Tensorflow will
# figure that out for us.
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis
# Matrix multiplication
hypothesis = tf.matmul(W, x_data)
'''


# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W), sess.run(b)
        '''
        # b를 더하는 부분도 행렬에 추가했을 경우
        print step, sess.run(cost), sess.run(W)
        '''


