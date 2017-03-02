# This Python file uses the following encoding: utf-8

# https://youtu.be/9i7FBbcZPMA?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 09-1 - XOR을 위한 텐스플로우 딥넷트웍

import tensorflow as tf
import numpy as np


# Load data in text file
# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True)
# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Deep => more W, b
# Wide => higher second parameter in array
W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

# Our hypothesis
# Logistic regression
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Minimize
a = tf.Variable(0.01)   # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Fit the line.
    for step in xrange(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)


    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy: ", accuracy.eval({X:x_data, Y:y_data})

