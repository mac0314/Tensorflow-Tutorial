# This Python file uses the following encoding: utf-8

# https://youtu.be/eDKxY5Z5dVQ?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 09-2 - Tensor Board로 딥네트웍 들여다보기


import tensorflow as tf
import numpy as np


# Load data in text file
# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True)
# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

# Name variables
X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

# Deep => more W, b
# Wide => higher second parameter in array
W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name="Weight1")
W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0), name="Weight2")
W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name="Weight3")

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

# Our hypothesis
# Logistic regression
# Add scope for better graph hierarchy
with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope("layer3") as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
with tf.name_scope("layer4") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    # Add scalar variables
    cost_summ = tf.summary.scalar("cost", cost)

# Minimize
with tf.name_scope("train") as scope:
    a = tf.Variable(0.01)   # Learning rate, alpha
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# Add histogram
w1_hist = tf.summary.histogram("weights1", W1)
w2_hist = tf.summary.histogram("weights2", W2)
w3_hist = tf.summary.histogram("weights3", W3)

b1_hist = tf.summary.histogram("biases1", b1)
b2_hist = tf.summary.histogram("biases2", b2)
b3_hist = tf.summary.histogram("biases3", b3)

y_hist = tf.summary.histogram("y", Y)

# Before starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Launch Tensorboard in terminal
    # tensorboard --logdir=./logs/xor_logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

    # Fit the line.
    for step in xrange(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            summary, _ = sess.run([merged, train], feed_dict={X:x_data, Y:y_data})
            writer.add_summary(summary, step)
            #print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)


    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy: ", accuracy.eval({X:x_data, Y:y_data})



