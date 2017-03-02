# This Python file uses the following encoding: utf-8

# https://youtu.be/1vCOoBwYQVU?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 07 - 학습 rate, training/test 셋으로 성능평가


import tensorflow as tf
import random as rand
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Create model

# Set model weight
W = tf.Variable(tf.zeros([784, 10]))    # mnist data image of shape 28*28=784
b = tf.Variable(tf.zeros([10]))         # 0-9 digits recognition => 10 classes

# Construct model
hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)   # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

init = tf.initialize_all_variables()

training_epoch = 25
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (sess.run(b))


    print ("Optimization Finished!")


    # Predict & Show
    # Get one and predict
    r = rand.randint(0, mnist.test.num_examples-1)
    print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    print "Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]})

    # Show the img
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()



    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

