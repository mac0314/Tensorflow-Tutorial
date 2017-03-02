# This Python file uses the following encoding: utf-8

# https://youtu.be/ls8jHqRnEQk?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 10 - 딥러닝으로 MNIST 98%이상 해보기


import tensorflow as tf
import random as rand
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
learning_rate = 0.001
training_epoch = 15
batch_size = 100
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Create model

# Set model weight
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

# Construct model
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))  # Hidden layer with RELU activation
hypothesis = tf.add(tf.matmul(L2, W3), B3)  # No need to use softmax here

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=Y, logits=hypothesis, dim=-1))   # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

init = tf.initialize_all_variables()

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

