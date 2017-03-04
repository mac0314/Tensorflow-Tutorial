# This Python file uses the following encoding: utf-8

# https://youtu.be/6KlkiKyjEu0?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 11 - ConvNet을 TensorFlow로 구현하자 (MNIST 99%)


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# MNIST variables
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

# Parameters
learning_rate = 0.001
training_epoch = 100
batch_size = 128
test_size = 256
display_step = 10

# tf Graph Input - X, Y
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X-input') # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 10], name='Y-input')  # 0-9 digits recognition => 10 classes


# Set model weight
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))      # 3*3*1 conv, 32 outputs
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))     # 3*3*32 conv, 64 outputs
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))    # 3*3*32 conv, 128 outputs
W4 = tf.Variable(tf.random_normal([2048, 625], stddev=0.01))        # FC 128*4*4 inputs, 625 outputs
W_o = tf.Variable(tf.random_normal([625, 10], stddev=0.01))          # FC 625 inputs, 10 outputs(labels)

tf.summary.histogram('Weight1', W1)
tf.summary.histogram('Weight2', W2)
tf.summary.histogram('Weight3', W3)
tf.summary.histogram('Weight4', W4)
tf.summary.histogram('Weight5', W_o)

# Construct model
dropout_cnn_rate = tf.placeholder(tf.float32)
dropout_fcc_rate = tf.placeholder(tf.float32)

X_image = tf.reshape(X, [-1, 28, 28, 1], name='X-input-reshape')

with tf.name_scope('Layer1'):
    l1a = tf.nn.relu(tf.nn.conv2d(X_image, W1, strides=[1, 1, 1, 1], padding='SAME'))
    print l1a   # l1a shape = (?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print l1    # l1 shape = (?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, dropout_cnn_rate)

with tf.name_scope('Layer2'):
    l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME'))
    print l2a   # l2a shape = (?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print l2    # l2 shape = (?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, dropout_cnn_rate)

with tf.name_scope('Layer3'):
    l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    print l3a   # l3a shape = (?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print l3    # l3 shape = (?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])
    print l3
    l3 = tf.nn.dropout(l3, dropout_cnn_rate)

with tf.name_scope('Layer4'):
    l4 = tf.nn.relu(tf.matmul(l3, W4))
    l4 = tf.nn.dropout(l4, dropout_fcc_rate)

with tf.name_scope('Layer5'):
    hypothesis = tf.matmul(l4, W_o)


# Minimize error using cross entropy
with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=Y, logits=hypothesis, dim=-1))   # Softmax loss
    tf.summary.scalar('Cost', cost)

# Gradient Descent
with tf.name_scope('Train'):
    train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost) # RMSProp Optimizer

# Accuracy
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

init = tf.global_variables_initializer()


# Launch the graph in a session
with tf.Session() as sess:
    sess.run(init)

    # Launch Tensorboard in terminal
    # tensorboard --logdir=./logs/mnist/cnn
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/mnist/cnn", sess.graph)

    # Training cycle
    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))

        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                           dropout_cnn_rate: 0.8, dropout_fcc_rate: 0.5})

            print 'Cost:', \
                sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          dropout_cnn_rate: 0.8, dropout_fcc_rate: 0.5})

        # Fit training using batch data
        _, summary = sess.run([train_op, merged], feed_dict={X: trX[start:end], Y: trY[start:end], dropout_cnn_rate: 0.7,
                                                              dropout_fcc_rate: 0.5})
        # Summarize
        writer.add_summary(summary, i)

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print 'Accuracy:', i, sess.run(accuracy, feed_dict={X: teX[test_indices],
                                                        Y: teY[test_indices],
                                                        dropout_cnn_rate: 1.0,
                                                        dropout_fcc_rate: 1.0})

    writer.close()

