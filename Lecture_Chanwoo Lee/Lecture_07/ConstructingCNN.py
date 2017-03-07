# This Python file uses the following encoding: utf-8

# https://youtu.be/mLdqxGB2Pc4?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 7강. Constructing CNN 1

import tensorflow as tf
import os
from PIL import Image
import numpy as np

image_dir = os.getcwd() + "/profile_images/0004_4.jpg"
label_dir = os.getcwd() + "/profile_images/label.csv"

imagename_list = [image_dir]
labelname_list = [label_dir]

imagename_queue = tf.train.string_input_producer(imagename_list)
labelname_queue = tf.train.string_input_producer(labelname_list)

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read(imagename_queue)
label_key, label_value = label_reader.read(labelname_queue)

image = tf.image.decode_jpeg(image_value)
label = tf.decode_csv(label_value, record_defaults=[[0]])

x = tf.cast(image, tf.float32)
y_ = tf.cast(label, tf.float32)
y_ = tf.reshape(y_, [-1, 1])

image_width = 2048
image_height = 1536

# x = tf.placeholder(tf.float32, shape=[None, image_width, image_height])
# y_ = tf.placeholder(tf.float32, shape=[None, 1])

# hidden1 convolution
# RGB이기 때문에 3으로 줌
# 이미지가 32개의 부분으로 나뉘어진다.
W_hidden1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32]))
b_hidden1 = tf.Variable(tf.zeros([32]))

x_image = tf.reshape(x, [-1, image_width, image_height, 3])


conv1 = tf.nn.conv2d(x_image, W_hidden1, strides=[1, 1, 1, 1], padding="SAME")
hidden1 = tf.nn.relu(conv1 + b_hidden1)

# hidden2 convolution
W_hidden2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
b_hidden2 = tf.Variable(tf.zeros([64]))

conv2 = tf.nn.conv2d(hidden1, W_hidden2, strides=[1, 1, 1, 1], padding="SAME")
hidden2 = tf.nn.relu(conv2 + b_hidden2)

# Densely Connected Layer
h_flat = tf.reshape(hidden2, [-1, image_width * image_height * 64])

W_fc = tf.Variable(tf.truncated_normal([image_width * image_height * 64, 10]))
b_fc = tf.Variable(tf.zeros([10]))

h_fc = tf.nn.relu(tf.matmul(h_flat, W_fc) + b_fc)

# Readout Layer
W_out = tf.Variable(tf.truncated_normal([10, 1]))
b_out = tf.Variable(tf.zeros([1]))

prediction = tf.matmul(h_fc, W_out) + b_out

print prediction
print y_

# Back propagation
# loss
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=prediction, logits=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train_step)
        cost, acc = sess.run([cross_entropy, accuracy])
        print "--------------------------"
        print "loss : ", cost
        print "accuracy : ", acc

    image = sess.run(image)

    # Image.fromarray(image).show()

    coord.request_stop()
    coord.join(thread)


