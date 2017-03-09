# This Python file uses the following encoding: utf-8

# https://youtu.be/o8W0BkTCSR8?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 9강. Constructing CNN - final

import tensorflow as tf
import os


image_dir = os.getcwd() + "/Temp_data_Set/Test_Dataset_png/"
image_list = os.listdir(image_dir)
image_list.sort()

label_dir = os.getcwd() + "/Temp_data_Set/Test_Dataset_csv/Label.csv"

# Check file_list
# print file_list[:20]
# print len(file_list)

# 단순 for문일 경우 xrange를 쓰는 것이 속도면에서 유리하다.
for i in xrange(len(image_list)):
    image_list[i] = image_dir + image_list[i]

label_list = [label_dir]

imagename_queue = tf.train.string_input_producer(image_list)
labelname_queue = tf.train.string_input_producer(label_list)

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read(imagename_queue)
label_key, label_value = label_reader.read(labelname_queue)

image_width = 49
image_height = 61

image = tf.image.decode_png(image_value)
image = tf.reshape(image, [image_width, image_height, 1])
label = tf.decode_csv(label_value, record_defaults=[[0]])


image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=64, num_threads=4,
                                                  capacity=50000, min_after_dequeue=10000)

x = tf.cast(image_batch, tf.float32)
y_ = tf.cast(label_batch, tf.float32)
y_ = tf.reshape(y_, [-1, 1])


# Parameters
W_hidden1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
b_hidden1 = tf.Variable(tf.zeros([32])) # 이미지가 32개의 부분으로 나뉘어진다.

W_hidden2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
b_hidden2 = tf.Variable(tf.zeros([64]))

W_fc = tf.Variable(tf.truncated_normal([image_width * image_height * 64, 10]))
b_fc = tf.Variable(tf.zeros([10]))

W_out = tf.Variable(tf.truncated_normal([10, 1]))
b_out = tf.Variable(tf.zeros([1]))


# CNN Model
x_image = tf.reshape(x, [-1, image_width, image_height, 1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_hidden1, strides=[1, 1, 1, 1], padding="SAME") + b_hidden1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_hidden2, strides=[1, 1, 1, 1], padding="SAME") + b_hidden2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")

# Densely Connected Layer
h_flat = tf.reshape(h_pool2, [-1, image_width * image_height * 64])

h_fc = tf.nn.relu(tf.matmul(h_flat, W_fc) + b_fc)
dropout_rate = 0.7
drop_fc = tf.nn.dropout(h_fc, dropout_rate)

prediction = tf.matmul(drop_fc, W_out) + b_out


# Back propagation
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=prediction, logits=y_))   # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # to manage thread
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train_step)
        cost, acc = sess.run([cross_entropy, accuracy])
        print "--------------------------"
        print "loss : ", cost
        print "accuracy : ", acc

        # Check data
        # print sess.run(image_key)
        # print sess.run(tf.shape(x))

    image = sess.run(image)

    coord.request_stop()
    coord.join(thread)


