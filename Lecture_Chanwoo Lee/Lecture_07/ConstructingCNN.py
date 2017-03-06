# This Python file uses the following encoding: utf-8

# https://youtu.be/mLdqxGB2Pc4?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 7강. Constructing CNN 1

import tensorflow as tf
import os
from PIL import Image
import numpy as np

image_dir = os.getcwd() + "/profile_images/0004_4.jpg"

print image_dir

filename_list = [image_dir]

filename_queue = tf.train.string_input_producer(filename_list)

reader = tf.WholeFileReader()

key, value = reader.read(filename_queue)

image_decoded = tf.image.decode_jpeg(value)

#print key, value


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    image = sess.run(image_decoded)

    Image.fromarray(image).show()

    coord.request_stop()
    coord.join(thread)

