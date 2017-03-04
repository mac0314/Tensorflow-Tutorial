# This Python file uses the following encoding: utf-8

# https://youtu.be/hPkmxczEj6k?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 6강. File load in Tensorflow and Queue Thread

import tensorflow as tf

# filename_queue = tf.train.string_input_producer(['test0.csv', 'test1.csv'])
filename_queue = tf.train.string_input_producer(['test%d.csv' % i for i in range(2)])

record_size = 6

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5, col6, col7, col8, col9 = tf.decode_csv(
    value, record_defaults=record_defaults
)

features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(record_size):
        # Retrieve a single instance:
        example, label = sess.run([features, col5])
        print 'example : ', example, 'label : ', label

    coord.request_stop()
    coord.join(threads)
