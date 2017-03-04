# This Python file uses the following encoding: utf-8

# https://youtu.be/A8wJYfDUYCk?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# ML lab 12 - TensorFlow에서 RNN 구현하기

# previous version - not working

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

import numpy as np


char_rdic = ['h','e','l','o']  # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)}  # char -> id

sample = [char_dic[c] for c in "hello"] # to index

x_data = np.array([ [1, 0, 0, 0],  # h
                    [0, 1, 0, 0],  # e
                    [0, 0, 1, 0],  # l
                    [0, 0, 1, 0]],  # l
                  dtype='f')

# x_data = tf.one_hot(sample[:-1], len(char_dic), 1.0, 0.0, -1)


# Configuration
char_vocab_size = len(char_dic)
rnn_size = 4 #char_vocab_size  # 1 hot coding (one of 4)
time_step_size = 4  # 'hell' -> predict 'ello'
batch_size = 1      # one sample

# RNN model
rnn_cell = rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = rnn.rnn(rnn_cell, X_split, state)
#outputs, state = tf.nn.seq2seq.rnn_decoder ( X_split, state, rnn_cell)

print (state)
print (outputs)

# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
# targets: list of 1D batch-sized int32 Tensors of the same length as logits.
# weights: list of 1D batch-sized float-Tensors of the same length as logits.
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])


loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)


# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print result, [char_rdic[t] for t in result]

