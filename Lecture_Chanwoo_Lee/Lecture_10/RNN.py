# This Python file uses the following encoding: utf-8

# https://youtu.be/qPxtsCldUMg?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 10강. RNN part. 1

# sine tracker

import tensorflow as tf
import matplotlib.pyplot as plt

# C에서의 define과 비슷한 역할 - 메모리를 차지하지 않고 치환해준다.
CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("samples", 1000, "simulation data samples")
CONSTANT.DEFINE_integer("hidden", 5, "hidden layers in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size into rnn")
CONSTANT.DEFINE_integer("batch_size", 10, "minibatch size for training")
CONSTANT.DEFINE_integer("state_size", 15, "state size in rnn")
CONSTANT.DEFINE_integer("recurrent", 5, "recurrent step")
CONSTANT.DEFINE_float("learning_rate", 0.01, 'learning rate for optimizer')

CONST = CONSTANT.FLAGS


class RNN(object):
    """
        RNN class
    """
    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._build_model()
        self._build_train()
        self._initialize()
        #self._pack_test()

    def run(self):
        """
            run function
        :return:
        """
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(1000):
            _, loss = self.sess.run([self.train, self.loss])
            if i % 20 == 0:
                print(i, "loss: ", loss)
        print("loss: ", loss)
        # print(self._run_session())
        self._close_session()

    @classmethod    # protected, cls를 써준다.
    def _run_session(cls, run_graph):
        output = cls.sess.run(run_graph)
        return output

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()
        cls.coord = tf.train.Coordinator()
        cls.thread = tf.train.start_queue_runners(cls.sess, cls.coord)

    @classmethod
    def _close_session(cls):
        cls.coord.request_stop()
        cls.coord.join(cls.thread)
        cls.sess.close()

    @classmethod
    def _gen_sim_data(cls):
        # python 3.5 이상에서는 range가 xrange로 작동함. 리스트가 아니다. 따라서 [i for i in range(CONST.samples + 1)] 을 통해 만들어 주어야 한다.
        #cls.ts_x = tf.constant(range(CONST.samples + 1), dtype=tf.float32) # 이전 버전
        cls.ts_x = tf.constant([i for i in range(CONST.samples + 1)], dtype=tf.float32)
        ts_y = tf.sin(cls.ts_x * 0.1)   # 조밀하게 샘플링한다.

        sp_batch = (int(CONST.samples/CONST.hidden), CONST.hidden, CONST.vec_size)
        cls.batch_input = tf.reshape(ts_y[:-1], sp_batch)
        cls.batch_label = tf.reshape(ts_y[1:], sp_batch)

    @classmethod
    def _build_batch(cls):
        batch_set = [cls.batch_input, cls.batch_label]
        cls.b_train, cls.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    @classmethod
    def _build_model(cls):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(CONST.state_size)
        # tf 1.0
        # tf.pack -> tf.stack,
        # tf.unpack -> tf.unstack
        output, _ = tf.contrib.rnn.static_rnn(rnn_cell, tf.unstack(cls.b_train, axis=1), dtype=tf.float32)

        cls.output_w = tf.Variable(tf.truncated_normal([CONST.hidden, CONST.state_size, CONST.vec_size]))
        output_b = tf.Variable(tf.zeros([CONST.vec_size]))

        cls.pred = tf.matmul(output, cls.output_w) + output_b

    @classmethod
    def _build_train(cls):
        cls.loss = 0
        for i in range(CONST.hidden):
            cls.loss += tf.losses.mean_squared_error(tf.unstack(cls.b_label, axis=1)[i], cls.pred[i])
        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    @classmethod
    def _pack_test(cls):
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.constant([7, 8, 9])
        d = tf.stack([a, b, c])
        d_ = [a, b, c]

        print(cls.sess.run(a))
        print(cls.sess.run(b))
        print(cls.sess.run(c))
        print(cls.sess.run(d))
        print(cls.sess.run(tf.shape(d)))
        print(cls.sess.run(d_))

# Tensorflow
def main(_):
    """
        Main function
    :param _:
    :return:
    """
    rnn = RNN()
    rnn.run()

    #plt.plot([1, 2, 3, 4])
    #plt.show()

# Python program counter
if __name__ == "__main__":
    tf.app.run()


