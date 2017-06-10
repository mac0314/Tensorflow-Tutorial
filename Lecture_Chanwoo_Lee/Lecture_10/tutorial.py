# This Python file uses the following encoding: utf-8

# https://youtu.be/qPxtsCldUMg?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 10강. RNN part. 1

import tensorflow as tf

# C에서의 define과 비슷한 역할 - 메모리를 차지하지 않고 치환해준다.
CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("test_int", 10, "Test integer")
CONST = CONSTANT.FLAGS


# Tensorflow
def main(_):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        '''
        # tf.Print는 다음과 같은 방식으로 사용
        # 첫번째 파라미터는 리스트여야 한다.
        # tf.Print([graph_end_point1, graph_end_point2], what_you_print, "message you want")

        a = tf.model()
        b = tf.model(a)
        c = tf.model(b)
        d = tf.model(c)
        e = tf.model(c)

        # sess.run(d)를 실행할 경우
            a = tf.model()
            b = tf.model(a)
            c = tf.model(b)
            d = tf.model(c)
        #   가 실행 된다.

        # sess.run(e)를 실행할 경우
            a = tf.model()
            b = tf.model(a)
            c = tf.model(b)
            e = tf.model(c)
        #   가 실행 된다.

        tf.Print([d, e], c, "message you want")

        # 이렇게 실행할 경우 도중에 c가 계산이 되지 않기 때문에 에러가 뜬다.
        tf.Print([b], c, "message you want")
        '''

    print "test_int: ", CONST.test_int


# Python program counter
if __name__ == "__main__":
    tf.app.run()


