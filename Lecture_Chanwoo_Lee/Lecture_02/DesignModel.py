# This Python file uses the following encoding: utf-8

# https://youtu.be/eQ-UHjyvEck?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 2강. Model 설계하기

import tensorflow as tf


# 밑에 shape가 None이라 input data를 2차원으로 넣어주어야함
input_data = [[1, 5, 3, 7, 8, 10, 12]]
label_data = [0, 0, 0, 1, 0]


# 사이즈 정의
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5


Learning_Rate = 0.05


# 배치 사이즈를 모를 경우 shape를 None으로 둔다.
x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[CLASSES])


# 정보를 Dictionary 형태로 만들어준다.
tensor_map = {x: input_data, y_: label_data}

'''
실행하는 순서의 코드

# hidden1 weight
W_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32)
# hidden1 bias
B_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32)

hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + B_h1)

# hidden2 weight
W_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
# hidden2 bias
B_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32)

hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + B_h2)


# output weight
W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)
# output bias
B_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32)

y = tf.sigmoid(tf.matmul(hidden2, W_o) + B_o)
'''

'''
실제 코드 작성
'''
# hidden1 weight
W_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32)
# hidden1 bias
B_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32)

# hidden2 weight
W_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
# hidden2 bias
B_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32)

# output weight
W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)
# output bias
B_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32)


hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + B_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + B_h2)
y = tf.sigmoid(tf.matmul(hidden2, W_o) + B_o)


# cost를 raw하게 계산
'''
cost 출력 테스트

cost = -y_ * tf.log(y) - (1 - y_) * tf.log(1 - y)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

# 배치의 숫자대로 코스트를 계산해준다.
print sess.run(cost, feed_dict=tensor_map)
'''

# cost를 하나로 만들어줌, 평균
cost = tf.reduce_mean(-y_ * tf.log(y) - (1 - y_) * tf.log(1 - y))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)


sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# cost 확인
for i in range(1000):
    _, loss = sess.run([train, cost], feed_dict=tensor_map)
    if i % 100 == 0:
        print "Step : ", i
        print "loss : ", loss

sess.close()

