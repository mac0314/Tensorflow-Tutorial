# This Python file uses the following encoding: utf-8

# https://youtu.be/j26O-216qww?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 4강. Model Restore

import tensorflow as tf


input_data = [[1, 5, 3, 7, 8, 10, 12],
              [5, 8, 10, 3, 9, 7, 1]
              ]
label_data = [[0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0]
              ]

# 사이즈 정의
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5


Learning_Rate = 0.05


# 배치 사이즈를 모를 경우 shape를 None으로 둔다.
x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])


# 정보를 Dictionary 형태로 만들어준다.
tensor_map = {x: input_data, y_: label_data}

# 저장했던 tf.Variable의 name과 일치해야한다.
# 마지막 파라미터에 'name='의 형식으로 넣어준다.
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


# Load variables (weight, bias)
param_list = [W_h1, B_h1, W_h2, B_h2, W_o, B_o]
# Create a saver
# 이름으로 리스트를 받아도 상관없음
saver = tf.train.Saver(param_list)


hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + B_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + B_h2)
y = tf.sigmoid(tf.matmul(hidden2, W_o) + B_o)


# y와 y_ 가장 큰 값의 인덱스를 구해서 비교한다, bool type으로 리턴
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 값을 캐스팅한 후 평균을 내 정확도를 체크한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

# Restore model
# Variable을 초기화해줄 필요가 없다.
# init = tf.global_variables_initializer()
# sess.run(init)
saver.restore(sess, './checkpoint/lecture_3.ckpt')

# Evaluate
for i in range(1000):
    acc = sess.run([accuracy], feed_dict=tensor_map)
    if i % 100 == 0:
        print "Step : ", i
        print "accuracy : ", acc

sess.close()

