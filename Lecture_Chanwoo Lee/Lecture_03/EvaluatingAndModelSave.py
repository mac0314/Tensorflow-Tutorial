# This Python file uses the following encoding: utf-8

# https://youtu.be/_PvUrx5KaIg?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 3강. Evaluating & Model save

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

# 마지막 파라미터에 'name='의 형식으로 값을 넣어서 변수의 이름을 지정한다.
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


# Save variables (weight, bias)
param_list = [W_h1, B_h1, W_h2, B_h2, W_o, B_o]
# Create a saver
saver = tf.train.Saver(param_list)

# 학습을 하지 않을 경우 그냥 랜덤, 카오스의 결과가 나온다.
# 특정한 값을 넣었을 때 같은 결과값을 받아오는 함수라고 말할 수 없다.
hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + B_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + B_h2)
y = tf.sigmoid(tf.matmul(hidden2, W_o) + B_o)


# cost를 raw하게 계산
cost_ = -y_ * tf.log(y) - (1 - y_) * tf.log(1 - y)
cost = tf.reduce_mean(tf.reduce_sum(cost_, reduction_indices=1))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

# y와 y_ 가장 큰 값의 인덱스를 구해서 비교한다, bool type으로 리턴
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 값을 캐스팅한 후 평균을 내 정확도를 체크한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# 이미 저장해놓은 weight, bias가 있고 추가로 학습하고 싶은 경우
# Variables를 초기화하는 부분을 지워준 후에 Saver를 restore하면 된다.

# Evaluate
for i in range(1000):
    _, loss, acc = sess.run([train, cost, accuracy], feed_dict=tensor_map)
    if i % 100 == 0:
        # 파일에 tf.Variable 이름과 값이 저장된다.
        # 이름을 지정해주지 않을 경우 파일에 저장한 값의 순서가 꼬일 수 있다.
        saver.save(sess, './checkpoint/lecture_3.ckpt')
        print "Step : ", i
        print "loss : ", loss
        print "accuracy : ", acc

sess.close()

