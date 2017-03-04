# This Python file uses the following encoding: utf-8

# https://youtu.be/IZu9Jkk3_HA?list=PL1H8jIvbSo1qOtjQXFzBxMWjL_Gc5x3yG
# Tensorflow 5강. Tensorboard 활용

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
W_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32, name='W_h1')
# hidden1 bias
B_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32, name='B_h1')

# hidden2 weight
W_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32, name='W_h2')
# hidden2 bias
B_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32, name='B_h2')

# output weight
W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32, name='W_o')
# output bias
B_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32, name='B_o')


# Load variables (weight, bias)
param_list = [W_h1, B_h1, W_h2, B_h2, W_o, B_o]
# Create a saver
# 이름으로 리스트를 받아도 상관없음
saver = tf.train.Saver(param_list)

with tf.name_scope('hidden_layer_1') as h1scope:
    hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + B_h1, name='hidden1')


with tf.name_scope('hidden_layer_2') as h2scope:
    hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + B_h2, name='hidden2')

with tf.name_scope('output_layer') as oscope:
    y = tf.sigmoid(tf.matmul(hidden2, W_o) + B_o, name='y')


# cost를 raw하게 계산
with tf.name_scope('calculate'):
    cost_ = -y_ * tf.log(y) - (1 - y_) * tf.log(1 - y)
    cost = tf.reduce_mean(tf.reduce_sum(cost_, reduction_indices=1))
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

with tf.name_scope('evaluate'):
    # y와 y_ 가장 큰 값의 인덱스를 구해서 비교한다, bool type으로 리턴
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 값을 캐스팅한 후 평균을 내 정확도를 체크한다.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:

    # Restore model
    # Variable을 초기화해줄 필요가 없다.
    #init = tf.global_variables_initializer()
    #sess.run(init)
    saver.restore(sess, './checkpoint/lecture_3.ckpt')
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/', sess.graph)

    # Evaluate
    for i in range(1001):
        summary, _, loss, acc = sess.run([merged, train, cost, accuracy], feed_dict=tensor_map)
        if i % 100 == 0:
            writer.add_summary(summary, i)
            #saver.save(sess, './checkpoint/lecture_3.ckpt')
            print "-------------------------"
            print "Step : ", i
            print "loss : ", loss
            print "accuracy : ", acc

