# This Python file uses the following encoding: utf-8

# https://youtu.be/a74pFg8paVc?list=PL1H8jIvbSo1qlXVcdZTH2xsYFp3e1Nmjf
# Tensorflow 1강. Tensorflow의 자료형

import tensorflow as tf

placeholder = tf.placeholder(tf.float32, shape=[3, 3])
variables = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
constant = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)

'''
a = 5 * 4
print a

print placeholder
print variables
print constant

Tensorflow에서는 위처럼 직접 값에 접근하거나 계산하는 것을 허용하지 않음
Tensor의 메타 정보만 출력해줌
Tensorflow의 연산하는 방식이 따로 있음
세션에 담아 실행해야함

Python 	level
C 		level
Device 	level

Tensorflow는 파이썬 레벨에서 모든 자료를 Session에 저장하여 Device level에 임베딩한다.
이것을 C를 통해 세션의 연산을 실행함으로써 실행 속도를 증가시킨다.
파이썬은 C를 통해 만들어진 언어이기 때문에 호환이 잘된다.

constant는 상수를 저장하는 데이터 타입이다.
'''
a = tf.constant([5])
b = tf.constant([4])
c = tf.constant([10])

d = a * b + c

session = tf.Session()

result = session.run(d)

print result

'''
Variable 변수를 저장하는 데이터 타입으로 반드시 초기화가 필요한 자료형
weight를 담아두는데 사용함
global_variables_initializer 메소드를 통해 초기화함
세션을 통해 먼저 실행해주어야함
'''

var1 = tf.Variable([5])
var2 = tf.Variable([4])
var3 = tf.Variable([5])

var4 = var1 * var2 + var3

init = tf.global_variables_initializer()
session.run(init)

result = session.run(var4)

print result

'''
placeholder는 데이터를 담는 그릇
placeholder의 shape는 선택사항, 생략 가능함
'''

value1 = 5
value2 = 3
value3 = 2

ph1 = tf.placeholder(dtype=tf.float32)
ph2 = tf.placeholder(dtype=tf.float32)
ph3 = tf.placeholder(dtype=tf.float32)

result_value = ph1 * ph2 + ph3
feed_dict = {ph1: value1, ph2: value2, ph3: value3}

result = session.run(result_value, feed_dict)

print result

'''
placeholder는 애초에 그래프를 만들지 않음
placeholder는 input data와 label data을 텐서로 바꿔줄때 사용함
placeholder는 저장시키는 것이 아니라 Tensor와 Data를 매핑시킨다
이것을 feeding이라고 하는데 실행 이전에 해주어야한다.
'''

# session은 파일처럼 작용되기 때문에 닫아주어야함
session.close()