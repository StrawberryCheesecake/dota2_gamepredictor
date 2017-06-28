import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant([[1],[2],[3],[4]])
b = tf.constant([[1,2,3,4]])
c = tf.matmul(b,a)
sess = tf.Session()
print(sess.run(c))
sess.close()
