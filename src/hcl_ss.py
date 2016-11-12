# coding: utf-8

'''
HCL 关键字 单字训练 CNN


'''


import sys
sys.path.append('../data_set/hcl/')
from hcl import input_data

hcl = input_data([1]+[i for i in range(3755)], 20, 2, True, True, (28, 28), False)

import tensorflow as tf
sess = tf.InteractiveSession()

#placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 3755])

#variable initialize
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#convolution function
def conv2d(x, W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

#pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#1st layer convolutional neural network
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd layer cnn
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#densely connected layer
W_fc1  = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax layer to read out
W_fc2 = weight_variable([1024, 3755])
b_fc2 = bias_variable([3755])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) +b_fc2)

#cost function, cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

#train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#correction and accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initialize variables
sess.run(tf.initialize_all_variables())

#train
for i in range(2000):
	#train object
	batch = hcl.train.next_batch(10)
	#check out
	if i%7 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print 'step %d, train accuracy %g' %(i, train_accuracy)
	#train step run
	train_step.run(feed_dict={x:batch[0],y_:batch[1], keep_prob:0.5})

#test result
batch = hcl.test.next_batch(50)
print 'test accuracy %g' % accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})

save_path = '/mnt/hgfs/3A/python/ml/hwr_pj/data_set/result/saveSSX.bin'
saver = tf.train.Saver()
save_path = saver.save(sess, save_path)
print 'save to %s' % save_path

#restore
'''
save_path = '/mnt/hgfs/3A/python/ml/hwr_pj/data_set/result/saveSSX.bin'
saver = tf.train.Saver()
saver.restore(sess, save_path)
print 'restore from %s' % save_path

batch = hcl.test.next_batch(50)
print 'test accuracy %g' % accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
'''