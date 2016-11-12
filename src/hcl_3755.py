# coding: utf-8

'''
HCL 关键字 单字训练 CNN


'''
import sys
sys.path.append('../data_set/hcl/')
from hcl import input_data
import pickle
import os

hcl = input_data([1]+[i for i in range(3755)], 50, 20, True, True, (32, 32), False)

keep_prob = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5]

import tensorflow as tf
sess = tf.InteractiveSession()

#placeholder
x = tf.placeholder(tf.float32, shape=[None, 1024])
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
W_conv1 = weight_variable([3,3,1,50])
b_conv1 = bias_variable([50])

x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#2nd layer cnn
W_conv2 = weight_variable([3,3,50,100])
b_conv2 = bias_variable([100])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob[0])

h_pool2 = max_pool_2x2(h_conv2_drop)


#3rd layer cnn
W_conv3 = weight_variable([3,3,100,150])
b_conv3 = bias_variable([150])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob[1])


#4th layer cnn
W_conv4 = weight_variable([3,3,150,200])
b_conv4 = bias_variable([200])

h_conv4 = tf.nn.relu(conv2d(h_conv3_drop, W_conv4) + b_conv4)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob[2])

h_pool4 = max_pool_2x2(h_conv4_drop)

#5th layer cnn
W_conv5 = weight_variable([3,3,200, 250])
b_conv5 = bias_variable([250])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_conv5_drop = tf.nn.dropout(h_conv5, keep_prob[3])


#6th layer cnn
W_conv6 = weight_variable([3,3,250,300])
b_conv6 = bias_variable([300])

h_conv6 = tf.nn.relu(conv2d(h_conv5_drop, W_conv6) + b_conv6)
h_conv6_drop = tf.nn.dropout(h_conv6, keep_prob[4])

h_pool6 = max_pool_2x2(h_conv6_drop)


#7th layer cnn
W_conv7 = weight_variable([3,3,300,350])
b_conv7 = bias_variable([350])

h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
h_conv7_drop = tf.nn.dropout(h_conv7, keep_prob[5])


#8th layer cnn
W_conv8 = weight_variable([3,3,350,400])
b_conv8 = bias_variable([400])

h_conv8 = tf.nn.relu(conv2d(h_conv7_drop, W_conv8) + b_conv8)
h_conv8_drop = tf.nn.dropout(h_conv8, keep_prob[6])

h_pool8 = max_pool_2x2(h_conv8_drop)

#densely connected layer
W_fc9  = weight_variable([400*2*2, 1024])
b_fc9 = bias_variable([1024])

h_pool8_flat = tf.reshape(h_pool8, [-1, 400*2*2])
h_fc9 = tf.nn.relu(tf.matmul(h_pool8_flat, W_fc9) + b_fc9)
h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob[7])

#densely connected layer
W_fc10  = weight_variable([1024, 4096])
b_fc10 = bias_variable([4096])

h_fc10 = tf.nn.relu(tf.matmul(h_fc9_drop, W_fc10) + b_fc10)

#softmax layer to read out
W_fc11 = weight_variable([4096, 3755])
b_fc11 = bias_variable([3755])

y_conv = tf.nn.softmax(tf.matmul(h_fc10, W_fc11) +b_fc11)

#cost function, cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))

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
	batch = hcl.train.next_batch(20)
	#check out
	if i%7 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
		print 'step %d, train accuracy %g' %(i, train_accuracy)
	#train step run
	train_step.run(feed_dict={x:batch[0],y_:batch[1]})

#test result
batch = hcl.test.next_batch(50)
print 'test accuracy %g' % accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})

save_path = '/mnt/hgfs/3A/python/ml/hwr_pj/data_set/result/save3755.bin'
saver = tf.train.Saver()
save_path = saver.save(sess, save_path)
print 'save to %s' % save_path

'''
#restore

save_path = '/mnt/hgfs/3A/python/ml/hwr_pj/data_set/result/save3755.bin'
saver = tf.train.Saver()
saver.restore(sess, save_path)
print 'restore from %s' % save_path

#test result
batch = hcl.test.next_batch(50)
print 'test accuracy %g' % accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})

'''