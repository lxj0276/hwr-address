# coding: utf-8
# cnn_creater.py

import tensorflow as tf

class cnn:
	def __init__(self, x_shape, cnn_reshape, y_shape, cnn_layer_n, cnn_weights, keep_prob, fnn_reshape, fnn_layer_n, fnn_weights, softmax_weight, saver_path = None):
		self.x = tf.placeholder(tf.float32, shape=[None, x_shape])
		self.y_ = tf.placeholder(tf.float32, shape=[None, y_shape])
		self.sess = tf.InteractiveSession()
		self.cnn_layers = cnn.make_cnn_layers(self.x, cnn_reshape, cnn_layer_n, cnn_weights, keep_prob)
		self.fnn_layers = cnn.make_fnn_layers(self.cnn_layers[-1], cnn_layer_n, fnn_reshape, fnn_layer_n, fnn_weights, keep_prob)
		self.y_conv = cnn.make_softmax(self.fnn_layers[-1], softmax_weight)
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.sess.run(tf.initialize_all_variables())

	def train(self, data_set, steps, batch_size):
		for i in range(steps):
			batch = data_set.train.next_batch(batch_size)
			if i%10 == 0:
				train_accuracy = self.accuracy.eval(feed_dict={self.x:batch[0], self.y_:batch[1]})
				print 'step %d, train accuracy %g' %(i, train_accuracy)
			self.train_step.run(feed_dict={self.x:batch[0], self.y_:batch[1]})
	
	def restore(self, saver_path):
		saver = tf.train.Saver()
		saver.restore(self.sess, save_path)
	
	def test(self, data_set, size = 200):
		batch = data_set.test.next_batch(size)
		print 'test accuracy %g' % self.accuracy.eval(feed_dict={self.x:batch[0], self.y_:batch[1]})
		print self.y_conv.eval(feed_dict={self.x:batch[0], self.y_:batch[1]})

	def predict(self, x):
		return self.y_conv.eval(feed_dict={self.x:x})
		
	@staticmethod
	def make_cnn_layers(x, reshape, cnn_layer_n, weights, keep_prob):
		x_image = tf.reshape(x, reshape)
		cnn_layers = []
		f = lambda x, i: cnn.max_pool_2x2(tf.nn.dropout(tf.nn.relu(cnn.conv2d(x, cnn.weight_variable(weights[i])) + cnn.bias_variable([weights[i][-1]])), keep_prob[i]))
		x = f(x_image, 0)
		cnn_layers.append(x)
		for i in range(1,cnn_layer_n):
			x = f(x,i)
			cnn_layers.append(x)
		return cnn_layers
		
	@staticmethod
	def make_fnn_layers(x, cnn_layer_n, fnn_reshape, fnn_layer_n, fnn_weights, keep_prob):
		x = tf.reshape(x, fnn_reshape)
		f = lambda x,i: tf.nn.dropout(tf.nn.relu(tf.matmul(x, cnn.weight_variable(fnn_weights[i])) + cnn.bias_variable([fnn_weights[i][-1]])), keep_prob[cnn_layer_n + i])
		fnn_layers = []
		for i in range(0,fnn_layer_n):
			x = f(x,i)
			fnn_layers.append(x)
		return fnn_layers
		
	@staticmethod
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)
	@staticmethod
	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
	@staticmethod
	def conv2d(x, W):
		return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')
	@staticmethod
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	@staticmethod
	def make_softmax(x, weight):
		return tf.nn.softmax(tf.matmul(x, cnn.weight_variable(weight)) + cnn.bias_variable([weight[-1]]))

if __name__ == '__main__':
	x_shape = 1024
	cnn_reshape = [-1,32,32,1]
	y_shape = 10
	cnn_layer_n = 2
	cnn_weights = [[3, 3, 1, 32], [3, 3, 32, 64]]
	keep_prob = [1, 1, 1, 1, 0.5]
	fnn_reshape = [-1, 8*8*64]
	fnn_layer_n = 1
	fnn_weights = [[8*8*64, 1024]]
	softmax_weight = [1024, 10]
	a = cnn(x_shape, cnn_reshape, y_shape, cnn_layer_n, cnn_weights, keep_prob, fnn_reshape, fnn_layer_n, fnn_weights, softmax_weight)
	import sys
	sys.path.append('../data_set/hcl/')
	from hcl import input_data
	b = input_data(['省','市','县','区','乡','镇','村','巷','弄','X'], 50, 50, True, True, (32, 32), False)
	a.train(b, 2000, 50)
	a.test(b, 200)
	x = b.test.next_batch(20)
	print a.predict(x[0])