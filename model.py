import tensorflow as tf
import numpy as np

def cnn(X):

	conv1 = tf.layers.conv2d(
		inputs=X,
		filters=32,
		kernel_size=[1, 7],
		strides=[1,2],
		padding='same',
		activation=tf.nn.relu)
	pool1=tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=[1, 2],
		strides=[1,2])

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=16,
		kernel_size=[1,5],
		strides=1,
		padding='same',
		activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=[1,2],
		strides=[1,2])

	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=8,
		kernel_size=[1,3],
		strides=1,
		padding='same',
		activation=tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(
		inputs=conv3,
		pool_size=[1,2],
		strides=[1,2])

	flat4 = tf.layers.flatten(inputs=pool3)
	fc4 = tf.layers.dense(inputs=flat4, units=32)

	fc5 = tf.layers.dense(inputs=fc4, units=6)
	logits = tf.nn.softmax(logits=fc5)

	return logits
