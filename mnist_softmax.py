# based on tutorial at https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners
# test accuracy is about 92%

import argparse
import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	# create model
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.matmul(x, W) + b

	y_ = tf.placeholder(tf.float32, [None, 10])

	# training setup, using NLL
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# start session
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	# run training
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # TEST CONDITION NUMBER
	Wnp = W.eval(session=sess)
	print('CONDITION NUMBER:', np.linalg.cond(Wnp))        

	# test model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

	
