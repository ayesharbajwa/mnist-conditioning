# CODE adapted from https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/example/fgmt_mnist.py

"""
Use fast gradient sign method to craft adversarial on MNIST.
Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pprint
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp
from attacks import fgmt, fgm
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
img_size = 28
flattened_img_size = 784
img_chan = 1
n_classes = 10
VALIDATION_SPLIT = 0.1

W = tf.Variable(tf.zeros([flattened_img_size, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

def get_data():

    print('\nLoading MNIST')
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train, [-1, flattened_img_size])
    X_test = np.reshape(X_test, [-1, flattened_img_size])
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255

    to_categorical = tf.keras.utils.to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('\nSplitting data')
    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def model(x, logits=False, training=False):
    logits_ = tf.matmul(x, W) + b
    Winit = W.initialized_value()
    y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y


class Dummy:
    pass


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        ckpt = env.saver.save(sess, 'model/{}'.format(name))
        print(ckpt)

    env.saver.restore(sess, ckpt)      # restore from path returned by saver

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print('\n\nGETTING CONDITION NUMBERS\n')
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        if tensor.ndim > 1:
            print('tensor name:', key)
            print('tensor shape:', tensor.shape)
            print('cond number (L2):', condition_number(tensor, p=2))
            print('cond number (L1):', condition_number(tensor, p=1))
            print('cond number (inf):', condition_number(tensor, p=np.inf))
            print('cond number (fro):', condition_number(tensor, p='fro'))
            print('\n')


def condition_number(A, p):
    Aplus = np.linalg.pinv(A)
    return np.linalg.norm(A, ord=p) * np.linalg.norm(Aplus, ord=p)


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def plot_results(sess, env, X_test, y_test, X_adv, name):
    """
    Plot results and save figure
    """
    
    print('\nRandomly sample adversarial data from each category')
    y1 = predict(sess, env, X_test)
    y2 = predict(sess, env, X_adv)

    z0 = np.argmax(y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)

    X_tmp = np.empty((n_classes, img_size, img_size))
    y_tmp = np.empty((n_classes, n_classes))
    X_adv_reshaped = np.reshape(X_adv, [-1, img_size, img_size])
    for i in range(10):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        cur = np.random.choice(ind)
        X_tmp[i] = np.squeeze(X_adv_reshaped[cur])
        y_tmp[i] = y2[cur]

    print('\nPlotting results')
    fig = plt.figure(figsize=(n_classes, 1.2))
    gs = gridspec.GridSpec(1, n_classes, wspace=0.05, hspace=0.05)

    label = np.argmax(y_tmp, axis=1)
    proba = np.max(y_tmp, axis=1)
    for i in range(n_classes):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                      fontsize=12)

    print('\nSaving figure')
    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    fig_name = 'img/' + name + '_mnist_simple.png'
    plt.savefig(fig_name)
