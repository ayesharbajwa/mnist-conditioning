"""
Use fast gradient sign method to craft adversarial on MNIST.
Dependencies: python3, tensorflow v1.4, numpy, matplotlib

CODE adapted from: 
https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/example/fgmt_mnist.py
"""

from common_mnist_simple import *
tf.logging.set_verbosity(tf.logging.ERROR)

def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgm, feed_dict={
            env.x: X_data[start:end],
            env.fgm_eps: eps,
            env.fgm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv

def make_fgmt(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgm, feed_dict={
            env.x: X_data[start:end],
            env.fgm_eps: eps,
            env.fgm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


if __name__ == '__main__':

    attack = 'fgsm'
    
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = get_data()
    W = tf.Variable(tf.zeros([flattened_img_size, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))
    env = Dummy()

    print('\nConstruction graph')
    with tf.variable_scope('model'):
        env.x = tf.placeholder(tf.float32, (None, flattened_img_size), name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')
        env.ybar, logits = model(env.x, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                           logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()

    with tf.variable_scope('model', reuse=True):
        env.fgm_eps = tf.placeholder(tf.float32, (), name='fgm_eps')
        env.fgm_epochs = tf.placeholder(tf.int32, (), name='fgm_epochs')
        if attack == 'fgmt':
            env.x_fgm = fgmt(model, env.x, epochs=env.fgm_epochs, eps=env.fgm_eps)
        elif attack == 'fgsm':
            env.x_fgm = fgm(model, env.x, epochs=env.fgm_epochs, eps=env.fgm_eps)

    print('\nInitializing graph')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('\nTraining')
    train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
          name='mnist')

    print('\nEvaluating on clean data')
    evaluate(sess, env, X_test, y_test)

    print('\nGenerating adversarial data')
    if attack == 'fgmt':
        X_adv = make_fgmt(sess, env, X_test, eps=0.02, epochs=12)
    elif attack == 'fgsm':
        X_adv = make_fgsm(sess, env, X_test, eps=0.02, epochs=12)

    print('\nEvaluating on adversarial data')
    evaluate(sess, env, X_adv, y_test)

    plot_results(sess, env, X_test, y_test, X_adv, attack)
    print('\nFINISHED running ' + attack + ' MNIST for SIMPLE')
