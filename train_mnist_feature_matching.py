import argparse
import sys
import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input
from model import Discriminator, Generator


def noise_gen(batch_size, z_dim):
    noise = np.zeros((batch_size, z_dim), dtype=np.float32)
    for i in range(batch_size):
        noise[i, :] = np.random.uniform(0, 1, z_dim)
    return noise


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10)
args = parser.parse_args()
print(args)

# load MNIST data
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0] / args.batch_size)
testx = data['x_test']
testy = data['y_test'].astype(np.int32)
testy = np.reshape(testy, [testy.shape[0], 1])
nr_batches_test = int(testx.shape[0] / args.batch_size)

# select labeled data
rng = np.random.RandomState(args.seed)
data_rng = np.random.RandomState(args.seed_data)
inds = data_rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy == j][:args.count])
    tys.append(trainy[trainy == j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

# set up tensorflow and keras
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1)

# network
discriminator_model = Discriminator()
discriminator = discriminator_model.model
discriminator_feature = discriminator_model.feature
generator_model = Generator()
generator = generator_model.model

# loss function computation
x_label = Input([28 ** 2])
x_unlabel = Input([28 ** 2])
labels = Input([1], dtype=tf.int32)
noise = Input([100])
fake_image = generator(noise)
output_before_softmax_label = discriminator(x_label)
output_before_softmax_unlabel = discriminator(x_unlabel)
output_before_softmax_fake = discriminator(fake_image)

z_exp_label = tf.reduce_mean(tf.reduce_logsumexp(output_before_softmax_label))
z_exp_unlabel = tf.reduce_mean(tf.reduce_logsumexp(output_before_softmax_unlabel))
z_exp_fake = tf.reduce_mean(tf.reduce_logsumexp(output_before_softmax_fake))
index_flattened = tf.range(0, args.batch_size) * output_before_softmax_label.shape[1] + labels
l_label = tf.gather(tf.reshape(output_before_softmax_label, [-1]), index_flattened)
l_unlabel = tf.reduce_logsumexp(output_before_softmax_unlabel)
loss_label = -tf.reduce_mean(l_label) + tf.reduce_mean(z_exp_label)
loss_unlabel = -0.5 * tf.reduce_mean(l_unlabel) + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unlabel)) + \
               0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(output_before_softmax_fake)))
loss_discriminator = tf.add(loss_label, tf.multiply(loss_unlabel, args.unlabeled_weight))

feature_generated = tf.reduce_mean(discriminator_feature(fake_image), axis=0)
feature_real = tf.reduce_mean(discriminator_feature(x_unlabel), axis=0)
loss_generator = tf.reduce_mean(tf.square(feature_generated - feature_real))

train_err = tf.reduce_mean(
    tf.to_float(tf.not_equal(tf.argmax(output_before_softmax_label, axis=1), tf.cast(labels, tf.int64))))
test_error = tf.reduce_mean(
    tf.to_float(tf.not_equal(tf.argmax(output_before_softmax_label, axis=1), tf.cast(labels, tf.int64))))

# train settings
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
discriminator_gradients = discriminator_optimizer.compute_gradients(loss_discriminator, discriminator.trainable_weights)
# discriminator_grads_and_vars = zip(discriminator_gradients, discriminator.trainable_weights)
discriminator_train = discriminator_optimizer.apply_gradients(discriminator_gradients)

generator_optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
generator_gradients = generator_optimizer.compute_gradients(loss_generator, generator.trainable_weights)
# generator_grads_and_vars = zip(generator_gradients, generator.trainable_weights)
generator_train = generator_optimizer.apply_gradients(generator_gradients)

sess.run(tf.global_variables_initializer())

for epoch in range(300):
    begin = time.time()

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(trainx_unl.shape[0] / txs.shape[0])):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainy = np.reshape(trainy, [trainy.shape[0], 1])
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

    # train
    loss_label_record = 0.
    loss_unlabel_record = 0.
    train_err_record = 0.
    for t in range(nr_batches_train):
        noise_feed = noise_gen(args.batch_size, 100)
        _, loss_label_this, loss_unlabel_this, train_err_this = sess.run(
            [discriminator_train, loss_label, loss_unlabel, train_err], feed_dict={
                x_label: trainx[t * args.batch_size:(t + 1) * args.batch_size],
                x_unlabel: trainx_unl[t * args.batch_size:(t + 1) * args.batch_size],
                labels: trainy[t * args.batch_size:(t + 1) * args.batch_size],
                noise: noise_feed
            })
        loss_label_record += loss_label_this
        loss_unlabel_record += loss_unlabel_this
        train_err_record += train_err_this

        noise_feed = noise_gen(args.batch_size, 100)
        _, loss_generator_this = sess.run([generator_train, loss_generator], feed_dict={
            noise: noise_feed,
            x_unlabel: trainx_unl2[t * args.batch_size:(t + 1) * args.batch_size]
        })
    loss_label_record /= nr_batches_train
    loss_unlabel_record /= nr_batches_train
    train_err_record /= nr_batches_train

    # test
    test_err_record = 0.
    for t in range(nr_batches_test):
        test_err_this = sess.run(test_error, feed_dict={
            x_label: testx[t * args.batch_size:(t + 1) * args.batch_size],
            labels: testy[t * args.batch_size:(t + 1) * args.batch_size]
        })
        test_err_record += test_err_this
    test_err_record /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (
        epoch, time.time() - begin, loss_label_record, loss_unlabel_record, train_err_record, test_err_record))
    sys.stdout.flush()
