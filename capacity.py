import os
import time

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("lr", 5e-4, "Learning rate.")
flags.DEFINE_float("epsilon", 1e-8, "Adam optimiser epsilon.")
flags.DEFINE_integer("gamma", 100, "Capacity constraint coefficient.")
flags.DEFINE_float("std_init", 1e-1, "Weight initialisation std.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("num_latents", 5, "Latent dimensionality.")

DATASET_PATH = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

def load_sprites(only_images=True):
    dataset_zip = np.load(DATASET_PATH, encoding='latin1')
    imgs = dataset_zip['imgs'].reshape(-1, 4096)
    if only_images:
        return imgs
    else:
        latents = dataset_zip['latents_values']
        latents = latents[:, 1:6]
        # Normalise factors.
        latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
        images_and_latents = np.concatenate((latents, imgs), axis=1)
        return images_and_latents

def prepare_sprites():
    images_and_latents = load_sprites(False)
    inds = np.arange(images_and_latents.shape[0])
    np.random.shuffle(inds)
    # Assumed 5000 test examples.
    test_inds = inds[:5000]
    train_inds = inds[5000:]
    return images_and_latents[train_inds], images_and_latents[test_inds]


def encoder(x):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        w_mean = tf.get_variable(
            'mean', initializer=tf.truncated_normal([FLAGS.num_latents],
                                                    stddev=FLAGS.std_init))
        w_sigma = tf.get_variable(
            'sigma', initializer=tf.truncated_normal([FLAGS.num_latents],
                                                     stddev=FLAGS.std_init))
        mean = w_mean * x
        logstd = w_sigma * x
    return mean, logstd

def decoder(x):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 4*4*32, activation_fn=tf.nn.relu)
        x = tf.reshape(x, [-1] + [4, 4, 32])
        x = tf.layers.conv2d_transpose(x, 32, 4, 2, activation=tf.nn.relu,
                                       padding='same')
        x = tf.layers.conv2d_transpose(x, 32, 4, 2, activation=tf.nn.relu,
                                       padding='same')
        x = tf.layers.conv2d_transpose(x, 32, 4, 2, activation=tf.nn.relu,
                                       padding='same')
        x = tf.layers.conv2d_transpose(x, 1, 4, 2, padding='same')
        x = tf.reshape(x, [-1, 4096])
    return x

def build_enc_dec_connection(x):
    mean, logstd = encoder(x)
    eps = tf.random_normal(tf.shape(mean))
    non_sampled_z = mean + tf.exp(logstd) * eps
    dec_out = decoder(non_sampled_z)
    return mean, logstd, dec_out

def build_loss(mean, logstd, targets, decoder_output, C):

    def cross_entropy_loss(targets, logits):
        cross_entropy_per_logit = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=targets, logits=logits)
        cross_entropy = tf.reduce_mean(
            tf.reduce_sum(cross_entropy_per_logit, axis=1))
        return cross_entropy

    def kl_divergence(mean, logstd):
        kl_per_latent = -0.5 * (1 + 2 * logstd - (tf.exp(2 * logstd) + tf.square(mean)))
        kl = tf.reduce_mean(tf.reduce_sum(kl_per_latent, axis=1))
        return kl, kl_per_latent

    # Define loss.
    cross_entropy = cross_entropy_loss(targets, decoder_output)
    kl, kl_per_latent = kl_divergence(mean, logstd)
    capacity_loss = FLAGS.gamma * tf.abs(kl - C)
    loss = cross_entropy + capacity_loss

    return loss, kl_per_latent

def build_optimisation(loss):
    # Train op.
    optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.lr, epsilon=FLAGS.epsilon)
    optimise = optimizer.minimize(loss)
    return optimise

def build_VAE(x, y, C):
    mean, logstd, dec_out = build_enc_dec_connection(x)
    loss, kl_per_latent = build_loss(mean, logstd, y, dec_out, C)
    return loss, kl_per_latent


def train():
    tf.reset_default_graph()

    f = tf.placeholder(tf.float32, [None, FLAGS.num_latents])
    y = tf.placeholder(tf.float32, [None, 4096])

    # Train op counter.
    ep = tf.get_variable(
        "ep", [], tf.float32,
        initializer=tf.constant_initializer(0, dtype=tf.float32), trainable=False)
    ep_increment = tf.assign_add(ep, tf.constant(1, tf.float32))

    # VAE
    # Capacity.
    C = tf.maximum(0., ep - 20000.) * 0.75/10000. + 0.5

    loss, kl_per_latent = build_VAE(f, y, C)
    optimise = build_optimisation(loss)
    optimise = tf.group(optimise, ep_increment)

    raw_train, raw_test = prepare_sprites()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        j = 0
        test_feed_dict = {f: raw_test[:, :FLAGS.num_latents],
                          y: raw_test[:, FLAGS.num_latents:]}
        start_time = time.time()
        for ep in range(26):

            np.random.shuffle(raw_train)
            train_sprites = raw_train[:, FLAGS.num_latents:]
            train_latents = raw_train[:, :FLAGS.num_latents]

            for i in range(int(train_sprites.shape[0]//FLAGS.batch_size)):
                j += 1
                if j > 0 and j % 6000 == 0:
                    k = sess.run(kl_per_latent, feed_dict=test_feed_dict)
                    print('Iteration: {}, KL: {}, Time: {:.2f}'.format(
                        j, np.round(np.mean(k, 0), 1), time.time() - start_time))
                    start_time = time.time()

                train_feed_dict = {
                    f: train_latents[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size],
                    y: train_sprites[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]}
                sess.run(optimise, feed_dict=train_feed_dict)


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
