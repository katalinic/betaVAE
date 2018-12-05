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
flags.DEFINE_integer("epochs", 26, "Training epochs.")

DATASET_PATH = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
TEST_SIZE = 5000


class SpritesDataset():
    def __init__(self):
        dataset_zip = np.load(DATASET_PATH, encoding='latin1')
        imgs, latents = dataset_zip['imgs'], dataset_zip['latents_values']
        self.imgs = imgs.reshape(-1, 4096)
        latents = latents[:, 1:6]
        latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
        # Shuffle initially for train, test split since input is ordered.
        inds = np.arange(len(self))
        np.random.shuffle(inds)
        self.imgs = self.imgs[inds]
        self.latents = latents[inds]

    def __len__(self):
        return len(self.imgs)

    def gen(self):
        for i in range(0, len(self), FLAGS.batch_size):
            yield self.imgs[i:i + FLAGS.batch_size], self.latents[i:i + FLAGS.batch_size]


def input_pipeline(gen):
    dataset = tf.data.Dataset()
    dataset = dataset.from_generator(gen, output_types=(tf.uint8, tf.float32),
        output_shapes=((FLAGS.batch_size, 4096), (FLAGS.batch_size, FLAGS.num_latents)))
    train_dataset = dataset.skip(TEST_SIZE // FLAGS.batch_size)
    test_dataset = dataset.take(TEST_SIZE // FLAGS.batch_size)

    train_dataset = train_dataset.shuffle(buffer_size=5000)
    train_dataset = train_dataset.repeat(FLAGS.epochs)
    train_dataset = train_dataset.prefetch(1)
    train_iter = train_dataset.make_initializable_iterator()
    next_img_latent_pair = train_iter.get_next()

    test_dataset = test_dataset.repeat()  # Intentional infinite repeat.
    test_dataset = test_dataset.batch(TEST_SIZE // FLAGS.batch_size)
    test_iter = test_dataset.make_initializable_iterator()
    next_test_img_latent_pair = test_iter.get_next()

    return next_img_latent_pair, next_test_img_latent_pair, train_iter, test_iter

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

    sprites = SpritesDataset()
    # Train and test Datasets.
    next_img_latent_pair, next_test_img_latent_pair, train_iter, test_iter = \
        input_pipeline(sprites.gen)
    train_img, train_latent = next_img_latent_pair
    train_img = tf.to_float(train_img)
    test_img, test_latent = next_test_img_latent_pair
    test_img = tf.to_float(test_img)
    test_img = tf.reshape(test_img, (-1, 4096))
    test_latent = tf.reshape(test_latent, (-1, FLAGS.num_latents))

    # Train op counter.
    ep = tf.get_variable(
        "ep", [], tf.float32,
        initializer=tf.constant_initializer(0, dtype=tf.float32), trainable=False)
    ep_increment = tf.assign_add(ep, tf.constant(1, tf.float32))

    # VAE
    # Capacity.
    C = tf.maximum(0., ep - 20000.) * 0.75/10000. + 0.5

    loss, _ = build_VAE(train_latent, train_img, C)
    optimise = build_optimisation(loss)
    optimise = tf.group(optimise, ep_increment)

    # Test loss.
    _, kl_per_latent = build_VAE(test_latent, test_img, C)

    init = tf.group(
        tf.global_variables_initializer(),
        train_iter.initializer,
        test_iter.initializer)

    with tf.Session() as sess:
        sess.run(init)
        j = 0
        start_time = time.time()
        while True:
            try:
                sess.run(optimise)
            except tf.errors.OutOfRangeError:
                break
            j += 1
            # Evaluate every 6k training iterations.
            if j % 6000 == 0:
                k = sess.run(kl_per_latent)
                print('Iteration: {}, KL: {}, Time: {:.2f}'.format(
                    j, np.round(np.mean(k, 0), 1), time.time() - start_time))
                start_time = time.time()


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
