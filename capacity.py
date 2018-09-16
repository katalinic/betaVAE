import os
import time

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate.")
flags.DEFINE_float("epsilon", 1e-8, "Adam optimiser epsilon.")
flags.DEFINE_integer("gamma", 100, "Capacity constraint coefficient.")
flags.DEFINE_float("std_init", 1e-1, "Weight initialisation std.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("num_latents", 5, "Latent dimensionality.")


def load_sprites(only_images=True):
    dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
    imgs = dataset_zip['imgs'].reshape(-1, 4096)
    if only_images:
        return imgs
    else:
        latents = dataset_zip['latents_values']
        # Normalise factors.
        latents[:,1:6]=(latents[:,1:6]-np.mean(latents[:,1:6],axis=0))/np.std(latents[:,1:6],axis=0)
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


class capacity_bVAE(object):
    def __init__(self, x, y, num_latents, gamma, lr, max_epochs=350000):
        self.input_size = 4096
        self.y = y
        self.max_epochs = max_epochs
        self.ep = tf.get_variable("ep", [], tf.float32,
            initializer=tf.constant_initializer(0, dtype=tf.float32), trainable=False)
        self.ep_increment = tf.assign_add(self.ep, tf.constant(1, tf.float32))
        self.C = tf.maximum(0., self.ep - 20000.) * 0.75/10000. + 0.5
        self.kl_per_latent, self.ce, self.loss, self.optimise, self.mean, self.logstd, self.dec_out = \
            self.build_VAE(x, self.y, num_latents, self.C, gamma, lr)

    def encoder(self, x, num_latents):
        with tf.variable_scope("encoder"):
            self.w_mean = tf.Variable(tf.truncated_normal([num_latents], stddev=FLAGS.std_init))
            self.w_sigma = tf.Variable(tf.truncated_normal([num_latents], stddev=FLAGS.std_init))
            mean = self.w_mean * x
            logstd = self.w_sigma * x
        return mean, logstd

    def decoder(self, x, reuse=False):
        with tf.variable_scope("decoder"):
            x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
            x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
            x = tf.contrib.layers.fully_connected(x, 4 * 4 * 32, activation_fn=tf.nn.relu)
            x = tf.reshape(x, [-1] + [4, 4, 32])
            x = tf.layers.conv2d_transpose(x, 32, [4,4], strides=(2,2),
                                           activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d_transpose(x, 32, [4,4], strides=(2,2),
                                           activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d_transpose(x, 32, [4,4], strides=(2,2),
                                           activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d_transpose(x, 1, [4,4], strides=(2,2), padding='same')
            x = tf.reshape(x, [-1] + [self.input_size])
        return x

    def _build_enc_dec_connection(self, x, num_latents):
        mean, logstd = self.encoder(x, num_latents)
        eps = tf.random_normal(tf.shape(mean))
        non_sampled_z = mean + tf.exp(logstd)*eps
        dec_out = self.decoder(non_sampled_z)
        return mean, logstd, dec_out

    def _build_optimisation(self, mean, logstd, y, decoder_output, C, gamma, lr):
        cross_entropy_per_logit = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, logits=decoder_output)
        cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy_per_logit, axis=1))
        kl_per_latent = -0.5 * (1 + 2 * logstd - (tf.exp(2 * logstd) + tf.square(mean)))
        kl = tf.reduce_mean(tf.reduce_sum(kl_per_latent, axis=1))
        cap_loss = gamma * tf.abs(kl - C)
        loss = cross_entropy + cap_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=FLAGS.epsilon)
        optimise = tf.group(optimizer.minimize(loss), self.ep_increment)
        return kl_per_latent, cross_entropy, loss, optimise

    def build_VAE(self, x, y, num_latents, C, gamma, lr):
        mean, logstd, dec_out = self._build_enc_dec_connection(x, num_latents)
        kl_per_latent, ce, loss, optimise = self._build_optimisation(
            mean, logstd, y, dec_out, C, gamma, lr)
        return kl_per_latent, ce, loss, optimise, mean, logstd, dec_out


def train():
    tf.reset_default_graph()

    f = tf.placeholder(tf.float32, [None, FLAGS.num_latents])
    y = tf.placeholder(tf.float32, [None, 4096])
    vae = capacity_bVAE(f, y, num_latents=FLAGS.num_latents, gamma=FLAGS.gamma, lr=FLAGS.learning_rate)

    raw_train, raw_test = prepare_sprites()
    kl_monitor, ll_monitor = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        j = 0
        test_feed_dict = {f : raw_test[:, 1:6], vae.y : raw_test[:, 6:]}
        start_time = time.time()
        for ep in range(26):
            np.random.shuffle(raw_train)
            train_sprites = raw_train[:,6:]
            train_latents = raw_train[:,1:6]
            for i in range(int(train_sprites.shape[0]//FLAGS.batch_size)):
                batch_ys = train_sprites[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
                batch_latents = train_latents[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                train_feed_dict = {f : batch_latents, vae.y : batch_ys}
                if i > 0 and i % 6000 == 0:
                    j += 6000
                    k, l, c = sess.run([vae.kl_per_latent, vae.ce, vae.C], feed_dict=test_feed_dict)
                    print('Iteration: {}, KL: {}, C: {:.2f}, L: {:.2f}, Time: {:.2f}'.format(
                        j, np.round(np.mean(k, 0), 1), c, l, time.time() - start_time))
                    start_time = time.time()
                sess.run(vae.optimise, feed_dict=train_feed_dict)

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
