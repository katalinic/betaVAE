import time

import tensorflow as tf
import numpy as np

from inputs import prepare_train_test_inputs
from vae import build_VAE, build_optimisation

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("lr", 5e-4, "Learning rate.")
flags.DEFINE_float("epsilon", 1e-8, "Adam optimiser epsilon.")
flags.DEFINE_integer("capacity_coef", 100, "Capacity constraint coefficient.")
flags.DEFINE_float("std_init", 1e-1, "Weight initialisation std.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("num_latents", 5, "Latent dimensionality.")
flags.DEFINE_integer("num_epochs", 26, "Training epochs.")


def train(constants):
    tf.reset_default_graph()

    train_input, test_input = prepare_train_test_inputs(constants)
    train_latent, train_img, train_iter = train_input
    test_latent, test_img, test_iter = test_input

    ep = tf.get_variable(
        "ep", [], tf.float32,
        initializer=tf.constant_initializer(
            0, dtype=tf.float32), trainable=False)
    ep_increment = tf.assign_add(ep, tf.constant(1, tf.float32))

    C = tf.maximum(0., ep - 20000.) * 0.75/10000. + 0.5

    loss, _ = build_VAE(train_latent, train_img, C, constants)
    optimise = build_optimisation(loss, constants)
    optimise = tf.group(optimise, ep_increment)

    # Test loss.
    _, kl_per_latent = build_VAE(test_latent, test_img, C, constants)

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
    train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
