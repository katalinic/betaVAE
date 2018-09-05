import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model
import sprites

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer("enc_num_input", 4096, "Input dimensionality.")
flags.DEFINE_integer("dec_num_output", 4096, "Input dimensionality.")
flags.DEFINE_integer("num_latents", 10, "Latent dimensionality."
flags.DEFINE_integer("num_epochs", 20, "Number of training epochs.")
flags.DEFINE_integer("beta", 4, "Beta coefficient.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_float("lr", 1e-2, "Learning rate.")
flags.DEFINE_boolean("training", False, "Boolean for training or testing.")
flags.DEFINE_boolean("restore", False, "Boolean for restoring trained models.")
flags.DEFINE_integer("num_samples", 10, "Number of samples for visualisation.")

model_directory = './model_z{}_b{}/'.format(FLAGS.num_latents, FLAGS.beta)
plot_directory = './plots_z{}_b{}/'.format(FLAGS.num_latents, FLAGS.beta)
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

# Build graph.
x = tf.placeholder(tf.float32, [None, FLAGS.enc_num_input])
y = tf.placeholder(tf.float32, [None, FLAGS.dec_num_output])
z = tf.placeholder(tf.float32, [None, FLAGS.num_latents])
loss, optimise, mean, logstd, sample_dec_out = model.build_VAE(
    x, y, z, FLAGS.num_latents, FLAGS.beta, FLAGS.lr)

def train():
    print("Loading sprites.")
    train_sprites, test_sprites = sprites.prepare_sprites()
    print("Training commenced.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for ep in range(FLAGS.num_epochs):
            start_time = time.time()
            np.random.shuffle(train_sprites)
            for i in range(int(train_sprites.shape[0]//FLAGS.batch_size)):
                batch_xs = train_sprites[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                sess.run(optimise, feed_dict={x: batch_xs, y: batch_xs})
            test_loss = sess.run(loss, feed_dict={x: test_sprites, y: test_sprites})
            print('Epoch : {}, Loss: {}, Time: {}'.format(
                ep, test_loss, time.time() - start_time))

            # Save model.
            if not os.path.exists(model_directory):
                os.mkdir(model_directory)
            saver.save(sess, model_directory + 'model.checkpoint', global_step=ep)

def test():
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    print("Creating latent traversal.")
    with tf.Session() as sess:
        # Load model.
        saver = tf.train.Saver()
        chckpoint = tf.train.get_checkpoint_state(model_directory)
        saver.restore(sess, chckpoint.model_checkpoint_path)

        image = sprites.load_sprites()[0].reshape(1, 4096)
        _mean = sess.run(mean, feed_dict={x : image})
        varied_mean = np.arange(-3, 3.01, 1)
        num_variations = len(varied_mean)

        # Generate variation images.
        if not os.path.exists(plot_directory):
            os.mkdir(plot_directory)
        fig = plt.figure(figsize=(10, 10))
        for latent in range(FLAGS.num_latents):
            all_means = np.tile(_mean, num_variations).reshape(num_variations, -1)
            all_means[:, latent] = varied_mean
            image_samples = sess.run(sample_dec_out, feed_dict={z : all_means})
            image_samples = sigmoid(image_samples)
            for n in range(1, num_variations + 1):
                ax = plt.subplot(num_variations, 1, n)
                ax.axis('off')
                ax.imshow(image_samples[n - 1].reshape(64, 64), cmap='gray')
            plt.savefig(plot_directory + 'sprites_e{}.png'.format(latent),
                        dpi=300, transparent=True, bbox_inches='tight')

def main(_):
    if FLAGS.training:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
