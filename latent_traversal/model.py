import tensorflow as tf


def encoder(x, num_latents):
    with tf.variable_scope("encoder"):
        x = tf.layers.dense(x, 1200, activation=tf.nn.relu)
        x = tf.layers.dense(x, 1200, activation=tf.nn.relu)
        mean = tf.layers.dense(x, num_latents, activation=None)
        logstd = tf.layers.dense(x, num_latents, activation=None)
    return mean, logstd


def decoder(x, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        x = tf.layers.dense(x, 1200, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 1200, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 1200, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 4096, activation=None)
    return x


def enc_dec_connection(x, num_latents):
    mean, logstd = encoder(x, num_latents)
    eps = tf.random_normal(tf.shape(mean))
    non_sampled_z = mean + tf.exp(logstd) * eps
    dec_out = decoder(non_sampled_z)
    return mean, logstd, dec_out


def optimisation(mean, logstd, y, decoder_output, beta, lr):
    cross_entropy_per_logit = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=decoder_output)
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(cross_entropy_per_logit, axis=1))
    kl_per_latent = -0.5 * (1 + 2 * logstd - (tf.exp(2 * logstd)
                            + tf.square(mean)))
    kl = tf.reduce_mean(tf.reduce_sum(kl_per_latent, axis=1))
    loss = cross_entropy + kl
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    optimise = optimizer.minimize(loss)
    return loss, optimise


def build_VAE(x, y, z, num_latents, beta, lr):
    mean, logstd, dec_out = enc_dec_connection(x, num_latents)
    loss, optimise = optimisation(mean, logstd, y, dec_out, beta, lr)
    sample_dec_out = decoder(z, reuse=True)
    return loss, optimise, mean, logstd, sample_dec_out
