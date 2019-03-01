import tensorflow as tf


def encoder(x, constants):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        w_mean = tf.get_variable(
            'mean',
            initializer=tf.truncated_normal([constants.num_latents],
                                            stddev=constants.std_init))
        w_sigma = tf.get_variable(
            'sigma',
            initializer=tf.truncated_normal([constants.num_latents],
                                            stddev=constants.std_init))
        mean = w_mean * x
        logstd = w_sigma * x
    return mean, logstd


def decoder(x):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 256, activation=tf.nn.relu)
        x = tf.layers.dense(x, 256, activation=tf.nn.relu)
        x = tf.layers.dense(x, 4 * 4 * 32, activation=tf.nn.relu)
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


def cross_entropy_loss(targets, logits):
    cross_entropy_per_logit = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits)
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(cross_entropy_per_logit, axis=1))
    return cross_entropy


def kl_divergence(encoder_output):
    mean, logstd = encoder_output
    kl_per_latent = -0.5 * (1 + 2 * logstd - (tf.exp(2 * logstd)
                            + tf.square(mean)))
    kl = tf.reduce_mean(tf.reduce_sum(kl_per_latent, axis=1))
    return kl, kl_per_latent
