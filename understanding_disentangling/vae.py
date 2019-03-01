import tensorflow as tf

from networks import encoder, decoder, cross_entropy_loss, kl_divergence


def build_enc_dec_connection(observation, constants):
    mean, logstd = encoder(observation, constants)
    eps = tf.random_normal(tf.shape(mean))
    non_sampled_z = mean + tf.exp(logstd) * eps
    dec_out = decoder(non_sampled_z)
    return (mean, logstd), dec_out


def build_loss(encoder_output, decoder_output, target, capacity,
               capacity_coef):
    cross_entropy = cross_entropy_loss(target, decoder_output)
    kl, kl_per_latent = kl_divergence(encoder_output)
    capacity_loss = capacity_coef * tf.abs(kl - capacity)
    loss = cross_entropy + capacity_loss
    return loss, kl_per_latent


def build_optimisation(loss, constants):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=constants.lr, epsilon=constants.epsilon)
    optimise = optimizer.minimize(loss)
    return optimise


def build_VAE(observation, target, capacity, constants):
    encoder_output, decoder_output = build_enc_dec_connection(
        observation, constants)
    loss, kl_per_latent = build_loss(encoder_output, decoder_output, target,
                                     capacity, constants.capacity_coef)
    return loss, kl_per_latent
