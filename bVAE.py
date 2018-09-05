import tensorflow as tf
import numpy as np

def encoder(x, num_latents):
    with tf.variable_scope("encoder"):
        enc_hid1 = tf.contrib.layers.fully_connected(x, 1200, activation_fn=tf.nn.relu)
        enc_hid2 = tf.contrib.layers.fully_connected(enc_hid1, 1200, activation_fn=tf.nn.relu)
        mean = tf.contrib.layers.fully_connected(enc_hid2, num_latents, activation_fn=None)
        var = tf.contrib.layers.fully_connected(enc_hid2, num_latents, activation_fn=None)
    return mean, var

def decoder(z, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        dec_hid1 = tf.contrib.layers.fully_connected(z, 1200, activation_fn=tf.nn.tanh)
        dec_hid2 = tf.contrib.layers.fully_connected(dec_hid1, 1200, activation_fn=tf.nn.tanh)
        dec_hid3 = tf.contrib.layers.fully_connected(dec_hid2, 1200, activation_fn=tf.nn.tanh)
        dec_out = tf.contrib.layers.fully_connected(dec_hid3, 4096, activation_fn=None)
    return dec_out

def enc_dec_connection(x, num_latents):
    mean, var = encoder(x, num_latents)
    eps = tf.random_normal(tf.shape(mean))
    non_sampled_z = mean + tf.exp(0.5*var)*eps#tf.sqrt(tf.exp(var))*eps
    dec_out = decoder(non_sampled_z)
    return mean, var, dec_out

def optimisation(mean, var, y, decoder_output, beta, lr):
    ce = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=decoder_output),axis=1))
    kl = -beta*tf.reduce_mean(0.5*tf.reduce_sum((1+var-(tf.exp(var)+tf.square(mean))),axis=1))
    loss = ce + kl
    optimizer = tf.train.AdagradOptimizer(learning_rate = lr)
    optimise = optimizer.minimize(loss)
    return loss, optimise

def build_VAE(x, y, z, num_latents, beta, lr):
    mean, var, dec_out = enc_dec_connection(x, num_latents)
    loss, optimise = optimisation(mean, var, y, dec_out, beta, lr)
    sample_dec_out = decoder(z, reuse=True)
    return loss, optimise, mean, var, sample_dec_out
