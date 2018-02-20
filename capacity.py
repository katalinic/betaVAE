import tensorflow as tf
import numpy as np
from sprites import load_sprites
import time

'''
Assuming a training iteration is one batch, 300000 of them, with assumed batch size of 32, that's 10 million
frames, or around 13 epochs; if 64, then 26 epochs
'''
class capacity_bVAE(object):
    def __init__(self, x, z, num_latents=10, batch_size=32, max_epochs=300000, gamma=1000, lr=5e-4):
        #assuming x is 4d tensor form
        self.input_size = np.prod(x.get_shape().as_list()[1:])
        y = tf.reshape(x, [-1]+[self.input_size]) #for sprites should be 4096
        self.max_epochs = max_epochs
        self.ep = tf.get_variable("ep", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
        self.ep_increment = tf.assign_add(self.ep, tf.constant(1, tf.int32))
        self.C = tf.train.polynomial_decay(0.5, self.ep, self.max_epochs, 25, power = 1.)
        self.kl_per_latent, self.ce, self.loss, self.optimise, self.mean, self.logstd, self.sample_dec_out = \
            self.build_VAE(x, y, z, num_latents, self.C, gamma, lr)

    def encoder(self, x, num_latents):
        with tf.variable_scope("encoder"):
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[4,4], strides=(2,2), activation = tf.nn.relu, padding='same') #32 32 32
            conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[4,4], strides=(2,2), activation = tf.nn.relu, padding='same') #16 16 32
            conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=[4,4], strides=(2,2), activation = tf.nn.relu, padding='same') #8 8 32
            conv4 = tf.layers.conv2d(conv3, filters=32, kernel_size=[4,4], strides=(2,2), activation = tf.nn.relu, padding='same') #4 4 32
            self.conv4_shape = conv4.get_shape().as_list()[1:]
            self.flat_dim = int(np.prod(self.conv4_shape))
            flattened = tf.reshape(conv4, [-1,self.flat_dim])
            enc_hid1 = tf.contrib.layers.fully_connected(flattened, 256, activation_fn = tf.nn.relu)
            enc_hid2 = tf.contrib.layers.fully_connected(enc_hid1, 256, activation_fn = tf.nn.relu)
            mean = tf.contrib.layers.fully_connected(enc_hid2, num_latents, activation_fn=None)
            logstd = tf.contrib.layers.fully_connected(enc_hid2, num_latents, activation_fn=None)
        return mean, logstd

    def decoder(self, z, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            dec_hid1 = tf.contrib.layers.fully_connected(z, 256, activation_fn = tf.nn.relu)
            dec_hid2 = tf.contrib.layers.fully_connected(dec_hid1, self.flat_dim, activation_fn = tf.nn.relu)
            dec_hid2 = tf.reshape(dec_hid2, [-1]+self.conv4_shape)
            de_conv4 = tf.layers.conv2d_transpose(dec_hid2, 32, [4,4], strides=(2,2), padding='same')
            de_conv3 = tf.layers.conv2d_transpose(de_conv4, 32, [4,4], strides=(2,2), padding='same')
            de_conv2 = tf.layers.conv2d_transpose(de_conv3, 32, [4,4], strides=(2,2), padding='same')
            de_conv1 = tf.layers.conv2d_transpose(de_conv2, 1, [4,4], strides=(2,2), padding='same')
            dec_out = tf.reshape(de_conv1, [-1]+[self.input_size])
        return dec_out

    def _build_enc_dec_connection(self, x, num_latents):
        mean, logstd = self.encoder(x, num_latents)
        eps = tf.random_normal(tf.shape(mean))
        non_sampled_z = mean + tf.exp(logstd)*eps
        dec_out = self.decoder(non_sampled_z)
        return mean, logstd, dec_out

    def _build_optimisation(self, mean, logstd, y, decoder_output, C, gamma, lr):
        ce = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=decoder_output),axis=1))
        kl_per_latent = -0.5*(1+2*logstd-(tf.exp(2*logstd)+tf.square(mean)))
        reduced_kl = tf.reduce_mean(tf.reduce_sum(kl_per_latent,axis=1))
        cap_loss = gamma*tf.losses.absolute_difference(reduced_kl, C)
        loss = ce + cap_loss
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        optimise = tf.group(optimizer.minimize(loss),self.ep_increment)
        return kl_per_latent, ce, loss, optimise

    def build_VAE(self, x, y, z, num_latents, C, gamma, lr):
        mean, logstd, dec_out = self._build_enc_dec_connection(x, num_latents)
        kl_per_latent, ce, loss, optimise = self._build_optimisation(mean, logstd, y, dec_out, C, gamma, lr)
        sample_dec_out = self.decoder(z, reuse=True)
        return kl_per_latent, ce, loss, optimise, mean, logstd, sample_dec_out

BATCH_SIZE = 32
num_latents = 10

#maybe device place on cpu
raw = load_sprites().reshape(-1,64,64,1)
input_placeholder = tf.placeholder(tf.float32, raw.shape)
input_data = tf.Variable(input_placeholder, trainable=False, collections=[])

queue = tf.FIFOQueue(capacity=50, dtypes=tf.float32, shapes=[64, 64, 1])
enqueue_op = queue.enqueue_many(input_data)

x = queue.dequeue_many(BATCH_SIZE)
qr = tf.train.QueueRunner(queue, [enqueue_op] * 2) #2 here set for num threads
tf.train.add_queue_runner(qr)

z = tf.placeholder(tf.float32, [None, num_latents])
#feed both x and y as x
vae = capacity_bVAE(x, z)

kl_monitor = []
ll_monitor = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(input_data.initializer, feed_dict = {input_placeholder : raw})
    coord = tf.train.Coordinator()
    enqueue_threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    start_time = time.time()

    for i in range(vae.max_epochs):
    # for i in range(1000):
        if i % 1000 == 0:
            k, l, _ = sess.run([vae.kl_per_latent, vae.ce, vae.optimise])
            kl_monitor.append(k)
            ll_monitor.append(l)
        else:
        # k, l, _ = sess.run([vae.kl_per_latent, vae.ce, vae.optimise])
            sess.run(vae.optimise)
    print (time.time()-start_time)

    coord.request_stop()
    coord.join(enqueue_threads)

    kl_arr = np.array(kl_monitor)
    ll_arr = np.array(ll_monitor)

    np.save('kl', kl_arr)
    np.save('ll', ll_arr)
