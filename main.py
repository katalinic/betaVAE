import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import bVAE
import sprites
import os
import time
import sys

num_epochs = 20
beta = 4
BATCH_SIZE = 32
lr = 1e-2
num_samples = 10 #number of samples to take
enc_num_input = 4096
dec_num_output = 4096
num_latents = 10
training = False
restore = False
sample_images = False

model_directory = './model_z{}_b{}/'.format(num_latents, beta)
plot_directory = './plots_z{}_b{}/'.format(num_latents, beta)
restore_directory = model_directory

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#Build graph
x = tf.placeholder(tf.float32, [None, enc_num_input])
y = tf.placeholder(tf.float32, [None, dec_num_output])
z = tf.placeholder(tf.float32, [None, num_latents])
loss, optimise, mean, var, sample_dec_out = bVAE.build_VAE(x, y, z, num_latents, beta, lr)

# Training.
if training:
    print ("Loading sprites.")
    train_sprites, test_sprites = sprites.prepare_sprites()
    #temporary subset
    train_sprites = train_sprites[:10000]
    print ("Training commenced.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            # Load model.
            saver = tf.train.Saver()
            saver.restore(sess, model_directory+'model.checkpoint')
        for ep in range(num_epochs):
            start_time = time.time()
            np.random.shuffle(train_sprites)
            for i in range(int(train_sprites.shape[0]//BATCH_SIZE)):
                batch_xs = train_sprites[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                # sess.run(optimise, feed_dict = {x: batch_xs, y: batch_xs})
                batch_loss, _ = sess.run([loss,optimise], feed_dict = {x: batch_xs, y: batch_xs})
                moving_loss = batch_loss if i==0 else 0.99*moving_loss+0.01*batch_loss
                sys.stdout.write('  ' + str(moving_loss) + ' ' + str(batch_loss)+'\r')
                # sys.stdout.flush()
            test_loss = sess.run(loss, feed_dict = {x: test_sprites, y: test_sprites})
            print ('Epoch : {}, Loss: {}, Time: {}'.format(ep, test_loss, time.time()-start_time))

            if sample_images:
                sampling_input = np.eye(num_latents)
                sampling_input = np.random.multivariate_normal(np.zeros(num_latents),np.eye(num_latents),num_samples).reshape(-1,num_latents)
                image_samples = sess.run(sample_dec_out, feed_dict = {z : sampling_input})
                fig = plt.figure(figsize=(10,10))
                for n in range(1,num_samples+1):
                    ax = plt.subplot(int(num_samples/2),2,n)
                    ax.imshow(image_samples[n-1].reshape(64,64),cmap='gray')
                plt.suptitle("Sampled digits")
                if not os.path.exists(plot_directory):
                    os.mkdir(plot_directory)
                plt.savefig(plot_directory+'sampled_sprites_e{}.png'.format(ep))

            # Save model.
            if not os.path.exists(model_directory):
                os.mkdir(model_directory)
            saver = tf.train.Saver()
            saver.save(sess, model_directory+'model.checkpoint',global_step=ep)

else:
    print ("Creating latent traversal.")
    with tf.Session() as sess:
        # Load model.
        saver = tf.train.Saver()
        saver.restore(sess, restore_directory+'model.checkpoint')

        image = sprites.load_sprites()[0].reshape(1,4096)
        _mean = sess.run(mean, feed_dict = {x : image})
        varied_mean = np.arange(-3, 3.01, 1)
        num_variations = len(varied_mean)

        # Generate variation images.
        if not os.path.exists(plot_directory):
            os.mkdir(plot_directory)
        fig = plt.figure(figsize=(10,10))
        for latent in range(num_latents):
            all_means = np.tile(_mean, num_variations).reshape(num_variations, -1)
            all_means[:,latent] = varied_mean
            image_samples = sess.run(sample_dec_out, feed_dict = {z : all_means})
            image_samples = sigmoid(image_samples)
            for n in range(1,num_variations+1):
                ax = plt.subplot(num_variations,1,n)
                ax.axis('off')
                ax.imshow(image_samples[n-1].reshape(64,64),cmap='gray')
            plt.savefig(plot_directory+'sprites_e{}.png'.format(latent),dpi=300, transparent=True, bbox_inches='tight')