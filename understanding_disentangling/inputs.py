import tensorflow as tf
import numpy as np

DATASET_PATH = 'datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
TEST_SIZE = 5000


class SpritesDataset():
    def __init__(self, batch_size):
        dataset_zip = np.load(DATASET_PATH, encoding='latin1')
        imgs, latents = dataset_zip['imgs'], dataset_zip['latents_values']
        self.imgs = imgs.reshape(-1, 4096)
        latents = latents[:, 1:6]
        latents -= np.mean(latents, axis=0)
        latents /= np.std(latents, axis=0)
        # Shuffle initially for train, test split since input is ordered.
        inds = np.arange(len(self))
        np.random.shuffle(inds)
        self.imgs = self.imgs[inds]
        self.latents = latents[inds]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.imgs)

    def generator(self):
        for i in range(0, len(self), self.batch_size):
            yield (self.imgs[i:i + self.batch_size],
                   self.latents[i:i + self.batch_size])


def input_pipeline(generator, constants):
    dataset = tf.data.Dataset()
    dataset = dataset.from_generator(
        generator,
        output_types=(tf.uint8, tf.float32),
        output_shapes=((constants.batch_size, 4096),
                       (constants.batch_size, constants.num_latents)))
    train_dataset = dataset.skip(TEST_SIZE // constants.batch_size)
    test_dataset = dataset.take(TEST_SIZE // constants.batch_size)

    train_dataset = train_dataset.shuffle(buffer_size=5000)
    train_dataset = train_dataset.repeat(constants.num_epochs)
    train_dataset = train_dataset.prefetch(1)
    train_iter = train_dataset.make_initializable_iterator()
    next_img_latent_pair = train_iter.get_next()

    test_dataset = test_dataset.repeat()  # Intentional infinite repeat.
    test_dataset = test_dataset.batch(TEST_SIZE // constants.batch_size)
    test_iter = test_dataset.make_initializable_iterator()
    next_test_img_latent_pair = test_iter.get_next()

    return (next_img_latent_pair, next_test_img_latent_pair, train_iter,
            test_iter)


def prepare_train_test_inputs(constants):
    sprites = SpritesDataset(constants.batch_size)
    # Train and test Datasets.
    next_img_latent_pair, next_test_img_latent_pair, train_iter, test_iter = \
        input_pipeline(sprites.generator, constants)
    train_img, train_latent = next_img_latent_pair
    train_img = tf.to_float(train_img)
    test_img, test_latent = next_test_img_latent_pair
    test_img = tf.to_float(test_img)
    test_img = tf.reshape(test_img, (-1, 4096))
    test_latent = tf.reshape(test_latent, (-1, constants.num_latents))
    return ((train_latent, train_img, train_iter),
            (test_latent, test_img, test_iter))
