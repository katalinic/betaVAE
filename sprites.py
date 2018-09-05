import numpy as np

def load_sprites():
    dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                          encoding='latin1')
    imgs = dataset_zip['imgs']
    return imgs.reshape(-1, 4096)

def prepare_sprites():
    imgs = load_sprites()
    inds = np.arange(imgs.shape[0])
    np.random.shuffle(inds)
    test_inds = inds[:5000]
    train_inds = inds[5000:]
    return imgs[train_inds], imgs[test_inds]
