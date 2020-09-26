""" Process all the Human3.6m dataset throuh the model efficient net """
import os
import sys

import efficientnet.tfkeras as efn
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from top_vae_3d_pose.args_def import BAR_FORMAT, ENVIRON
from top_vae_3d_pose.data_handler import (load_2d_data_raw,
                                          load_frames_from_keys)
from top_vae_3d_pose.models import EFFICIENT_NET_INPUT_SHAPE


def key2d_to_framekey2d(key2d, data):
    nframes = data[key2d].shape[0]
    return [(*key2d, i+1) for i in range(nframes)]


def process(h5file, effnet, keys, data):
    """ Process """
    for key in tqdm(keys):
        frame_keys = key2d_to_framekey2d(key, data)
        images = load_frames_from_keys(frame_keys, efficientnet_preprocess=False)

        grp = h5file.create_group("%s/%s/%s" % (key[0], key[1], key[2].replace('.h5', '')))
        tqdm.write(grp.name)

        results = []
        nframes = images.shape[0]

        idx = 0
        while idx < nframes:
            result = effnet(images[idx:idx + ENVIRON.FLAGS.batch_size], training=False).numpy()
            results.append(result)
            idx += ENVIRON.FLAGS.batch_size

        results = np.concatenate(results, axis=0)
        grp.create_dataset('data', data=results)
        tqdm.write(str(results.shape))


def main():
    effnet = efn.EfficientNetB0(weights='imagenet',
                                include_top=False,
                                pooling='max',
                                input_shape=EFFICIENT_NET_INPUT_SHAPE)

    print("effnet out shape", effnet.layers[-1].output_shape)

    train_set_2d, test_set_2d, _, _, _, _ = load_2d_data_raw()

    h5file = h5py.File('data/human36m_effnet.hdf5', 'w')

    process(h5file, effnet, list(train_set_2d.keys()), train_set_2d)
    process(h5file, effnet, list(test_set_2d.keys()), test_set_2d)

    h5file.close()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    ENVIRON.setup()
    main()
