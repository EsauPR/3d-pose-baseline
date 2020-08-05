""" Train a VAE model used to filter and enhance 3d points """

import json
from datetime import datetime

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import cameras
import data_utils
import viz
from top_vae_3d_pose import data_handler, losses, models
from top_vae_3d_pose.args_def import ENVIRON as ENV

import os
import sys

from skimage.io import imread

import efficientnet.tfkeras as efn
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

matplotlib.use('Agg')


def main():
    data2d, data3d = data_handler.load_2d_3d_data(key2d_with_frame=True)

    images = data_handler.load_frames_from_keys(data2d.mapkeys, efficientnet_preprocess=True)


    # model_2d23d = models.PoseBase()
    # # Dummy input for creation for bach normalization weigths
    # ainput = np.ones((10, 32), dtype=np.float32)
    # model_2d23d(ainput, training=False)
    # # Load weights for 2d to 3d prediction
    # model_2d23d.load_weights('pretrained_models/4874200_PoseBase/PoseBase')

    # log = []
    # for key2d in tqdm(data2d.train.keys(), ascii=True):
    #     key3d = data_handler.get_key3d(key2d)

    #     x_in = np.array_split(data2d.train[key2d], data2d.train[key2d].shape[0] // 256)
    #     x_out = np.array_split(data3d.train[key3d], data3d.train[key3d].shape[0] // 256)

    #     for x_batch, y_batch in zip(x_in, x_out):
    #         preds = model_2d23d(x_batch, training=False)
    #         log.append((preds - y_batch).numpy())
    #         del preds

    # tmp = np.concatenate(log)
    # del log
    # print(np.mean(tmp))
    # print(np.std(tmp))
    # print(np.min(tmp))
    # print(np.max(tmp))
    # del tmp

    # keys = [
    #     key for key in data2d.train.keys() if key[1] == 'Directions' and key[0] == 1
    # ]
    # print(keys)

    # key = (1, 'Directions', 'Directions 1.55011271.h5')

    # for key in keys:
    #     print(key, len(data2d.train[key]))




def main2():
    model = efn.EfficientNetB1(weights='imagenet')
    x = imread('imgs/panda.jpg')
    x = center_crop_and_resize(x, image_size=model.input_shape[1])
    x = preprocess_input(x)
    print(x.shape)

    img_tmp = x

    y = model.predict(np.expand_dims(x, 0))
    # y = model(np.expand_dims(x, 0), training=False)
    decode_predictions(y)
    print(y.shape)


if __name__ == "__main__":
    ENV.setup()
    main()
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    # with tf.device('/device:GPU:2'):
    #     main2()
