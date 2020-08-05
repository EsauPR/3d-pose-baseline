""" Module to load and handle the train and test data """

import os
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from skimage.io import imread
from tqdm import tqdm

import cameras
import data_utils
from top_vae_3d_pose.args_def import ENVIRON as ENV
from top_vae_3d_pose.models import EFFICIENT_NET_INPUT_SHAPE

matplotlib.use('Agg')
# matplotlib.use('TkAgg')


def plot_history(data, xlabel='Epochs', ylabel='loss', fname='loss.png'):
    """ Plot """
    plt.figure(figsize=(12, 6))

    legends = []
    for pname, y_data in data:
        x_data = np.arange(len(y_data)) + 1
        plt.plot(x_data, y_data)
        legends.append(pname)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legends)
    plt.savefig('imgs/vae_concat_seq/%s' % fname)
    plt.close()


def save_history(data, fname):
    """ Save history in a file """
    with open('./experiments/vae_concat_seq/%s' % fname, 'wb') as f:
        np.save(f, np.array(data))


def add_noise(data):
    """ Add noise to the 3d joints
        one joint is select randomly and a bigger noise is added by 'joint_noise_factor'
    """
    data_noised = np.array(data)
    nsamples, points_dim = data.shape

    joint_noise_factor = ENV.FLAGS.noise_3d[1]
    noise = ENV.FLAGS.noise_3d[0]
    noise = 0.22108747

    # gaussian noise
    noised_all = np.random.randn(nsamples, points_dim) * noise + 0.0011787938

    # print(np.mean(noised_all))
    # print(np.std(noised_all))
    # print(np.min(noised_all))
    # print(np.max(noised_all))

    # Select the joint for each sample and add more noise
    joints_idx = np.random.choice(np.arange(points_dim), nsamples)
    noise_single_joint = np.random.randn(nsamples, 3) * (noise + joint_noise_factor)

    # Array of probs to add or no noise a any sample
    probs = np.random.randn(nsamples)

    for i in range(nsamples):
        if probs[i] < 0.5:
            continue
        data_noised[i] += noised_all[i]
        # Add noise for a single joint
        jid = joints_idx[i] - joints_idx[i] % 3
        data_noised[i][jid] += noise_single_joint[i][0]
        data_noised[i][jid + 1] += noise_single_joint[i][1]
        data_noised[i][jid + 2] += noise_single_joint[i][2]

    return data_noised


def get_data_params():
    """ Returns the actions, subjects and cams used """
    actions = data_utils.define_actions(ENV.FLAGS.action)
    # Load camera parameters
    subjects_ids = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(ENV.FLAGS.cameras_path, subjects_ids)

    return actions, subjects_ids, rcams


def load_3d_data_raw():
    """ Returns the 3d data """
    actions, _, rcams = get_data_params()
    return  data_utils.read_3d_data(actions,
                                    ENV.FLAGS.data_dir,
                                    ENV.FLAGS.camera_frame,
                                    rcams,
                                    ENV.FLAGS.predict_14)


def load_2d_data_raw():
    """ Returns the 2d data """
    actions, _, rcams = get_data_params()
    if ENV.FLAGS.use_sh:
        return data_utils.read_2d_predictions(actions, ENV.FLAGS.data_dir)
    return data_utils.create_2d_data(actions, ENV.FLAGS.data_dir, rcams)


def join_data(train, test, order_keys_train, order_keys_test):
    """ Join the data in the order given by order_keys_train and order_keys_test """
    # Join all the data over all the actions
    train = np.concatenate([train[key] for key in order_keys_train], axis=0).astype('float32')
    test = np.concatenate([test[key] for key in order_keys_test], axis=0).astype('float32')

    # Join Train an test
    return np.concatenate([train, test], axis=0)


def suffle_and_split(data, suffle_idx, train_size=0.8):
    """ Suffle the data and split in train and test """
    data = data[suffle_idx, :]
    tsize = int(data.shape[0] * train_size)
    train = data[:tsize, :]
    test = data[tsize:, :]
    return train, test



def load_3d_data():
    """ Read the 3d points from h3m return the train and test points """
    train_set_3d, test_set_3d, \
    data_mean_3d, data_std_3d, \
    dim_to_ignore_3d, dim_to_use_3d, \
    _, _ = load_3d_data_raw()

    # Join all the data over all the actions
    train_keys = list(train_set_3d.keys())
    test_keys = list(test_set_3d.keys())
    all_set = join_data(train_set_3d, test_set_3d, train_keys, test_keys)

    # pylint: disable=E1136  # pylint/issues/3139
    idx = np.random.choice(all_set.shape[0], all_set.shape[0], replace=False)
    train_set_3d, test_set_3d = suffle_and_split(all_set, idx)

    Metadata = namedtuple("Metadata", "mean std dim_ignored dim_used")
    meta = Metadata(data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d)

    train = Dataset(train_set_3d,
                    batch_size=ENV.FLAGS.batch_size,
                    shuffle=True,
                    metadata=meta)
    test = Dataset(test_set_3d,
                   batch_size=ENV.FLAGS.batch_size,
                   shuffle=True,
                   metadata=meta)

    return train, test



class Dataset(tf.keras.utils.Sequence):
    """ Dataset used to train only VAE model
        The input data are the 3d points and a gaussian noise is added to this
    """
    def __init__(self, data, batch_size=64, shuffle=True, metadata=None):
        self.data = data
        self.metadata = metadata
        self.shape = self.data.shape
        self.batch_size = batch_size
        self.batch_step = 0
        self.shuffle = shuffle
        self.indexes = np.arange(self.data.shape[0])
        self.data_noised = add_noise(self.data)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        return self.data_noised[idxs, :], self.data[idxs, :]


    def __iter__(self):
        return self


    def __next__(self):
        if self.batch_step == self.__len__():
            raise StopIteration()
        batch = self.__getitem__(self.batch_step)
        self.batch_step += 1
        return batch


    def on_epoch_end(self, new_noise=False):
        """ Updates indexes after each epoch """
        self.batch_step = 0
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
        # Add new noise after each epoch
        if new_noise:
            self.data_noised = add_noise(self.data)




def get_key3d(key2d):
    """ Returns the key for 3d dataset using the 2d key """
    camera_frame = ENV.FLAGS.camera_frame
    (subj, b, fname) = key2d
    # keys should be the same if 3d is in camera coordinates
    key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
    key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d
    return key3d


def keys2d_to_list(train_set_2d, test_set_2d, key2d_with_frame=False):
    """ Repeat each key by the length of the 2d data to match with the same index """
    keys_2d_train = list(train_set_2d.keys())
    keys_2d_test = list(test_set_2d.keys())

    keys = []
    for key in keys_2d_train:
        frame = 1
        for _ in range(train_set_2d[key].shape[0]):
            if key2d_with_frame:
                keys.append((*key, frame))
                frame += 1
            else :
                keys.append(key)

    for key in keys_2d_test:
        frame = 1
        for _ in range(test_set_2d[key].shape[0]):
            if key2d_with_frame:
                keys.append((*key, frame))
                frame += 1
            else :
                keys.append(key)

    return np.array(keys)


def load_frame_from_key(key2d, efficientnet_preprocess=False):
    """ Return the image frame that belongs to the 2d key """
    subject, _, action, frame = key2d
    action = action.replace('.h5', '')

    action_replaces = [
        ('WalkDog', 'WalkingDog', [1]),
        ('Photo', 'TakingPhoto', [1]),
    ]
    for ac_rpl in action_replaces:
        if ac_rpl[0] in action and int(subject) in ac_rpl[2]:
            action = action.replace(ac_rpl[0], ac_rpl[1])

    img_path = 'training/subject/S%s/image_frames/%s/frame_%06d.jpg' % (subject, action, int(frame))
    img_path = os.path.join(ENV.FLAGS.human_36m_path, img_path)
    img = imread(img_path)

    if efficientnet_preprocess:
        img_ccr = center_crop_and_resize(img, image_size=EFFICIENT_NET_INPUT_SHAPE[1])
        del img
        img_p = preprocess_input(img_ccr)
        del img_ccr
        return img_p

    return img


def load_frames_from_keys(keys2d, efficientnet_preprocess=False):
    """ Return the images that belong to the keys """
    return np.array([
        load_frame_from_key(key2d, efficientnet_preprocess=efficientnet_preprocess)
        for key2d in tqdm(keys2d, ascii=True, leave=False)
    ])


def load_2d_3d_data(return_raw=False, key2d_with_frame=False):
    """ Load the 2d and 3d data and returns two datasets for train and test """
    train_set_2d, test_set_2d, \
    data_mean_2d, data_std_2d, \
    dim_to_ignore_2d, dim_to_use_2d = load_2d_data_raw()

    train_set_3d, test_set_3d, \
    data_mean_3d, data_std_3d, \
    dim_to_ignore_3d, dim_to_use_3d, \
    train_root_positions, test_root_positions = load_3d_data_raw()

    if return_raw:
        RawDataset = namedtuple("RawDataset", "train test mean std dim_ignored dim_used train_root_pos test_root_pos")
        points2d = RawDataset(train_set_2d, test_set_2d,
                              data_mean_2d, data_std_2d,
                              dim_to_ignore_2d, dim_to_use_2d,
                              None, None)
        points3d = RawDataset(train_set_3d, test_set_3d,
                              data_mean_3d, data_std_3d,
                              dim_to_ignore_3d, dim_to_use_3d,
                              train_root_positions, test_root_positions)

        return points2d, points3d

    all_keys = keys2d_to_list(train_set_2d, test_set_2d, key2d_with_frame=key2d_with_frame)
    # pylint: disable=E1136  # pylint/issues/3139
    idx = np.random.choice(all_keys.shape[0], all_keys.shape[0], replace=False)
    train_keys, test_keys = suffle_and_split(all_keys, idx)

    # Get the keys for 2d points
    keys_2d_train = list(train_set_2d.keys())
    keys_2d_test = list(test_set_2d.keys())
    # join the train and test
    all_set = join_data(train_set_2d, test_set_2d, keys_2d_train, keys_2d_test)
    # shuffle and split
    train_set_2d, test_set_2d = suffle_and_split(all_set, idx)

    # Get the keys for 3d points in the same order of the 2d keys
    keys_3d_train = [get_key3d(key2d) for key2d in keys_2d_train]
    keys_3d_test = [get_key3d(key2d) for key2d in keys_2d_test]
    # join the train and test
    all_set = join_data(train_set_3d, test_set_3d, keys_3d_train, keys_3d_test)
    # shuffle and split with same 'idx'
    train_set_3d, test_set_3d = suffle_and_split(all_set, idx)

    # apply the same operations to root positions with the same order
    # join the train and test
    all_set = join_data(train_root_positions, test_root_positions, keys_3d_train, keys_3d_test)
    # shuffle and split with same 'idx'
    train_root_positions, test_root_positions = suffle_and_split(all_set, idx)

    # Create Objects from Dataset_2D_3D
    meta2d = Dataset2D3D.Metadata(data_mean_2d, data_std_2d,
                                  dim_to_ignore_2d, dim_to_use_2d,
                                  None)
    meta3d_train = Dataset2D3D.Metadata(data_mean_3d, data_std_3d,
                                        dim_to_ignore_3d, dim_to_use_3d,
                                        train_root_positions)
    meta3d_test = Dataset2D3D.Metadata(data_mean_3d, data_std_3d,
                                       dim_to_ignore_3d, dim_to_use_3d,
                                       test_root_positions)


    train = Dataset2D3D(train_set_2d, train_set_3d,
                        meta2d, meta3d_train,
                        train_keys,
                        batch_size=ENV.FLAGS.batch_size, shuffle=True)
    test = Dataset2D3D(test_set_2d, test_set_3d,
                       meta2d, meta3d_test,
                       test_keys,
                       batch_size=ENV.FLAGS.batch_size, shuffle=True)

    return train, test



class Dataset2D3D(tf.keras.utils.Sequence):
    """ Dataset used to train 3D_Pose + Vae model """
    Metadata = namedtuple("Metadata", "mean std dim_ignored dim_used root_positions")

    def __init__(self,
                 x_data,
                 y_data,
                 x_metadata,
                 y_metadata,
                 mapkeys,
                 batch_size=64,
                 shuffle=True):
        self.x_data = x_data
        self.y_data = y_data
        self.x_metadata = x_metadata
        self.y_metadata = y_metadata
        self.mapkeys = mapkeys
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_step = 0
        self.batch_idx = None
        self.indexes = np.arange(self.x_data.shape[0])


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.x_data.shape[0] / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.batch_idx = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        return self.x_data[self.batch_idx, :], self.y_data[self.batch_idx, :]


    def __iter__(self):
        return self


    def __next__(self):
        if self.batch_step == self.__len__():
            raise StopIteration()
        batch = self.__getitem__(self.batch_step)
        self.batch_step += 1
        return batch


    def on_epoch_end(self, avoid_suffle=False):
        """ Updates order indexes used for suffle after each epoch """
        self.batch_step = 0
        if avoid_suffle:
            return
        if self.shuffle is True:
            np.random.shuffle(self.indexes)


def load_dataset_3d_seq(seq_len=4):
    train_set_3d, test_set_3d, \
    data_mean_3d, data_std_3d, \
    dim_to_ignore_3d, dim_to_use_3d, \
    train_root_positions, test_root_positions = load_3d_data_raw()

    keys3d_train = list(train_set_3d.keys())
    keys3d_test = list(test_set_3d.keys())

    def get_next_seq(data, root_p):
        for i in range(data.shape[0] - seq_len):
            yield data[i:i+seq_len], data[i+seq_len-1], root_p[i:i+seq_len]

    def build(data_tmp, keys_tmp, data_tmp_root):
        x_seqs = []
        root_pos = []
        y_values = []
        mapkeys = []

        for key in keys_tmp:
            gen_seq = get_next_seq(data_tmp[key], data_tmp_root[key])
            x_s, y_v, rxy = zip(*[sv for sv in gen_seq])
            keys = [key for _ in range(len(x_s))]
            x_seqs += x_s
            y_values += y_v
            root_pos += rxy
            mapkeys += keys

        x_seqs = np.array(x_seqs)
        y_values = np.array(y_values)
        mapkeys = np.array(mapkeys)
        root_pos = np.array(root_pos)

        return x_seqs, y_values, mapkeys, root_pos


    x_train, y_train, mapkeys_train, rp_train = build(train_set_3d, keys3d_train, train_root_positions)
    x_test, y_test, mapkeys_test, rp_test = build(test_set_3d, keys3d_test, test_root_positions)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')


    meta_train = Dataset2D3D.Metadata(data_mean_3d,
                                      data_std_3d,
                                      dim_to_ignore_3d,
                                      dim_to_use_3d,
                                      rp_train)
    meta_test = Dataset2D3D.Metadata(data_mean_3d,
                                     data_std_3d,
                                     dim_to_ignore_3d,
                                     dim_to_use_3d,
                                     rp_test)

    train = Dataset2D3D(x_train,
                        y_train,
                        meta_train,
                        meta_train,
                        mapkeys_train,
                        batch_size=ENV.FLAGS.batch_size,
                        shuffle=True)
    test = Dataset2D3D(x_test,
                       y_test,
                       meta_test,
                       meta_test,
                       mapkeys_test,
                       batch_size=ENV.FLAGS.batch_size,
                       shuffle=True)

    return train, test
