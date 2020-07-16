""" Module to load and handle the train and test data """

from collections import namedtuple

import numpy as np
import tensorflow as tf

import cameras
import data_utils
from top_vae_3d_pose.args_def import ENVIRON as ENV


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
        self.add_noise()


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
        'Updates indexes after each epoch'
        self.batch_step = 0
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
        # Add new noise after each epoch
        if new_noise:
            self.add_noise()


    def add_noise(self, data=None):
        """ Add noise to the 3d joints
            one joint is select randomly and a bigger noise is added by 'joint_noise_factor'
        """
        if data is None:
            data = self.data
        self.data_noised = np.array(data)
        nsamples, points_dim = data.shape

        joint_noise_factor = ENV.FLAGS.noise_3d[1]

        # gaussian noise
        noised_all = np.random.randn(nsamples, points_dim) * ENV.FLAGS.noise_3d[0]

        # Select the joint for each sample and add more noise
        joints_idx = np.random.choice(np.arange(points_dim), nsamples)
        noise_single_joint = np.random.randn(nsamples, 3) * (ENV.FLAGS.noise_3d[0] + joint_noise_factor)

        # Array of probs to add or no noise a any sample
        probs = np.random.randn(nsamples)

        for i in range(nsamples):
            if probs[i] < 0.5:
                continue
            self.data_noised[i] += noised_all[i]
            # Add noise for a single joint
            jid = joints_idx[i] - joints_idx[i] % 3
            self.data_noised[i][jid] += noise_single_joint[i][0]
            self.data_noised[i][jid + 1] += noise_single_joint[i][1]
            self.data_noised[i][jid + 2] += noise_single_joint[i][2]




def get_key3d(key2d):
    """ Returns the key for 3d dataset using the 2d key """
    camera_frame = ENV.FLAGS.camera_frame
    (subj, b, fname) = key2d
    # keys should be the same if 3d is in camera coordinates
    key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
    key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d
    return key3d


def keys2d_to_list(train_set_2d, test_set_2d):
    keys_2d_train = list(train_set_2d.keys())
    keys_2d_test = list(test_set_2d.keys())

    keys = []
    for key in keys_2d_train:
        for _ in range(train_set_2d[key].shape[0]):
            keys.append(key)

    for key in keys_2d_test:
        for _ in range(test_set_2d[key].shape[0]):
            keys.append(key)

    return np.array(keys)


def load_2d_3d_data():
    """ Load the 2d and 3d data and returns two datasets for train and test """
    train_set_2d, test_set_2d, \
    data_mean_2d, data_std_2d, \
    dim_to_ignore_2d, dim_to_use_2d = load_2d_data_raw()

    train_set_3d, test_set_3d, \
    data_mean_3d, data_std_3d, \
    dim_to_ignore_3d, dim_to_use_3d, \
    train_root_positions, test_root_positions = load_3d_data_raw()

    all_keys = keys2d_to_list(train_set_2d, test_set_2d)
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

    def __init__(self, x_data, y_data, x_metadata, y_metadata, mapkeys, batch_size=64, shuffle=True):
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


    def on_epoch_end(self):
        """ Updates order indexes used for suffle after each epoch """
        self.batch_step = 0
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
