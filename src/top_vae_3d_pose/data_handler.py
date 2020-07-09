from collections import namedtuple
from datetime import datetime

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import cameras
import data_utils
import viz
from top_vae_3d_pose.args_def import ENVIRON as ENV

matplotlib.use('Agg')


def gen_sample_img(real_points, noised_points, mean, std, dim_ignored, max_factor, model=None, idx=None):
    """ Plot 3d poses, real, with noise and decode from vae model if a model is provided
        pass 'idx' to select samples otherwise idx will be randomly generated
    """
    # select random samples
    nsamples = 9
    if idx is None:
        idx = np.random.choice(real_points.shape[0], nsamples, replace=False)
    real_points = real_points[idx, :]
    noised_points = noised_points[idx, :]

    # Use the model to generate new samples or use the noised samples provided
    if model is not None:
        z = model.reparametrize(*model.encode(noised_points))
        if ENV.FLAGS.f_loss == 'xent':
            pred_points = model.decode(z).numpy()
        else:
            pred_points = model.decode(z).numpy()

    real_points *= max_factor
    noised_points *= max_factor

    # unnormalioze data
    real_points = data_utils.unNormalizeData(real_points, mean, std, dim_ignored)
    noised_points = data_utils.unNormalizeData(noised_points, mean, std, dim_ignored)
    if model is not None:
        pred_points = data_utils.unNormalizeData(pred_points, mean, std, dim_ignored)

    # def cam2world_centered(data_3d_camframe):
    #     data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
    #     data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
    #     # subtract root translation
    #     return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )
    # poses3d = cam2world_centered(poses3d)

    # Make athe plot and save the fig
    fig = plt.figure(figsize=(19.2, 10.8))
    # fig = plt.figure(figsize=(10.24, 7.48))
    if model is None:
        gs1 = gridspec.GridSpec(3, 6) # 5 rows, 9 columns
    else:
        gs1 = gridspec.GridSpec(3, 9) # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 1, 1
    for _ in np.arange(nsamples):
        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx-1], projection='3d')
        p3d = real_points[exidx-1, :]
        # print('3D points', p3d)
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

        ax3 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = noised_points[exidx-1, :]
        # print('3D points', p3d)
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

        if model is not None:
            ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
            p3d = pred_points[exidx-1, :]
            # print('3D points', p3d)
            viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

        exidx = exidx + 1
        if model is None:
            subplot_idx = subplot_idx + 2
        else:
            subplot_idx = subplot_idx + 3

    # plt.show()
    if ENV.FLAGS.noised_sample:
        file_name = "imgs/vae/noised_%f_%f.png" % (ENV.FLAGS.noise_3d[0], ENV.FLAGS.noise_3d[1])
    else:
        file_name = "imgs/vae/%s.png" % datetime.utcnow().isoformat()
    plt.savefig(file_name)
    print("Saved samples on: %s" % file_name)
    plt.close()


def plot_history(train_loss, test_loss, error_pred, error_noised):
    x_data = np.arange(len(train_loss)) + 1

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, train_loss)
    plt.plot(x_data, test_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["Train loss", "Test loss"])
    plt.savefig('imgs/vae/0_loss.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, error_pred)
    plt.plot(x_data, error_noised)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(["Error pred", "Error noised"])
    plt.savefig('imgs/vae/0_error.png')
    plt.close()


def load_3d_data():
    """ Read the 3d points from h3m return the train and test points """
    actions = data_utils.define_actions(ENV.FLAGS.action)

    # Load camera parameters
    subjects_ids = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(ENV.FLAGS.cameras_path, subjects_ids)

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, \
    data_mean_3d, data_std_3d, \
    dim_to_ignore_3d, dim_to_use_3d, \
    _, _ = data_utils.read_3d_data(actions,
                                ENV.FLAGS.data_dir,
                                ENV.FLAGS.camera_frame,
                                rcams,
                                ENV.FLAGS.predict_14)

    # Join all the data over all the actions
    train_set_3d = np.concatenate([item for item in train_set_3d.values()], axis=0).astype('float32')
    test_set_3d = np.concatenate([item for item in test_set_3d.values()], axis=0).astype('float32')

    # Join Train an test and suffle it
    all_set = np.concatenate([train_set_3d, test_set_3d], axis=0)

    max_factor = np.max(np.abs(all_set)) if ENV.FLAGS.apply_tanh else 1.0
    all_set /= max_factor

    print('mean:', np.mean(all_set))
    print('std:', np.std(all_set))
    print('max:', np.max(all_set))
    print('min:', np.min(all_set))

    np.random.shuffle(all_set)
    # 400,000 samples for train and the remaining samples for train
    train_set_3d = all_set[:400000, :]
    test_set_3d = all_set[400000:, :]

    Metadata = namedtuple("Metadata", "mean std dim_ignored dim_used max_factor")
    meta = Metadata(data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, max_factor)
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
    """ Dataset """
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
