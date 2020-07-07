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


def gen_sample_img(real_points, noised_points, mean, std, dim_ignored, model=None, idx=None):
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
            pred_points = model.decode(z, apply_sigmoid=True).numpy()
        else:
            pred_points = model.decode(z, apply_sigmoid=False).numpy()

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
        file_name = "imgs/vae/noised_%f.png" % ENV.FLAGS.noise_3d
    else:
        file_name = "imgs/vae/%s.png" % datetime.utcnow().isoformat()
    plt.savefig(file_name)
    print("Saved samples on: %s" % file_name)
    plt.close()


def load_data():
    """ Read the 3d points from h3m return the train and test points """
    Dataset = namedtuple("Dataset", "train test mean std dim_ignored dim_used")
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


    train_set_3d = np.concatenate([item for item in train_set_3d.values()], axis=0).astype('float32')
    test_set_3d = np.concatenate([item for item in test_set_3d.values()], axis=0).astype('float32')

    return Dataset(train_set_3d, test_set_3d,
                   data_mean_3d, data_std_3d,
                   dim_to_ignore_3d, dim_to_use_3d)



def add_noise(data, joint_noise_factor=2.0):
    """ Add noise to the 3d joints
        one joint is select randomly and a bigger noise is added by 'joint_noise_factor'
    """
    nsamples, points_dim = data.shape

    # Add gaussian noise over all the data
    noised_data = data + np.random.randn(nsamples, points_dim) * ENV.FLAGS.noise_3d

    # Select the joint for each sample and add more noise
    joints_idx = np.random.choice(np.arange(points_dim), nsamples)
    noise = np.random.randn(nsamples, 3) * (ENV.FLAGS.noise_3d + joint_noise_factor)
    for i in range(nsamples):
        jid = joints_idx[i] - joints_idx[i] % 3
        noised_data[i][jid] += noise[i][0]
        noised_data[i][jid + 1] += noise[i][1]
        noised_data[i][jid + 2] += noise[i][2]

    return noised_data


def make_dataset(train, test, batch_size=128):
    """ Return a dataset (train and test) iterable for tf train loop
        train will iterate real and noised tuple points
        test will iterate only real points
    """
    # Add noise to the truth 3d points
    x_train = add_noise(train)
    y_train = train

    train = tf.data.Dataset.from_tensor_slices((x_train.astype('float32'), y_train)) \
        .shuffle(x_train.shape[0]) \
        .batch(batch_size)

    test = tf.data.Dataset.from_tensor_slices(test) \
        .shuffle(test.shape[0]).batch(ENV.FLAGS.batch_size)

    return train, test
