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

matplotlib.use('Agg')
# matplotlib.use('TkAgg')



def to_world(points_3d, key2d, root_pos):
    """ Trasform coordenates from camera to world coordenates  """
    _, _, rcams = data_handler.get_data_params()
    n_cams = 4
    n_joints_h36m = 32

    # Add global position back
    points_3d = points_3d + np.tile(root_pos, [1, n_joints_h36m])

    # Load the appropriate camera
    key3d = data_handler.get_key3d(key2d[:3])
    subj, _, sname = key3d
    subj = int(subj)

    cname = sname.split('.')[1] # <-- camera name
    scams = {(subj, c+1): rcams[(subj, c+1)] for c in range(n_cams)} # cams of this subject
    scam_idx = [scams[(subj, c+1)][-1] for c in range(n_cams)].index(cname) # index of camera used
    the_cam = scams[(subj, scam_idx+1)] # <-- the camera used
    R, T, f, c, k, p, name = the_cam
    assert name == cname

    def cam2world_centered(data_3d_camframe):
        data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
        data_3d_worldframe = data_3d_worldframe.reshape((-1, n_joints_h36m*3))
        # subtract root translation
        return data_3d_worldframe - np.tile(data_3d_worldframe[:, :3], (1, n_joints_h36m))

    # Apply inverse rotation and translation
    return cam2world_centered(points_3d)


def gen_sample_img(dataset, model=None, idx=None):
    """ Plot 3d poses, real, with noise and decode from vae model if a model is provided
        pass 'idx' to select samples otherwise idx will be randomly generated
    """
    # select random samples
    nsamples = 10
    if idx is None:
        idx = np.random.choice(dataset.x_data.shape[0], nsamples, replace=False)

    keys2d = dataset.mapkeys[idx, :]
    img_frames = data_handler.load_frames_from_keys(keys2d, efficientnet_preprocess=True)
    img_frames2 = data_handler.load_frames_from_keys(keys2d, efficientnet_preprocess=False)

    points_2d = dataset.x_data[idx, :]
    points_3d = dataset.y_data[idx, :]
    out_3d, out_3d_vae = model(points_2d, frame_inputs=img_frames, training=False)

    # unnormalioze data
    points_2d = data_utils.unNormalizeData(points_2d,
                                           dataset.x_metadata.mean,
                                           dataset.x_metadata.std,
                                           dataset.x_metadata.dim_ignored)
    points_3d = data_utils.unNormalizeData(points_3d,
                                           dataset.y_metadata.mean,
                                           dataset.y_metadata.std,
                                           dataset.y_metadata.dim_ignored)
    out_3d = data_utils.unNormalizeData(out_3d,
                                        dataset.y_metadata.mean,
                                        dataset.y_metadata.std,
                                        dataset.y_metadata.dim_ignored)
    out_3d_vae = data_utils.unNormalizeData(out_3d_vae,
                                            dataset.y_metadata.mean,
                                            dataset.y_metadata.std,
                                            dataset.y_metadata.dim_ignored)

    if ENV.FLAGS.camera_frame:
        root_pos = dataset.y_metadata.root_positions[idx, :]

        points_3d = np.array([to_world(p3d.reshape((1, -1)),
                                       keys2d[i],
                                       root_pos[i].reshape((1, 3)))[0]
                              for i, p3d in enumerate(points_3d)])
        out_3d = np.array([to_world(p3d.reshape((1, -1)), keys2d[i],
                                    root_pos[i].reshape((1, 3)))[0]
                           for i, p3d in enumerate(out_3d)])
        out_3d_vae = np.array([to_world(p3d.reshape((1, -1)), keys2d[i],
                                        root_pos[i].reshape((1, 3)))[0]
                               for i, p3d in enumerate(out_3d_vae)])

    # 1080p	= 1,920 x 1,080


    for imgi in np.arange(nsamples):
        subplot_idx, exidx = 1, 0
        fig = plt.figure(figsize=(19.2, 5.4))

        nfigs = 3

        gs1 = gridspec.GridSpec(1, nfigs) # 5 rows, 9 columns
        gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
        plt.axis('off')

        ax5 = plt.subplot(gs1[subplot_idx-1])
        plt.imshow(img_frames2[imgi])
        ax5.axis('off')

        ax5.title.set_text('Imagen')
        ax5.title.set_size(24)


        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx])
        p2d = points_2d[imgi, :]
        viz.show2Dpose(p2d, ax1)
        ax1.invert_yaxis()
        ax1.title.set_text('Predicción Pose 2D')
        ax1.title.set_size(24)
        ax1.axis('off')

        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        p3d = out_3d[imgi, :]
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

        ax3.title.set_text('Predicción Pose 3D')
        ax3.title.set_size(24)

        # # Plot 3d predictions + vae
        # ax4 = plt.subplot(gs1[subplot_idx+2], projection='3d')
        # p3d = out_3d_vae[exidx, :]
        # viz.show3Dpose(p3d, ax4, lcolor="#9b59b6", rcolor="#2ecc71")
        # ax4.title.set_text('Predicción Pose 3D')

        # # Plot 3d gt
        # ax2 = plt.subplot(gs1[subplot_idx+2], projection='3d')
        # p3d = points_3d[exidx, :]
        # viz.show3Dpose(p3d, ax2)


        file_name = "imgs/3d_effnet_vae/test/test_%d.png" % imgi
        plt.savefig(file_name)
        print("Saved samples on: %s" % file_name)
        # plt.show()
        plt.close()




        subplot_idx, exidx = 1, 0
        fig = plt.figure(figsize=(19.2, 5.4))

        nfigs = 3

        gs1 = gridspec.GridSpec(1, nfigs+1) # 5 rows, 9 columns
        gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
        plt.axis('off')

        ax5 = plt.subplot(gs1[subplot_idx-1])
        plt.imshow(img_frames2[imgi])
        ax5.axis('off')
        ax5.title.set_text('Imagen')
        ax5.title.set_size(24)
        ax5.title.set_position([.5, 1.061])


        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx])
        p2d = points_2d[imgi, :]
        viz.show2Dpose(p2d, ax1)
        ax1.invert_yaxis()
        ax1.title.set_text('Predicción Pose 2D')
        ax1.title.set_size(24)
        ax1.axis('off')

        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        p3d = out_3d[imgi, :]
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")
        ax3.title.set_text('Predicción Pose 3D')
        ax3.title.set_size(24)

        # Plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx+2], projection='3d')
        p3d = points_3d[imgi, :]
        viz.show3Dpose(p3d, ax2)
        ax2.title.set_text('GT')
        ax2.title.set_size(24)


        file_name = "imgs/3d_effnet_vae/test/test_gt%d.png" % imgi
        plt.savefig(file_name)
        print("Saved samples on: %s" % file_name)
        # plt.show()
        plt.close()


def train():
    """ Train function """
    data_train, data_test = data_handler.load_2d_3d_data(key2d_with_frame=True)

    model = models.Pose3DVae(latent_dim=ENV.FLAGS.latent_dim,
                             enc_dim=ENV.FLAGS.enc_dim,
                             dec_dim=ENV.FLAGS.dec_dim,
                             efficient_net=0)
    # Dummy input for creation for bach normalization weigths
    ainput = np.ones((10, 32), dtype=np.float32)
    model.pose3d(ainput, training=False)
    # Load weights for 2d to 3d prediction
    # model.load_weights('./experiments/3d_effnet_vae/last_model_weights')

    # optimizer = get_optimizer()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, './experiments/3d_effnet_vae/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)


    # Indexes for sampling
    idx = np.random.choice(data_test.x_data.shape[0], 15, replace=False)

    gen_sample_img(data_test, model=model, idx=idx)



def main():
    """ Main """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    with tf.device('/device:GPU:%d' % ENV.FLAGS.gpu_device):
        train()


if __name__ == "__main__":
    ENV.setup()
    main()
