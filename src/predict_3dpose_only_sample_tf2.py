"""Predicting 3d poses from 2d joints"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

# from absl import app
# from absl import flags
import argparse

import matplotlib.pyplot as plt
import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import procrustes

import viz
import cameras
import data_utils
# import linear_model

from top_vae_3d_pose import models

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float,
                    default=1.0, help="Learning rate")
parser.add_argument("--dropout", type=float, default=1,
                    help="Dropout keep probability. 1 means no dropout")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size to use during training")
parser.add_argument("--epochs", type=int, default=200,
                    help="How many epochs we should train for")
parser.add_argument("--camera_frame", action='store_true', default=False,
                    help="Convert 3d poses to camera coordinates")
parser.add_argument("--max_norm", action='store_true', default=False,
                    help="Apply maxnorm constraint to the weights")
parser.add_argument("--batch_norm", action='store_true', default=False,
                    help="Use batch_normalization")

# Data loading
parser.add_argument("--predict_14", action='store_true',
                    default=False, help="predict 14 joints")
parser.add_argument("--use_sh", action='store_true', default=False,
                    help="Use 2d pose predictions from StackedHourglass")
parser.add_argument("--action", type=str, default="All",
                    help="The action to train on. 'All' means all the actions")

# Architecture
parser.add_argument("--linear_size", type=int, default=1024,
                    help="Size of each model layer.")
parser.add_argument("--num_layers", type=int, default=2,
                    help="Number of layers in the model.")
parser.add_argument("--residual", action='store_true', default=False,
                    help="Whether to add a residual connection every 2 layers")

# Evaluation
parser.add_argument("--procrustes", action='store_true', default=False,
                    help="Apply procrustes analysis at test time")
parser.add_argument("--evaluateActionWise", action='store_true',
                    default=False,  help="The dataset to use either h36m or heva")

# Directories
parser.add_argument("--cameras_path", type=str, default="data/h36m/cameras.h5",
                    help="Directory to load camera parameters")
parser.add_argument("--data_dir", type=str,
                    default="data/h36m/", help="Data directory")
parser.add_argument("--train_dir", type=str,
                    default="experiments", help="Training directory.")

# openpose
parser.add_argument("--pose_estimation_json", type=str, default="/tmp/",
                    help="pose estimation json output directory, openpose or tf-pose-estimation")
parser.add_argument("--interpolation", action='store_true',
                    default=False, help="interpolate openpose json")
parser.add_argument("--multiplier", type=float, default=0.1,
                    help="interpolation frame range")
parser.add_argument("--write_gif", action='store_true',
                    default=False, help="write final anim gif")
parser.add_argument("--gif_fps", type=int, default=30,
                    help="output gif framerate")
parser.add_argument("--verbose", type=int, default=2,
                    help="0:Error, 1:Warning, 2:INFO*(default), 3:debug")
parser.add_argument("--cache_on_fail", action='store_true', default=True,
                    help="caching last valid frame on invalid frame")

# Train or load
parser.add_argument("--sample", action='store_true', default=False,
                    help="Set to True for sampling.")
parser.add_argument("--use_cpu", action='store_true', default=False,
                    help="Whether to use the CPU")
parser.add_argument("--load", type=int, default=0,
                    help="Try to load a previous checkpoint.")

# Misc
parser.add_argument("--use_fp16", action='store_true', default=False,
                    help="Train using fp16 instead of fp32.")


FLAGS = parser.parse_args()


train_dir = os.path.join(FLAGS.train_dir,
                         FLAGS.action,
                         'dropout_{0}'.format(FLAGS.dropout),
                         'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
                         'lr_{0}'.format(FLAGS.learning_rate),
                         'residual' if FLAGS.residual else 'not_residual',
                         'depth_{0}'.format(FLAGS.num_layers),
                         'linear_size{0}'.format(FLAGS.linear_size),
                         'batch_size_{0}'.format(FLAGS.batch_size),
                         'procrustes' if FLAGS.procrustes else 'no_procrustes',
                         'maxnorm' if FLAGS.max_norm else 'no_maxnorm',
                         'batch_normalization' if FLAGS.batch_norm else 'no_batch_normalization',
                         'use_stacked_hourglass' if FLAGS.use_sh else 'not_stacked_hourglass',
                         'predict_14' if FLAGS.predict_14 else 'predict_17')

print( train_dir )
summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries

# To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(summaries_dir))



def sample():
    """Get samples from a model and visualize them"""

    actions = data_utils.define_actions( FLAGS.action )

    # Load camera parameters
    SUBJECT_IDS = [1,5,6,7,8,9,11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

    if FLAGS.use_sh:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
    else:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )
    print( "done reading and normalizing data." )


    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = 128


    # model = models.Pose3DBase(1024, 2, 0.5, residual=True, batch_norm=True, predict_14=False)
    model = models.PoseBase()
    #dummy input for creation for bach normlaization weigths
    ainput = np.ones((10, 32), dtype=np.float32)
    model(ainput, training=False)

    model.load_weights('pretrained_models/4874200_PoseBase/PoseBase')
    print("Model loaded")

    for key2d in test_set_2d.keys():

        (subj, b, fname) = key2d
        print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

        # keys should be the same if 3d is in camera coordinates
        key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
        key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh')) and FLAGS.camera_frame else key3d

        enc_in  = test_set_2d[ key2d ]
        n2d, _ = enc_in.shape
        dec_out = test_set_3d[ key3d ]
        n3d, _ = dec_out.shape
        assert n2d == n3d

        # Split into about-same-size batches
        enc_in   = np.array_split( enc_in,  n2d // batch_size )
        dec_out  = np.array_split( dec_out, n3d // batch_size )
        all_poses_3d = []

        for bidx in range( len(enc_in) ):

            # Dropout probability 0 (keep probability 1) for sampling
            dp = 1.0
            # _, _, poses3d = model.step(sess, enc_in[bidx], dec_out[bidx], dp, isTraining=False)
            poses3d = model(enc_in[bidx], training=False)

            # denormalize
            enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
            dec_out[bidx] = data_utils.unNormalizeData( dec_out[bidx], data_mean_3d, data_std_3d, dim_to_ignore_3d )
            poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
            all_poses_3d.append( poses3d )

        # Put all the poses together
        enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, all_poses_3d] )

        # Convert back to world coordinates
        if FLAGS.camera_frame:
            N_CAMERAS = 4
            N_JOINTS_H36M = 32

            # Add global position back
            dec_out = dec_out + np.tile( test_root_positions[ key3d ], [1,N_JOINTS_H36M] )

            # Load the appropriate camera
            subj, _, sname = key3d

            cname = sname.split('.')[1] # <-- camera name
            scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
            scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
            the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
            R, T, f, c, k, p, name = the_cam
            assert name == cname

            def cam2world_centered(data_3d_camframe):
                data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
                data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
                # subtract root translation
                return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )

            # Apply inverse rotation and translation
            dec_out = cam2world_centered(dec_out)
            poses3d = cam2world_centered(poses3d)

    print("\ntrainable")
    print(model.trainable_variables)
    print("Non trainable")
    print(model.non_trainable_variables)
    print("\n", flush=True)

    # Grab a random batch to visualize
    enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, poses3d] )
    idx = np.random.permutation( enc_in.shape[0] )
    enc_in, dec_out, poses3d = enc_in[idx, :], dec_out[idx, :], poses3d[idx, :]

    # Visualize random samples
    import matplotlib.gridspec as gridspec

    # 1080p	= 1,920 x 1,080
    fig = plt.figure( figsize=(19.2, 10.8) )

    gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 1, 1
    nsamples = 15
    for i in np.arange( nsamples ):

        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx-1])
        p2d = enc_in[exidx,:]
        viz.show2Dpose( p2d, ax1 )
        ax1.invert_yaxis()

        # Plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = dec_out[exidx,:]
        viz.show3Dpose( p3d, ax2 )

        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        p3d = poses3d[exidx,:]
        viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

        exidx = exidx + 1
        subplot_idx = subplot_idx + 3

    plt.show()


def main():
    if FLAGS.sample:
        sample()
    else:
        train()


if __name__ == "__main__":
    main()
