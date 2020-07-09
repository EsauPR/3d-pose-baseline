""" FLAGS Shared """

import os
import argparse


class __ENV():

    def __init__(self):
        self.FLAGS = None
        self.TRAIN_DIR = None
        self.SUMMARIES_DIR = None

        parser = argparse.ArgumentParser()

        parser.add_argument("--learning_rate", type=float,
                            default=0.001, help="Learning rate")
        parser.add_argument("--dropout", type=float, default=1,
                            help="Dropout keep probability. 1 means no dropout")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="Batch size to use during training")
        parser.add_argument("--epochs", type=int, default=200,
                            help="How many epochs we should train for")
        parser.add_argument("--camera_frame", action='store_true', default=False,
                            help="Convert 3d poses to camera coordinates")
        parser.add_argument("--batch_norm", action='store_true', default=False,
                            help="Use batch_normalization")

        # Data loading
        parser.add_argument("--noised_sample", action='store_true',
                            default=False, help="Generate samples with the noise used to train")
        parser.add_argument("--predict_14", action='store_true',
                            default=False, help="predict 14 joints")
        parser.add_argument("--use_sh", action='store_true', default=False,
                            help="Use 2d pose predictions from StackedHourglass")
        parser.add_argument("--action", type=str, default="All",
                            help="The action to train on. 'All' means all the actions")
        parser.add_argument("--optimizer", type=str, default="adam",
                            help="Optimizer to use [adam, rmsprop]")

        # Architecture
        parser.add_argument("--model", type=str, default='VAE',
                            help="Model to use: [VAE], default VAE.")
        parser.add_argument("--vae_dim", type=int, nargs='+', default=[200, 10],
                            help="Dimensions for hidden layer and latent space, default [24, 12]")


        # Training
        parser.add_argument("--step_log", type=int, default=1000,
                            help="std for noise to add to truth 3d points")
        parser.add_argument("--noise_3d", type=float, nargs='+', default=[1, 1],
                            help="std for noise to add to truth 3d points")
        parser.add_argument("--f_loss", type=str, default="mse",
                            help="Loss use in ELBO function [mae, mse]")
        parser.add_argument("--apply_tanh", action='store_true', default=False,
                            help="Apply sigmoid activation to the decoder output")
        parser.add_argument("--likelihood_factor", type=float, default=10,
                            help="Term to regularize the likelihood loss for ELBO")


        # Directories
        parser.add_argument("--cameras_path", type=str, default="data/h36m/cameras.h5",
                            help="Directory to load camera parameters")
        parser.add_argument("--data_dir", type=str,
                            default="data/h36m/", help="Data directory")
        parser.add_argument("--train_dir", type=str,
                            default="experiments", help="Training directory.")

        parser.add_argument("--sample", action='store_true', default=False,
                            help="Set to True for sampling.")
        parser.add_argument("--load", type=int, default=0,
                            help="Try to load a previous checkpoint.")

        parser.add_argument("--verbose", type=int, default=2,
                            help="0:Error, 1:Warning, 2:INFO*(default), 3:debug")

        self.__parser = parser


    def setup(self):
        """ Set the FLAGS """
        self.FLAGS = self.__parser.parse_args()

        self.TRAIN_DIR = os.path.join(self.FLAGS.train_dir,
                                      self.FLAGS.action,
                                      '{0}_top_model'.format(self.FLAGS.model),
                                      'dropout_{0}'.format(self.FLAGS.dropout),
                                      'epochs_{0}'.format(self.FLAGS.epochs) if self.FLAGS.epochs > 0 else '',
                                      'lr_{0}'.format(self.FLAGS.learning_rate),
                                      'batch_size_{0}'.format(self.FLAGS.batch_size))


        print(self.TRAIN_DIR)
        self.SUMMARIES_DIR = os.path.join( self.TRAIN_DIR, "log" ) # Directory for TB summaries

        # To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
        os.system('mkdir -p {}'.format(self.SUMMARIES_DIR))
        print(self.FLAGS)


ENVIRON = __ENV()
