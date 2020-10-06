""" FLAGS Shared """

import os
import argparse
import yaml

BAR_FORMAT = '{l_bar}{bar:30}{r_bar}'


class __ENV():

    def __init__(self):
        self.FLAGS = None
        self.TRAIN_DIR = None
        self.SUMMARIES_DIR = None
        self.BAR_FORMAT = BAR_FORMAT

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
        parser.add_argument("--workers", type=int, default=20, help="Number of workers")

        # Architecture
        parser.add_argument("--model", type=str, default='VAE',
                            help="Model to use: [VAE], default VAE.")
        parser.add_argument("--latent_dim", type=int, default=24,
                            help="VAE latent dimension")
        parser.add_argument("--enc_dim", type=int, nargs='+', default=[40, 32],
                            help="VAE encoder dimension")
        parser.add_argument("--dec_dim", type=int, nargs='+', default=[32, 40],
                            help="VAE decoder dimension")


        # Training
        parser.add_argument("--gpu_device", type=int, default=2, help="GPU device number")
        parser.add_argument("--step_log", type=int, default=1000,
                            help="std for noise to add to truth 3d points")
        parser.add_argument("--noise_3d", type=float, nargs='+', default=[1, 1],
                            help="std for noise to add to truth 3d points")
        parser.add_argument("--f_loss", type=str, default="mse",
                            help="Loss use in ELBO function [mae, mse]")
        parser.add_argument("--likelihood_factor", type=float, default=1.0,
                            help="Term to regularize the likelihood loss for ELBO")
        parser.add_argument("--dkl_factor", type=float, default=1.0,
                            help="Term to regularize the likelihood loss for ELBO")
        parser.add_argument("--kcs_factor", type=float, default=1.0,
                            help="Term to regularize the likelihood loss for ELBO")
        parser.add_argument("--max_norm", action='store_true', default=False,
                            help="Apply maxnorm constraint to the weights")
        parser.add_argument("--residual", action='store_true', default=False,
                            help="Whether to add a residual connection every 2 layers")
        parser.add_argument("--train_all", action='store_true', default=False,
                            help="Train all the nn or only the vae nn")
        parser.add_argument("--evaluate", action='store_true', default=False,
                            help="Evaluate the prediction from 2d to 3d + VAE filter k_inputs")
        parser.add_argument("--use_effb1", action='store_true', default=False,
                            help="Use Efficient Net B1 instead of B0")


        # Directories
        parser.add_argument("--cameras_path", type=str, default="data/h36m/cameras.h5",
                            help="Directory to load camera parameters")
        parser.add_argument("--data_dir", type=str,
                            default="data/h36m/", help="Data directory")
        parser.add_argument("--train_dir", type=str,
                            default="experiments", help="Training directory.")
        parser.add_argument("--human_36m_path", type=str, default="data/human3.6m_downloader",
                            help="Path where the videos and imgs frames are located")
        parser.add_argument("--effnet_outputs_path", type=str, default="data/human36m_effnet.hdf5",
                            help="File path for the outputs of the efficientNet model")
        parser.add_argument("--bones_mapping_dir", type=str, default="src/bones_mapping.yml",
                            help="File that maps the bones arquitecture")
        parser.add_argument("--cfg_file", type=str, default="src/train.yml",
                            help="Config file, the values passed through args are remplaced by the specified in the file")

        parser.add_argument("--sample", action='store_true', default=False,
                            help="Set to True for sampling.")
        parser.add_argument("--load", type=int, default=0,
                            help="Try to load a previous checkpoint.")

        parser.add_argument("--verbose", type=int, default=2,
                            help="0:Error, 1:Warning, 2:INFO*(default), 3:debug")


        self.__parser = parser


    def setup(self, read_from_config=False):
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
        # os.system('mkdir -p {}'.format(self.SUMMARIES_DIR))

        if read_from_config:
            self.reaload_from_config_file()

        print(self.FLAGS)


    def reaload_from_config_file(self):
        """ Read config file to FLAGS """
        with open(self.FLAGS.cfg_file) as fyml:
            cfg = yaml.load(fyml, Loader=yaml.FullLoader)

        for _, data in cfg.items():
            for key, value in data.items():
                self.FLAGS.__setattr__(key, value)


ENVIRON = __ENV()
