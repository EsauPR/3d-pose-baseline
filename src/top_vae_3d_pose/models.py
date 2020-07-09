""" Define the models to use """
import os

import tensorflow as tf
from tensorflow import keras

from top_vae_3d_pose.args_def import ENVIRON as ENV


class VAE(tf.keras.Model):
    """ Variational autoencoder """
    def __init__(self, input_size, latent_dim, inter_dim, apply_tanh=False):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self.apply_tanh = apply_tanh
        self._build_encoder()
        self._buil_decoder()


    def _build_encoder(self):
        enc_in = keras.Input(shape=(self.input_size,), name='enc_input')
        enc_h1 = enc_in
        for units in self.inter_dim:
            enc_h1 = keras.layers.Dense(units=units,
                                        activation='relu',
                                        name='enc_h_%d' % units)(enc_h1)
        # mean and variance for the latent space
        mean = keras.layers.Dense(units=self.latent_dim, name='mean')(enc_h1)
        log_var = keras.layers.Dense(units=self.latent_dim, name='log_varianze')(enc_h1)

        self.encoder = keras.Model(inputs=enc_in, outputs=[mean, log_var])
        self.encoder.summary()


    def _buil_decoder(self):
        dec_in = keras.Input(shape=(self.latent_dim), name='dec_input')
        dec_f = dec_in
        for units in self.inter_dim[::-1]:
            dec_f = keras.layers.Dense(units=units,
                                       activation='relu',
                                       name='dec_f_%d' % units)(dec_f)
        if self.apply_tanh:
            dec_out = keras.layers.Dense(units=self.input_size,
                                        activation='tanh',
                                        name='dec_f_out'
                                        )(dec_f)
        else:
            dec_out = keras.layers.Dense(units=self.input_size,
                                        name='dec_f_out'
                                        )(dec_f)

        self.decoder = keras.Model(inputs=dec_in, outputs=dec_out, name='decoder')
        self.decoder.summary()


    # @tf.function
    # def gen_sample(self):
    #     """ Generate a sample from z distribution """
    #     eps = tf.random.normal(shape=(100, self.latent_dim))
    #     return self.decode(eps)


    def encode(self, x):
        """ Encode a x and returns the mean and varianza for Q(z|x) """
        mean, var_log = self.encoder(x)
        return mean, var_log


    def reparametrize(self, mean, log_var):
        """ Reparametrize to obtain z """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * 0.5) + mean


    def decode(self, z):
        """ Decode a z sample """
        return self.decoder(z)


    def call(self, inputs):
        return self.decode(self.reparametrize(*self.encode(inputs)))


    def get_config(self):
        return {
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'inter_dim': self.inter_dim,
            'apply_tanh': self.apply_tanh,
        }
