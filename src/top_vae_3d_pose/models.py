""" Define the models to use """
import tensorflow as tf
from tensorflow import keras

from top_vae_3d_pose.args_def import ENVIRON as ENV


class VAE(tf.keras.Model):
    """ Variational autoencoder """
    def __init__(self, input_size, latent_dim, inter_dim):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self._build_encoder()
        self._buil_decoder()


    def _build_encoder(self):
        enc_in = keras.Input(shape=(self.input_size,), name='enc_input')
        enc_h1 = keras.layers.Dense(units=ENV.FLAGS.vae_dim[0],
                                    activation='relu',
                                    name='encoding')(enc_in)
        # mean and variance for the latent space
        mean = keras.layers.Dense(units=ENV.FLAGS.vae_dim[1], name='mean')(enc_h1)
        log_var = keras.layers.Dense(units=ENV.FLAGS.vae_dim[1], name='log_varianze')(enc_h1)

        self.encoder = keras.Model(inputs=enc_in, outputs=[mean, log_var])
        self.encoder.summary()


    def _buil_decoder(self):
        dec_in = keras.Input(shape=(ENV.FLAGS.vae_dim[1]), name='dec_input')
        dec_f = keras.layers.Dense(units=ENV.FLAGS.vae_dim[0],
                                   activation='relu',
                                   name='dec_f')(dec_in)
        dec_out = keras.layers.Dense(units=self.input_size)(dec_f)

        self.decoder = keras.Model(inputs=dec_in, outputs=dec_out, name='decoder')
        self.decoder.summary()


    @tf.function
    def gen_sample(self, apply_sigmoid=False):
        """ Generate a sample from z distribution """
        eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=apply_sigmoid)


    def encode(self, x):
        """ Encode a x and returns the mean and varianza for Q(z|x) """
        mean, var_log = self.encoder(x)
        return mean, var_log


    def reparametrize(self, mean, log_var):
        """ Reparametrize to obtain z """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * 0.5) + mean


    def decode(self, z, apply_sigmoid=False):
        """ Decode a z sample """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
