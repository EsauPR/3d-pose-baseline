""" Losses module """

import numpy as np
import tensorflow as tf

from top_vae_3d_pose.args_def import ENVIRON as ENV


class ELBO():
    @classmethod
    def compute_loss(cls, model, x_noised, x_truth):
        """ ELBO """
        n = x_noised.shape[1] * ENV.FLAGS.likelihood_factor

        # mean and log var from q(theta: z|x)
        mean, log_var = model.encode(x_noised)
        # Latent variables from distribution q
        z = model.reparametrize(mean, log_var)

        # Dkl(Q(z|x)||P(z))
        dkl_loss = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.square(mean) - 1.0 - log_var,
                                       axis=-1)

        # sample of P(x|z)
        px_z = model.decode(z)

        # if ENV.FLAGS.f_loss == 'xent':
        #     xent_loss = n * tf.keras.losses.binary_crossentropy(x_truth, px_z)
        #     return xent_loss + dkl_loss

        if ENV.FLAGS.f_loss == 'mse':
            mse_loss = n * tf.keras.losses.mse(x_truth, px_z)
            return mse_loss + dkl_loss

        if ENV.FLAGS.f_loss == 'mae':
            mae_loss = n * tf.keras.losses.mae(x_truth, px_z)
            return mae_loss + dkl_loss

        raise Exception('Invalid Loss Function: %s' % ENV.FLAGS.f_loss)


    @classmethod
    def compute_error_real_pred(cls, model, x_noised, x_truth, n=None):
        """ Return the error of the noised samples and decoded samples vs real samples """
        if n is None:
            n = x_noised.shape[1]

        mean, log_var = model.encode(x_noised)
        z = model.reparametrize(mean, log_var)

        # sample of P(x|z)
        px_z = model.decode(z)

        # if ENV.FLAGS.f_loss == 'xent':
        #     # Error bettwen real and prediction
        #     xent_loss = n * tf.keras.losses.binary_crossentropy(x_truth, px_z)
        #     # Error bettwen real and noised
        #     xent_loss_noised = n * tf.keras.losses.binary_crossentropy(x_truth, x_noised)
        #     return xent_loss, xent_loss_noised

        if ENV.FLAGS.f_loss == 'mse':
            # Error bettwen real and prediction
            mse_loss = n * tf.keras.losses.mse(x_truth, px_z)
            # Error bettwen real and noised
            mse_loss_noised = n * tf.keras.losses.mse(x_truth, x_noised)
            return mse_loss, mse_loss_noised

        if ENV.FLAGS.f_loss == 'mae':
            # Error bettwen real and prediction
            mae_loss = n * tf.keras.losses.mae(x_truth, px_z)
            # Error bettwen real and noised
            mae_loss_noised = n * tf.keras.losses.mae(x_truth, x_noised)
            return mae_loss, mae_loss_noised

        raise Exception('Invalid Loss Function: %s' % ENV.FLAGS.f_loss)


    @classmethod
    def compute_loss_3d_vs_vae(cls, real_out, out_3d, out_vae):
        # real_out = np.array(real_out)
        # tf.print(real_out.shape)
        # tf.print(out_3d.shape)
        # tf.print(out_vae.shape)
        # print("\n", flush=True)
        loss1 = tf.reduce_mean(tf.math.square(real_out - out_3d))
        loss2 = tf.reduce_mean(tf.math.square(real_out - out_vae))
        return loss1, loss2
