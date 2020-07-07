""" Losses module """

import tensorflow as tf

from top_vae_3d_pose.args_def import ENVIRON as ENV


class ELBO():
    @classmethod
    def compute_loss(cls, model, x_noised, x_truth):
        """ ELBO """
        n = x_noised.shape[1]
        mean, log_var = model.encode(x_noised)
        z = model.reparametrize(mean, log_var)

        # D(Q(z|x)||P(z))
        dkl_loss = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.square(mean) - 1.0 - log_var, axis=-1)

        if ENV.FLAGS.f_loss == 'xent':
            # P(x|z)
            px_z = model.decode(z, apply_sigmoid=True)
            xent_loss = n * tf.keras.losses.binary_crossentropy(x_truth, px_z)
            return xent_loss + dkl_loss
        elif ENV.FLAGS.f_loss == 'mse':
            # f(z)
            f_z = model.decode(z, apply_sigmoid=False)
            mse_loss = n * tf.keras.losses.mse(x_truth, f_z)
            return mse_loss + dkl_loss
        elif ENV.FLAGS.f_loss == 'mae':
            # f(z)
            f_z = model.decode(z, apply_sigmoid=False)
            mae_loss = n * tf.keras.losses.mae(x_truth, f_z)
            return mae_loss + dkl_loss
        else:
            raise Exception('Invalid Loss Function: %s' % ENV.FLAGS.f_loss)
