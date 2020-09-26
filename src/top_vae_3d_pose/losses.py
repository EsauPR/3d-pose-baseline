""" Losses module """

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from top_vae_3d_pose.args_def import ENVIRON as ENV



class ELBO():
    @classmethod
    def compute_loss(cls, model, inputs, targets):
        """ ELBO """
        likef = ENV.FLAGS.likelihood_factor
        kcsf = ENV.FLAGS.kcs_factor
        dklf = ENV.FLAGS.dkl_factor

        # mean and log var from q(theta: z|x)
        mean, log_var = model.encode(inputs)

        # Latent variables from distribution q
        z = model.reparametrize(mean, log_var)
        # sample of P(x|z)
        px_z = model.decode(z)

        # Dkl(Q(z|x)||P(z))
        dkl_loss = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.square(mean) - 1.0 - log_var,
                                       axis=-1)
        like_loss = tf.reduce_mean(tf.square(targets - px_z), axis=-1)
        kcs_error = cls.kcs_error(px_z, targets)

        dkl_loss = dklf * tf.reduce_mean(dkl_loss)
        like_loss = likef * tf.reduce_mean(like_loss)
        kcs_error = kcsf * tf.reduce_mean(kcs_error)

        loss = tf.stack([like_loss, kcs_error, dkl_loss])

        return loss


    @classmethod
    def compute_error_3d_vs_vae(cls, real_out, out_3d, out_vae):
        """ Return the MSE error between the real vs out_3d and real_out vs out_vae """

        loss1 = tf.reduce_mean(tf.square(real_out - out_3d))
        loss2 = tf.reduce_mean(tf.square(real_out - out_vae))
        return loss1, loss2


    @classmethod
    def compute_pred_error(cls, real_out, pred_out):
        """ Compute the MSE error between de predictions and the targets """
        error = tf.reduce_mean(tf.square(real_out - pred_out))
        return error


    @classmethod
    def kcs_error(cls, pred_out, real_out):
        """ Error for the lenght and angles betweeen bones using Kinematic Chain Space

            Paper:

            3D Human Pose Estimation using Spatio-Temporal Networks
            with Explicit Occlusion Training

        """
        # [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56,
        #     57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]
        # Mappgin for the joints that belong to a bone
        # map_i = np.array([1, 2, 3, 1, 7, 8,  1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        # map_j = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        map_i = np.array([1, 2, 3, 1, 5, 6, 1, 8,  9, 10,  9, 12, 13,  9, 15, 16]) - 1
        map_j = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]) - 1

        # Reshape to [num samples, joint, coordinates[x,y,z]]
        pred_out = tf.reshape(pred_out, (pred_out.shape[0], -1, 3))
        real_out = tf.reshape(real_out, (real_out.shape[0], -1, 3))

        # Compute the direction matrix for each sample by the map_i and map_j bones mapping
        # [num sample, direction[x,y,z], bone]
        dirs_pred = []
        dirs_real = []
        for k in range(pred_out.shape[0]):
            bones_pred_dir = []
            bones_real_dir = []
            for i, j in zip(map_i, map_j):
                if i == 0: # Hip, this joint is the origin and this is not predicted
                    bones_pred_dir.append(- pred_out[k][j-1])
                    bones_real_dir.append(- real_out[k][j-1])
                    continue
                bones_pred_dir.append(pred_out[k][i-1] - pred_out[k][j-1])
                bones_real_dir.append(real_out[k][i-1] - real_out[k][j-1])
            dirs_pred.append(tf.stack(bones_pred_dir))
            dirs_real.append(tf.stack(bones_real_dir))
            # print(dirs_pred[-1], flush=True)

        dir_matrix_pred = tf.transpose(tf.stack(dirs_pred), perm=[0, 2, 1])
        dir_matrix_real = tf.transpose(tf.stack(dirs_real), perm=[0, 2, 1])

        # Compute Φ
        # The diagonal elements in Φ indicates the bone length changes, and
        # other elements denote the change of angles between two bones
        phi = tf.linalg.matmul(dir_matrix_pred, dir_matrix_pred, transpose_a=True) - \
              tf.linalg.matmul(dir_matrix_real, dir_matrix_real, transpose_a=True)

        # Return the sum of every length change as error
        return tf.reduce_sum(tf.abs(phi), axis=[2, 1])
