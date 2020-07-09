import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from top_vae_3d_pose import data_handler, losses, models
from top_vae_3d_pose.args_def import ENVIRON as ENV


@tf.function
def train_step(model, x_noised, x_truth, optimizer):
    with tf.GradientTape() as tape:
        loss = losses.ELBO.compute_loss(model, x_noised, x_truth)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def get_optimizer():
    if ENV.FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(ENV.FLAGS.learning_rate)
    if ENV.FLAGS.optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(ENV.FLAGS.learning_rate)
    raise Exception('Optimizer not found: %s' % ENV.FLAGS.optimizer)


def train():
    data_train, data_test = data_handler.load_3d_data()
    print("Dataset dims")
    print(data_train.data.shape, data_test.data.shape)

    model = models.VAE(data_train.data.shape[1],
                       latent_dim=ENV.FLAGS.vae_dim[-1],
                       inter_dim=ENV.FLAGS.vae_dim[:-1],
                       apply_tanh=ENV.FLAGS.apply_tanh)
    optimizer = get_optimizer()

    # Indexes for sampling
    idx = np.random.choice(data_test.data.shape[0], 9, replace=False)

    # Logs for errors and losses
    error_pred_history = []
    error_noised_history = []
    loss_train_history = []
    loss_test_history = []

    for epoch in range(1, ENV.FLAGS.epochs + 1):
        print("\nStarting epoch:", epoch)

        loss_train = tf.keras.metrics.Mean()
        start_time = time.time()
        for step, (x_noised, x_truth) in enumerate(data_train):
            step_loss = train_step(model, x_noised, x_truth, optimizer)
            loss_train(step_loss)
            if step % ENV.FLAGS.step_log == 0:
                print("  Training loss at step %d: %.4f" % (step, tf.math.reduce_mean(step_loss)))
                print("  Seen : %s samples" % ((step + 1) * ENV.FLAGS.batch_size))
        end_time = time.time()
        loss_train_history.append(loss_train.result())

        loss_test = tf.keras.metrics.Mean()
        error_pred = tf.keras.metrics.Mean()
        error_noised = tf.keras.metrics.Mean()
        for x_noised, x_truth in data_test:
            loss_test(losses.ELBO.compute_loss(model, x_noised, x_truth))
            err_p, err_n = losses.ELBO.compute_error_real_pred(model, x_noised, x_truth)
            error_pred(err_p)
            error_noised(err_n)
        loss_test_history.append(loss_test.result())
        error_pred_history.append(error_pred.result())
        error_noised_history.append(error_noised.result())

        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, loss_test_history[-1], end_time - start_time))
        tf.print('Error real vs noised:', error_noised_history[-1])
        tf.print('Error real vs pred:', error_pred_history[-1])
        tf.print('\nSaving samples...')
        data_handler.gen_sample_img(data_test.data, data_test.data_noised,
                                    data_test.metadata.mean, data_test.metadata.std,
                                    data_test.metadata.dim_ignored, data_test.metadata.max_factor,
                                    model=model, idx=idx)

        # Reset data for next epoch
        data_train.on_epoch_end()
        data_test.on_epoch_end()

    data_handler.plot_history(loss_train_history, loss_test_history,
                              error_pred_history, error_noised_history)


def main():
    if ENV.FLAGS.noised_sample:
        # Generate a sample of data with noise

        data_train, data_test = data_handler.load_3d_data()
        print("Dataset dims")
        print(data_train.data.shape, data_test.data.shape)

        data_handler.gen_sample_img(data_test.data, data_test.data_noised,
                                    data_test.metadata.mean, data_test.metadata.std,
                                    data_test.metadata.dim_ignored, data_test.metadata.max_factor,
                                    idx=np.random.choice(data_test.data.shape[0], 9, replace=False))
    else:
        train()


if __name__ == "__main__":
    ENV.setup()
    main()
