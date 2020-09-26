""" Train a VAE model used to filter and enhance 3d points """

import json
import time
from datetime import datetime

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data_utils
import viz
from top_vae_3d_pose import data_handler, losses, models
from top_vae_3d_pose.args_def import ENVIRON as ENV

matplotlib.use('Agg')
# matplotlib.use('TkAgg')


def gen_sample_img(real_points, noised_points, mean, std, dim_ignored, model=None, idx=None):
    """ Plot 3d poses, real, with noise and decode from vae model if a model is provided
        pass 'idx' to select samples otherwise idx will be randomly generated
    """
    # select random samples
    nsamples = 9
    if idx is None:
        idx = np.random.choice(real_points.shape[0], nsamples, replace=False)
    real_points = real_points[idx, :]
    noised_points = noised_points[idx, :]

    # Use the model to generate new samples or use the noised samples provided
    if model is not None:
        z = model.reparametrize(*model.encode(noised_points))
        if ENV.FLAGS.f_loss == 'xent':
            pred_points = model.decode(z).numpy()
        else:
            pred_points = model.decode(z).numpy()


    # unnormalioze data
    real_points = data_utils.unNormalizeData(real_points, mean, std, dim_ignored)
    noised_points = data_utils.unNormalizeData(noised_points, mean, std, dim_ignored)
    if model is not None:
        pred_points = data_utils.unNormalizeData(pred_points, mean, std, dim_ignored)

    # Make athe plot and save the fig
    fig = plt.figure(figsize=(19.2, 10.8))
    # fig = plt.figure(figsize=(10.24, 7.48))
    if model is None:
        gs1 = gridspec.GridSpec(3, 6) # 5 rows, 9 columns
    else:
        gs1 = gridspec.GridSpec(3, 9) # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 1, 1
    for _ in np.arange(nsamples):
        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx-1], projection='3d')
        p3d = real_points[exidx-1, :]
        # print('3D points', p3d)
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

        ax3 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = noised_points[exidx-1, :]
        # print('3D points', p3d)
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

        if model is not None:
            ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
            p3d = pred_points[exidx-1, :]
            # print('3D points', p3d)
            viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

        exidx = exidx + 1
        if model is None:
            subplot_idx = subplot_idx + 2
        else:
            subplot_idx = subplot_idx + 3

    # plt.show()
    if ENV.FLAGS.noised_sample:
        file_name = "imgs/vae/noised_%f_%f.png" % (ENV.FLAGS.noise_3d[0], ENV.FLAGS.noise_3d[1])
    else:
        file_name = "imgs/vae/%s.png" % datetime.utcnow().isoformat()
    plt.savefig(file_name)
    print("Saved samples on: %s" % file_name)
    # plt.show()
    plt.close()


def plot_history(train_loss, test_loss, error_pred, error_noised):
    """ Plot the train vs test loss and prediction error vs noised error through the epochs """
    x_data = np.arange(len(train_loss)) + 1

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, train_loss)
    plt.plot(x_data, test_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["Train loss", "Test loss"])
    plt.savefig('imgs/vae/0_loss.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, error_pred)
    plt.plot(x_data, error_noised)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(["Error pred", "Error noised"])
    plt.savefig('imgs/vae/0_error.png')
    plt.close()


def get_optimizer():
    """ Returns the optimizer required by flags """
    if ENV.FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(ENV.FLAGS.learning_rate)
    if ENV.FLAGS.optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(ENV.FLAGS.learning_rate)
    raise Exception('Optimizer not found: %s' % ENV.FLAGS.optimizer)


@tf.function
def train_step(model, x_noised, x_truth, optimizer):
    """ Define a train step """
    with tf.GradientTape() as tape:
        loss = losses.ELBO.compute_loss(model, x_noised, x_truth)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train():
    """ Train function """
    data_train, data_test = data_handler.load_3d_data()
    print("Dataset dims")
    print(data_train.data.shape, data_test.data.shape)

    model = models.VAE(data_train.data.shape[1],
                       latent_dim=ENV.FLAGS.vae_dim[-1],
                       inter_dim=ENV.FLAGS.vae_dim[:-1])
    optimizer = get_optimizer()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './experiments/vae/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero.")


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

            err_n, err_p = losses.ELBO.compute_pred_error(x_noised, x_truth), \
                           losses.ELBO.compute_pred_error(model(x_noised, training=False), x_truth)

            error_pred(err_p)
            error_noised(err_n)
        loss_test_history.append(loss_test.result())
        error_pred_history.append(error_pred.result())
        error_noised_history.append(error_noised.result())

        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, loss_test_history[-1], end_time - start_time))
        tf.print('Error real vs noised:', error_noised_history[-1])
        tf.print('Error real vs pred:', error_pred_history[-1])
        tf.print('\nSaving samples...')
        gen_sample_img(data_test.data, data_test.data_noised,
                       data_test.metadata.mean, data_test.metadata.std,
                       data_test.metadata.dim_ignored,
                       model=model, idx=idx)

        # Reset data for next epoch
        data_train.on_epoch_end(new_noise=True)
        data_test.on_epoch_end(new_noise=True)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        print("Checkpoint saved: {}".format(save_path))

        plot_history(loss_train_history, loss_test_history,
                     error_pred_history, error_noised_history)

    # Save the weights of the las model and the config use to run and train
    model.save_weights('./experiments/vae/last_model_weights')
    with open('./experiments/vae/train.cfg', 'w') as cfg:
        json.dump(vars(ENV.FLAGS), cfg)



def main():
    if ENV.FLAGS.noised_sample:
        # Generate a sample of data with noise

        data_train, data_test = data_handler.load_3d_data()
        print("Dataset dims")
        print(data_train.data.shape, data_test.data.shape)

        gen_sample_img(data_test.data, data_test.data_noised,
                       data_test.metadata.mean, data_test.metadata.std,
                       data_test.metadata.dim_ignored,
                       idx=np.random.choice(data_test.data.shape[0], 9, replace=False))
    else:
        with tf.device('/device:GPU:2'):
            train()


if __name__ == "__main__":
    ENV.setup()
    main()
