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
    dataset = data_handler.load_data()
    print("Dataset dims")
    print(dataset.train.shape, dataset.test.shape)

    # Data for traning flow
    data, data_test = data_handler.make_dataset(dataset.train, dataset.test, batch_size=ENV.FLAGS.batch_size)
    # Test noised for test and sampling
    test_noised = data_handler.add_noise(dataset.test)

    model = models.VAE(dataset.train.shape[1], latent_dim=ENV.FLAGS.vae_dim[1], inter_dim=ENV.FLAGS.vae_dim[0])
    optimizer = get_optimizer()

    # Indexes for sampling
    idx = np.random.choice(dataset.test.shape[0], 9, replace=False)

    for epoch in range(1, ENV.FLAGS.epochs + 1):
        print("\nStarting epoch:", epoch)
        start_time = time.time()
        for step, (x_noised, x_truth) in enumerate(data):
            step_loss = train_step(model, x_noised, x_truth, optimizer)
            if step % ENV.FLAGS.step_log == 0:
                print("  Training loss at step %d: %.4f" % (step, -tf.math.reduce_mean(step_loss)))
                print("  Seen : %s samples" % ((step + 1) * ENV.FLAGS.batch_size))
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for x_test in data_test:
            loss(losses.ELBO.compute_loss(model, x_test, x_test))
        elbo = loss.result()

        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))
        print("\nSaving samples...")
        data_handler.gen_sample_img(dataset.test, test_noised,
                                    dataset.mean, dataset.std, dataset.dim_ignored,
                                    model=model, idx=idx)



def sample():
    pass


def main():
    if ENV.FLAGS.sample:
        sample()
    elif ENV.FLAGS.noised_sample:
        # Generate a sample of data with noise
        dataset = data_handler.load_data()
        print("Dataset dims")
        print(dataset.train.shape, dataset.test.shape)
        idx = np.random.choice(dataset.train.shape[0], 9, replace=False)
        print(idx)
        real_p = dataset.train[idx, :]
        noised_p = data_handler.add_noise(real_p)
        data_handler.gen_sample_img(real_p, noised_p, dataset.mean, dataset.std, dataset.dim_ignored)
    else:
        train()


if __name__ == "__main__":
    ENV.setup()
    main()
