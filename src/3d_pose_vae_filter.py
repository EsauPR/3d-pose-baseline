""" Train a VAE model used to filter and enhance 3d points """

import json
from datetime import datetime

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import cameras
import data_utils
import viz
from top_vae_3d_pose import data_handler, losses, models
from top_vae_3d_pose.args_def import ENVIRON as ENV

matplotlib.use('Agg')
# matplotlib.use('TkAgg')

# tf.debugging.set_log_device_placement(True)


def to_world(points_3d, key2d, root_pos):
    """ Trasform coordenates from camera to world coordenates  """
    _, _, rcams = data_handler.get_data_params()
    n_cams = 4
    n_joints_h36m = 32

    # Add global position back
    points_3d = points_3d + np.tile(root_pos, [1, n_joints_h36m])

    # Load the appropriate camera
    key3d = data_handler.get_key3d(key2d)
    subj, _, sname = key3d
    subj = int(subj)

    cname = sname.split('.')[1] # <-- camera name
    scams = {(subj, c+1): rcams[(subj, c+1)] for c in range(n_cams)} # cams of this subject
    scam_idx = [scams[(subj, c+1)][-1] for c in range(n_cams)].index(cname) # index of camera used
    the_cam = scams[(subj, scam_idx+1)] # <-- the camera used
    R, T, f, c, k, p, name = the_cam
    assert name == cname

    def cam2world_centered(data_3d_camframe):
        data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
        data_3d_worldframe = data_3d_worldframe.reshape((-1, n_joints_h36m*3))
        # subtract root translation
        return data_3d_worldframe - np.tile(data_3d_worldframe[:, :3], (1, n_joints_h36m))

    # Apply inverse rotation and translation
    return cam2world_centered(points_3d)


def gen_sample_img(dataset, model=None, idx=None):
    """ Plot 3d poses, real, with noise and decode from vae model if a model is provided
        pass 'idx' to select samples otherwise idx will be randomly generated
    """
    # select random samples
    nsamples = 15
    if idx is None:
        idx = np.random.choice(dataset.x_data.shape[0], nsamples, replace=False)

    points_2d = dataset.x_data[idx, :]
    points_3d = dataset.y_data[idx, :]
    out_3d, out_3d_vae = model(points_2d, training=False)

    # unnormalioze data
    points_2d = data_utils.unNormalizeData(points_2d,
                                           dataset.x_metadata.mean,
                                           dataset.x_metadata.std,
                                           dataset.x_metadata.dim_ignored)
    points_3d = data_utils.unNormalizeData(points_3d,
                                           dataset.y_metadata.mean,
                                           dataset.y_metadata.std,
                                           dataset.y_metadata.dim_ignored)
    out_3d = data_utils.unNormalizeData(out_3d,
                                        dataset.y_metadata.mean,
                                        dataset.y_metadata.std,
                                        dataset.y_metadata.dim_ignored)
    out_3d_vae = data_utils.unNormalizeData(out_3d_vae,
                                            dataset.y_metadata.mean,
                                            dataset.y_metadata.std,
                                            dataset.y_metadata.dim_ignored)

    if ENV.FLAGS.camera_frame:
        keys2d = dataset.mapkeys[idx, :]
        root_pos = dataset.y_metadata.root_positions[idx, :]

        points_3d = np.array([to_world(p3d.reshape((1, -1)),
                                       keys2d[i],
                                       root_pos[i].reshape((1, 3)))[0]
                              for i, p3d in enumerate(points_3d)])
        out_3d = np.array([to_world(p3d.reshape((1, -1)), keys2d[i],
                                    root_pos[i].reshape((1, 3)))[0]
                           for i, p3d in enumerate(out_3d)])
        out_3d_vae = np.array([to_world(p3d.reshape((1, -1)), keys2d[i],
                                        root_pos[i].reshape((1, 3)))[0]
                               for i, p3d in enumerate(out_3d_vae)])

    # 1080p	= 1,920 x 1,080
    fig = plt.figure(figsize=(19.2, 10.8))

    gs1 = gridspec.GridSpec(5, 12) # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 1, 0
    for _ in np.arange(nsamples):

        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx-1])
        p2d = points_2d[exidx, :]
        viz.show2Dpose(p2d, ax1)
        ax1.invert_yaxis()

        # Plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = points_3d[exidx, :]
        viz.show3Dpose(p3d, ax2)

        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        p3d = out_3d[exidx, :]
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

        # Plot 3d predictions + vae
        ax4 = plt.subplot(gs1[subplot_idx+2], projection='3d')
        p3d = out_3d_vae[exidx, :]
        viz.show3Dpose(p3d, ax4, lcolor="#9b59b6", rcolor="#2ecc71")

        exidx = exidx + 1
        subplot_idx = subplot_idx + 4

    file_name = "imgs/3d_vae/%s.png" % datetime.utcnow().isoformat()
    plt.savefig(file_name)
    print("Saved samples on: %s" % file_name)
    # plt.show()
    plt.close()


def plot_history(train_loss, test_loss, error_vae_out, error_3d_out):
    """ Plot the train vs test loss and vae out vs 3dpose out error through the epochs """
    x_data = np.arange(len(train_loss)) + 1

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, train_loss)
    plt.plot(x_data, test_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["Train loss", "Test loss"])
    plt.savefig('imgs/3d_vae/0_loss.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, error_vae_out)
    plt.plot(x_data, error_3d_out)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(["Error vae_out", "Error 3d_out"])
    plt.savefig('imgs/3d_vae/0_error.png')
    plt.close()


def save_history(loss_train, loss_test, error_3d_out, error_vae_out):
    """ Save the error and lost history on files """
    with open('./experiments/3d_vae/loss_train.npy', 'wb') as f:
        np.save(f, np.array(loss_train))
    with open('./experiments/3d_vae/loss_test.npy', 'wb') as f:
        np.save(f, np.array(loss_test))
    with open('./experiments/3d_vae/error_3d_out.npy', 'wb') as f:
        np.save(f, np.array(error_3d_out))
    with open('./experiments/3d_vae/error_vae_out.npy', 'wb') as f:
        np.save(f, np.array(error_vae_out))


def get_optimizer():
    """ Returns the optimizer required by flags """
    if ENV.FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(ENV.FLAGS.learning_rate)
    if ENV.FLAGS.optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(ENV.FLAGS.learning_rate)
    raise Exception('Optimizer not found: %s' % ENV.FLAGS.optimizer)


@tf.function
def train_step_vae(model, x_data, y_data, optimizer):
    """ Define a train step """
    with tf.GradientTape() as tape:
        x_out = model.pose3d(x_data, training=False)
        loss = losses.ELBO.compute_loss(model.vae, x_out, y_data)
    gradients = tape.gradient(loss, model.vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.vae.trainable_variables))
    return loss


def train():
    """ Train function """
    data_train, data_test = data_handler.load_2d_3d_data()
    print("Dataset dims")
    print(data_train.x_data.shape, data_train.y_data.shape)
    print(data_test.x_data.shape, data_test.y_data.shape)

    model = models.Pose3DVae(latent_dim=ENV.FLAGS.latent_dim,
                             enc_dim=ENV.FLAGS.enc_dim,
                             dec_dim=ENV.FLAGS.dec_dim)
    # Dummy input for creation for bach normalization weigths
    ainput = np.ones((10, 32), dtype=np.float32)
    model(ainput, training=False)
    # Load weights for 2d to 3d prediction
    model.pose3d.load_weights('pretrained_models/4874200_PoseBase/PoseBase')

    optimizer = get_optimizer()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './experiments/3d_vae/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero.")


    print("Trainable weights:", len(model.trainable_weights))
    if ENV.FLAGS.train_all:
        print("Training all")
    else:
        model.pose3d.trainable = False
        print("Training only VAE weights:", len(model.trainable_weights))


    # Indexes for sampling
    idx = np.random.choice(data_test.x_data.shape[0], 15, replace=False)

    # Logs for errors and losses
    error_vae_out_history = []
    error_3d_out_history = []
    loss_train_history = []
    loss_test_history = []

    for epoch in range(1, ENV.FLAGS.epochs + 1):
        print("\nStarting epoch:", epoch)

        loss_train = tf.keras.metrics.Mean()
        # start_time = time.time()
        for step, (x_train, y_train) in enumerate(tqdm(data_train, ascii=True)):
            step_loss = train_step_vae(model, x_train, y_train, optimizer)
            loss_train(step_loss)
            if step % ENV.FLAGS.step_log == 0:
                ltp = tf.math.reduce_mean(step_loss)
                tqdm.write(" Training loss at step %d: %.4f" % (step, ltp))
                tqdm.write(" Seen : %s samples" % ((step + 1) * ENV.FLAGS.batch_size))
        # end_time = time.time()
        loss_train_history.append(loss_train.result())

        print("Evaluation on Test data...")
        loss_test = tf.keras.metrics.Mean()
        error_vae_out = tf.keras.metrics.Mean()
        error_3d_out = tf.keras.metrics.Mean()
        for x_test, y_test in tqdm(data_test, ascii=True):
            x_out_3d = model.pose3d(x_test, training=False)
            vae_out = model.vae(x_out_3d, training=False)

            loss_test(losses.ELBO.compute_loss(model.vae, x_out_3d, y_test))

            err_3d, err_vae = losses.ELBO.compute_loss_3d_vs_vae(y_test, x_out_3d, vae_out)
            error_vae_out(err_vae)
            error_3d_out(err_3d)
        loss_test_history.append(loss_test.result())
        error_vae_out_history.append(error_vae_out.result())
        error_3d_out_history.append(error_3d_out.result())
        print('Epoch: {}, Test set ELBO: {}'.format(epoch, loss_test_history[-1]))
        tf.print('Error real vs 3d out:', error_3d_out_history[-1])
        tf.print('Error real vs vae out:', error_vae_out_history[-1])
        tf.print('\nSaving samples...')
        gen_sample_img(data_test, model=model, idx=idx)

        # Reset data for next epoch
        data_train.on_epoch_end()
        data_test.on_epoch_end(avoid_suffle=True)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        print("Checkpoint saved: {}".format(save_path))

        plot_history(loss_train_history, loss_test_history,
                     error_vae_out_history, error_3d_out_history)

    # Save the weights of the las model and the config use to run and train
    model.save_weights('./experiments/3d_vae/last_model_weights')
    with open('./experiments/3d_vae/train.cfg', 'w') as cfg:
        json.dump(vars(ENV.FLAGS), cfg)

    save_history(loss_train_history, loss_test_history, error_3d_out_history, error_vae_out_history)



def main():
    """ Main """
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    try:
       # Disable first GPU
       tf.config.set_visible_devices(physical_devices[2], 'GPU')
       logical_devices = tf.config.list_logical_devices('GPU')
       # Logical device was not created for first GPU
       assert len(logical_devices) == 1
    except:
       print("nel")

    print(tf.config.list_logical_devices('GPU'))
    train()


    # with tf.device('/device:GPU:%d' % ENV.FLAGS.gpu_device):
    #     train()


if __name__ == "__main__":
    ENV.setup()
    main()
