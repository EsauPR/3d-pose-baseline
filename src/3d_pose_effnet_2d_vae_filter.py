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
    key3d = data_handler.get_key3d(key2d[:3])
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

    keys2d = dataset.mapkeys[idx, :]
    img_frames = data_handler.load_frames_from_keys(keys2d, efficientnet_preprocess=True)

    points_2d = dataset.x_data[idx, :]
    points_3d = dataset.y_data[idx, :]
    out_3d, out_3d_vae = model(points_2d, frame_inputs=img_frames, training=False)

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

    file_name = "imgs/3d_effnet_2d_vae/%s.png" % datetime.utcnow().isoformat()
    plt.savefig(file_name)
    print("Saved samples on: %s" % file_name)
    # plt.show()
    plt.close()


def get_optimizer():
    """ Returns the optimizer required by flags """
    if ENV.FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(ENV.FLAGS.learning_rate)
    if ENV.FLAGS.optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(ENV.FLAGS.learning_rate)
    raise Exception('Optimizer not found: %s' % ENV.FLAGS.optimizer)


@tf.function
def train_step_vae(model, x_data, y_data, images_frames, optimizer):
    """ Define a train step """
    with tf.GradientTape() as tape:
        x_out_3d = model.pose3d(x_data, training=ENV.FLAGS.train_all)
        x_out_img = model.effnet(images_frames, training=ENV.FLAGS.train_all)
        x_out = model.concat([x_data, x_out_3d, x_out_img])

        loss = losses.ELBO.compute_loss(model.vae, x_out, y_data)
    gradients = tape.gradient(loss, model.vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.vae.trainable_variables))
    return loss


def train():
    """ Train function """
    data_train, data_test = data_handler.load_2d_3d_data(key2d_with_frame=True)
    print("Dataset dims")
    print(data_train.x_data.shape, data_train.y_data.shape)
    print(data_test.x_data.shape, data_test.y_data.shape)

    model = models.Pose3DVae(latent_dim=ENV.FLAGS.latent_dim,
                             enc_dim=ENV.FLAGS.enc_dim,
                             dec_dim=ENV.FLAGS.dec_dim,
                             efficient_net=0,
                             use_2d=True)
    # Dummy input for creation for bach normalization weigths
    ainput = np.ones((10, 32), dtype=np.float32)
    model.pose3d(ainput, training=False)
    # Load weights for 2d to 3d prediction
    model.pose3d.load_weights('pretrained_models/4874200_PoseBase/PoseBase')

    optimizer = get_optimizer()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './experiments/3d_effnet_2d_vae/tf_ckpts', max_to_keep=10)
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
        model.effnet.trainable = False
        print("Training only VAE weights:", len(model.trainable_weights))


    # Indexes for sampling
    idx = np.random.choice(data_test.x_data.shape[0], 15, replace=False)

    # Logs for errors and losses
    error_vae_out_history = []
    error_3d_out_history = []
    loss_train_history = []
    loss_test_history = []

    for epoch in range(1, ENV.FLAGS.epochs + 1):
        if epoch > 5:
            data_train.batch_size = ENV.FLAGS.batch_size + 600
        if epoch > 10:
            data_train.batch_size = ENV.FLAGS.batch_size * 3

        print("\nStarting epoch:", epoch)

        loss_train = tf.keras.metrics.Mean()
        # start_time = time.time()
        for step, (x_train, y_train) in enumerate(tqdm(data_train, ascii=True)):
            current_keys2d = data_train.mapkeys[data_train.batch_idx, :]
            images_frames = data_handler.load_frames_from_keys(current_keys2d,
                                                               efficientnet_preprocess=True)

            step_loss = train_step_vae(model, x_train, y_train, images_frames, optimizer)
            loss_train(step_loss)
            if step % ENV.FLAGS.step_log == 0:
                ltp = tf.math.reduce_mean(step_loss)
                tqdm.write(" Training loss at step %d: %.4f" % (step, ltp))
                tqdm.write(" Seen : %s samples" % ((step + 1) * ENV.FLAGS.batch_size))

                ckpt.step.assign_add(1)
                save_path = manager.save()
                tqdm.write("Checkpoint saved: {}".format(save_path))

        # end_time = time.time()
        loss_train_history.append(loss_train.result())

        print("Evaluation on Test data...")
        loss_test = tf.keras.metrics.Mean()
        error_vae_out = tf.keras.metrics.Mean()
        error_3d_out = tf.keras.metrics.Mean()
        for x_test, y_test in tqdm(data_test, ascii=True):
            current_keys2d = data_test.mapkeys[data_test.batch_idx, :]
            images_frames = data_handler.load_frames_from_keys(current_keys2d,
                                                               efficientnet_preprocess=True)
            x_out_3d = model.pose3d(x_test, training=False)
            x_out_img = model.effnet(images_frames, training=False)
            x_out = model.concat([x_test, x_out_3d, x_out_img])
            vae_out = model.vae(x_out, training=False)

            loss_test(losses.ELBO.compute_loss(model.vae, x_out, y_test))

            err_3d, err_vae = losses.ELBO.compute_loss_3d_vs_vae(y_test, x_out_3d, vae_out)
            error_3d_out(err_3d)
            error_vae_out(err_vae)

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

        data_handler.plot_history([('Train Loss', loss_train_history),
                                   ('Test Loss', loss_test_history)],
                                  xlabel='Epochs',
                                  ylabel='Loss',
                                  fname='loss.png')
        data_handler.plot_history([('Pred error', error_vae_out_history),
                                   # ('Frame err 4vs5', error_34_history),
                                   ('2d 3d error', error_3d_out_history)],
                                  xlabel='Epochs',
                                  ylabel='Error',
                                  fname='error.png')

        # Save the weights of the las model and the config use to run and train
        model.save_weights('./experiments/3d_effnet_2d_vae/last_model_weights')
        with open('./experiments/3d_effnet_2d_vae/train.cfg', 'w') as cfg:
            json.dump(vars(ENV.FLAGS), cfg)

    data_handler.save_history(loss_train_history, 'train_loss.npy')
    data_handler.save_history(loss_test_history, 'test_loss.npy')
    data_handler.save_history(error_3d_out_history, 'error_2d3d.npy')
    data_handler.save_history(error_vae_out_history, 'error_vae.npy')



def main():
    """ Main """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    with tf.device('/device:GPU:%d' % ENV.FLAGS.gpu_device):
        train()


if __name__ == "__main__":
    ENV.setup()
    main()
