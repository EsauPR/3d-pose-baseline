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


def to_world(points_3d, key3d, root_pos):
    """ Trasform coordenates from camera to world coordenates  """
    _, _, rcams = data_handler.get_data_params()
    n_cams = 4
    n_joints_h36m = 32

    # Add global position back
    points_3d = points_3d + np.tile(root_pos, [1, n_joints_h36m])

    # Load the appropriate camera
    # key3d = data_handler.get_key3d(key2d)
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

    x_in = dataset.x_data[idx, :]
    y_real = dataset.y_data[idx, :]
    y_out = model(x_in.reshape(x_in.shape[0], x_in.shape[1] * x_in.shape[2]), training=False)

    # unnormalize data
    x_in = [data_utils.unNormalizeData(p3d,
                                       dataset.x_metadata.mean,
                                       dataset.x_metadata.std,
                                       dataset.x_metadata.dim_ignored) for p3d in x_in]

    y_real = data_utils.unNormalizeData(y_real,
                                        dataset.y_metadata.mean,
                                        dataset.y_metadata.std,
                                        dataset.y_metadata.dim_ignored)
    y_out = data_utils.unNormalizeData(y_out,
                                       dataset.y_metadata.mean,
                                       dataset.y_metadata.std,
                                       dataset.y_metadata.dim_ignored)

    if ENV.FLAGS.camera_frame:
        keys3d = dataset.mapkeys[idx, :]
        root_pos = dataset.y_metadata.root_positions[idx, :]

        x_in = np.array([to_world(p3d,
                                  keys3d[i],
                                  root_pos[i])
                         for i, p3d in enumerate(x_in)])
        y_real = np.array([to_world(p3d.reshape((1, -1)), keys3d[i],
                                    root_pos[i][-1].reshape((1, 3)))[0]
                           for i, p3d in enumerate(y_real)])
        y_out = np.array([to_world(p3d.reshape((1, -1)), keys3d[i],
                                   root_pos[i][-1].reshape((1, 3)))[0]
                          for i, p3d in enumerate(y_out)])

    # 1080p	= 1,920 x 1,080
    fig = plt.figure(figsize=(19.2, 10.8))

    gs1 = gridspec.GridSpec(5, 6*3) # 5 rows, 18 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 0, 0
    for _ in np.arange(nsamples):
        # Sequence
        for pt3d in x_in[exidx]:
            # Plot 3d gt
            ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
            p3d = pt3d
            viz.show3Dpose(p3d, ax2)
            subplot_idx += 1

        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = y_out[exidx, :]
        viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")
        subplot_idx += 1

        # Plot 3d real
        ax4 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = y_real[exidx, :]
        viz.show3Dpose(p3d, ax4, lcolor="#9b59b6", rcolor="#2ecc71")
        subplot_idx += 1

        exidx = exidx + 1

    file_name = "imgs/vae_concat_seq/%s.png" % datetime.utcnow().isoformat()
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
def train_step_vae(model, x_data, y_data, optimizer):
    """ Define a train step """
    with tf.GradientTape() as tape:
        loss = losses.ELBO.compute_loss(model, x_data, y_data)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train():
    """ Train function """
    data_train, data_test = data_handler.load_dataset_3d_seq(seq_len=3)
    print("Dataset dims")
    print(data_train.x_data.shape, data_train.y_data.shape)
    print(data_test.x_data.shape, data_test.y_data.shape)

    seq_len, size = data_train.x_data[0].shape
    # The Vae model must process the seq as a single concatenate input
    model = models.VAE(seq_len*size,
                       latent_dim=ENV.FLAGS.latent_dim,
                       enc_dim=ENV.FLAGS.enc_dim,
                       dec_dim=ENV.FLAGS.dec_dim)

    optimizer = get_optimizer()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './experiments/vae_concat_seq/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero.")


    print("Trainable weights:", len(model.trainable_weights))

    # Indexes for sampling
    idx = np.random.choice(data_test.x_data.shape[0], 15, replace=False)

    # Logs for errors and losses

    loss_train_history = []
    loss_test_history = []
    pred_error_history = []
    error_34_history = []
    error_3_pred_history = []

    for epoch in range(1, ENV.FLAGS.epochs + 1):
        print("\nStarting epoch:", epoch)

        loss_train = tf.keras.metrics.Mean()
        # start_time = time.time()
        for step, (x_train, y_train) in enumerate(tqdm(data_train, ascii=True)):
            # x_train is a batch of seq of dimentions (batch_size, seq_len, input_size)
            batch_size, seq_len, size = x_train.shape
            x_train = x_train.reshape(batch_size, seq_len * size)
            x_train = data_handler.add_noise(x_train)

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
        pred_error = tf.keras.metrics.Mean()
        error_34 = tf.keras.metrics.Mean()
        error_3_pred = tf.keras.metrics.Mean()
        error_2_pred = tf.keras.metrics.Mean()
        error_1_pred = tf.keras.metrics.Mean()

        for x_test, y_test in tqdm(data_test, ascii=True):
            # x_test is a batch of seq of dimentions (batch_size, seq_len, input_size)
            batch_size, seq_len, size = x_test.shape
            y_test = x_test[:, 2, :]
            x_test3 = x_test[:, 1, :]
            x_test2 = x_test[:, 0, :]
            # x_test1 = x_test[:, 0, :]
            x_test = x_test.reshape(batch_size, seq_len * size)
            loss_test(losses.ELBO.compute_loss(model, x_test, y_test))
            preds = model(x_test, training=False)
            pred_error(losses.ELBO.compute_pred_error(y_test, preds))
            error_34(losses.ELBO.compute_pred_error(x_test3, y_test))
            error_3_pred(losses.ELBO.compute_pred_error(x_test3, preds))
            error_2_pred(losses.ELBO.compute_pred_error(x_test2, preds))
            # error_1_pred(losses.ELBO.compute_pred_error(x_test1, preds))
        loss_test_history.append(loss_test.result())
        pred_error_history.append(pred_error.result())
        error_34_history.append(error_34.result())
        error_3_pred_history.append(error_3_pred.result())

        print('Epoch: {}, Test set ELBO: {}'.format(epoch, loss_test_history[-1]))
        print('Epoch: {}, Error frame 2 vs 3: {}'.format(epoch, error_34_history[-1]))
        print('Epoch: {}, Prediction Error: {}'.format(epoch, pred_error_history[-1]))
        print('Epoch: {}, Error frame 2 vs pred: {}'.format(epoch, error_3_pred_history[-1]))
        print('Epoch: {}, Error frame 1 vs pred: {}'.format(epoch, error_2_pred.result()))
        # print('Epoch: {}, Error frame 1 vs pred: {}'.format(epoch, error_1_pred.result()))

        tf.print('\nSaving samples...')
        gen_sample_img(data_test, model=model, idx=idx)

        # Reset data for next epoch
        data_train.on_epoch_end()
        data_test.on_epoch_end(avoid_suffle=True)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        print("Checkpoint saved: {}".format(save_path))

        data_handler.plot_history([('Train Loss', loss_train_history),
                                   ('Test Loss', loss_test_history)],
                                  xlabel='Epochs',
                                  ylabel='Loss',
                                  fname='loss.png')
        data_handler.plot_history([('Pred error', pred_error_history),
                                   # ('Frame err 4vs5', error_34_history),
                                   ('Frame err 4vsPred', error_3_pred_history)],
                                  xlabel='Epochs',
                                  ylabel='Error',
                                  fname='error.png')

    # Save the weights of the las model and the config use to run and train
    model.save_weights('./experiments/vae_concat_seq/last_model_weights')
    with open('./experiments/vae_concat_seq/train.cfg', 'w') as cfg:
        json.dump(vars(ENV.FLAGS), cfg)

    data_handler.save_history(loss_train_history, 'train_loss.npy')
    data_handler.save_history(loss_test_history, 'test_loss.npy')




def evaluate():
    data2d, data3d = data_handler.load_2d_3d_data(return_raw=True)

    model_2d23d = models.PoseBase()
    # Dummy input for creation for bach normalization weigths
    ainput = np.ones((10, 32), dtype=np.float32)
    model_2d23d(ainput, training=False)
    # Load weights for 2d to 3d prediction
    model_2d23d.load_weights('pretrained_models/4874200_PoseBase/PoseBase')

    # Load VAE Model
    seq_len = 3
    human_3d_size = 48
    model_vae_kin = models.VAE(seq_len*human_3d_size,
                               latent_dim=ENV.FLAGS.latent_dim,
                               enc_dim=ENV.FLAGS.enc_dim,
                               dec_dim=ENV.FLAGS.dec_dim)
    model_vae_kin.load_weights('experiments/vae_concat_seq/last_model_weights')

    error_2d_3d = tf.keras.metrics.Mean()
    error_vae_kin = tf.keras.metrics.Mean()
    noise_log = []

    for key2d in tqdm(data2d.test.keys(), ascii=True):
        err23d = tf.keras.metrics.Mean()
        errvk = tf.keras.metrics.Mean()

        tqdm.write("Subject: {}, action: {}, fname: {}".format(*key2d))

        key3d = data_handler.get_key3d(key2d)

        x_in = data2d.test[key2d]
        x_out = data3d.test[key3d]

        # Make a batch of size x.shape[0] to start the generation of the buffer
        x_in = np.array_split(x_in, x_in.shape[0])
        x_out = np.array_split(x_out, x_out.shape[0])

        buffer = []

        for x_2d, y_3d in tqdm(zip(x_in, x_out), total=len(x_in), ascii=True):
            pred_3d = model_2d23d(x_2d, training=False)

            if len(buffer) == 0:
                # Start the buffer with the same predicion
                buffer = [pred_3d[0] for _ in range(seq_len)]

            buffer.append(pred_3d[0])
            buffer.pop(0)

            # print(pred_3d.shape)
            # print(buffer)
            # print(len(buffer))
            vin = np.array([np.concatenate(buffer)])
            ref_3d = model_vae_kin(vin, training=False)

            # Add the last ref to the buffer
            buffer[-1] = ref_3d[0]

            err1 = losses.ELBO.compute_pred_error(y_3d, pred_3d)
            err2 = losses.ELBO.compute_pred_error(y_3d, ref_3d)
            err23d(err1)
            errvk(err2)
            error_2d_3d(err1)
            error_vae_kin(err2)

            noise_log.append(err1)

        tqdm.write("Err 2d-3d: {}, VAE: {}".format(err23d.result(), errvk.result()))

    print("Pred error 2d to 3d:", error_2d_3d.result())
    print("Pred error vae filter:", error_vae_kin.result())

    print(tf.math.reduce_mean(noise_log))
    print(tf.math.reduce_std(noise_log))
    print(tf.math.reduce_min(noise_log))
    print(tf.math.reduce_max(noise_log))




def main():
    """ Main """
    with tf.device('/device:GPU:%d' % ENV.FLAGS.gpu_device):
        if ENV.FLAGS.evaluate:
            evaluate()
        else:
            train()


if __name__ == "__main__":
    ENV.setup()
    main()
