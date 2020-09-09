""" Define the models to use """

import tensorflow as tf
from tensorflow import keras

import efficientnet.tfkeras as efn


EFFICIENT_NET_INPUT_SHAPE = (224, 224, 3)


class VAE(tf.keras.Model):
    """ Variational autoencoder """
    def __init__(self, input_size, latent_dim, enc_dim, dec_dim, human_3d_size=48):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.human_3d_size = human_3d_size
        self._build_encoder()
        self._buil_decoder()


    def _build_encoder(self):
        enc_in = keras.Input(shape=(self.input_size,), name='enc_input')
        enc_h1 = enc_in
        for units in self.enc_dim:
            enc_h1 = keras.layers.Dense(units=units,
                                        activation='relu',
                                        name='enc_h_%d' % units)(enc_h1)
        # mean and variance for the latent space
        mean = keras.layers.Dense(units=self.latent_dim, name='mean')(enc_h1)
        log_var = keras.layers.Dense(units=self.latent_dim, name='log_varianze')(enc_h1)

        self.encoder = keras.Model(inputs=enc_in, outputs=[mean, log_var])
        self.encoder.summary()


    def _buil_decoder(self):
        dec_in = keras.Input(shape=(self.latent_dim), name='dec_input')
        dec_f = dec_in

        for units in self.dec_dim:
            dec_f = keras.layers.Dense(units=units,
                                       activation='relu',
                                       name='dec_f_%d' % units)(dec_f)

        dec_out = keras.layers.Dense(units=self.human_3d_size, name='dec_f_out')(dec_f)

        self.decoder = keras.Model(inputs=dec_in, outputs=dec_out, name='decoder')
        self.decoder.summary()


    # @tf.function
    # def gen_sample(self):
    #     """ Generate a sample from z distribution """
    #     eps = tf.random.normal(shape=(100, self.latent_dim))
    #     return self.decode(eps)


    def encode(self, x):
        """ Encode a x and returns the mean and varianza for Q(z|x) """
        mean, var_log = self.encoder(x)
        return mean, var_log


    def reparametrize(self, mean, log_var):
        """ Reparametrize to obtain z """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * 0.5) + mean


    def decode(self, z):
        """ Decode a z sample """
        return self.decoder(z)


    def call(self, inputs, training=False):
        return self.decode(self.reparametrize(*self.encode(inputs)))


    def get_config(self):
        return {
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'enc_dim': self.enc_dim,
            'dec_dim': self.dec_dim,
            'human_3d_size': self.human_3d_size,
        }



def kaiming(shape, dtype=tf.float32, partition_info=None):
    """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

    Args
        shape: dimensions of the tf array to initialize
        dtype: data type of the array
        partition_info: (Optional) info about how the variable is partitioned.
        See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
        Needed to be used as an initializer.
    Returns
        Tensorflow array with initial weights
    """
    return tf.random.truncated_normal(shape, dtype=dtype) * tf.math.sqrt(2/float(shape[0]))



class Linear3DBase(tf.keras.layers.Layer):
    """ Class for linear layer by 3d pose baseline """
    def __init__(self,
                 units,
                 input_size,
                 dropout_keep_prob,
                 max_norm=True,
                 batch_norm=True,
                 is_output_layer=False):
        super(Linear3DBase, self).__init__()

        self.dropout_keep_prob = dropout_keep_prob
        self.max_norm = max_norm
        self.batch_norm = batch_norm
        self.is_output_layer = is_output_layer

        with tf.name_scope('linear'):
            with tf.name_scope('w'):
                self.w = tf.Variable(
                    shape=(input_size, units),
                    initial_value=kaiming(shape=(input_size, units)),
                    # initial_value = next(w_gen),
                    trainable=True,
                )
            with tf.name_scope('b'):
                self.b = tf.Variable(
                    shape=(units,),
                    initial_value=kaiming(shape=(units,)),
                    # initial_value = next(w_gen),
                    trainable=True
                )
        if not is_output_layer:
            # gamma = next(w_gen)
            # beta = next(w_gen)
            with tf.name_scope('batch_normalization'):
                self.btn = keras.layers.BatchNormalization(
                    # gamma_initializer=lambda _, dtype=None: gamma,
                    # beta_initializer= lambda _, dtype=None: beta
                )


    def call(self, inputs, training=True):
        if self.max_norm:
            w = tf.clip_by_norm(self.w, 1)
            y3 = tf.matmul(inputs, w) + self.b
        else:
            y3 = tf.matmul(inputs, self.w) + self.b

        if self.is_output_layer:
            return y3

        if self.batch_norm:
            y3 = self.btn(y3, training=training)

        tf.nn.relu(y3)
        if training:
            y3 = tf.nn.dropout(y3, self.dropout_keep_prob)

        return y3



class TwoLinear3DBase(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 input_size,
                 dropout_keep_prob,
                 max_norm=True,
                 batch_norm=True,
                 residual=True):
        super(TwoLinear3DBase, self).__init__()
        self.units = units
        self.dropout_keep_prob = dropout_keep_prob
        self.max_norm = max_norm
        self.residual = residual

        with tf.name_scope('two_linear'):
            self.l1 = Linear3DBase(self.units,
                                   self.units,
                                   self.dropout_keep_prob,
                                   max_norm=self.max_norm,
                                   batch_norm=batch_norm,
                                   is_output_layer=False)
            self.l2 = Linear3DBase(self.units,
                                   self.units,
                                   self.dropout_keep_prob,
                                   max_norm=self.max_norm,
                                   batch_norm=batch_norm,
                                   is_output_layer=False)


    def call(self, inputs, training=True):
        y_out = self.l1(inputs, training=training)
        y_out = self.l2(y_out, training=training)
        if self.residual:
            return inputs + y_out
        return y_out



class Pose3DBase(tf.keras.Model):
    """ Updated Model for 3d-pose-baseline """
    def __init__(self,
                 linear_size,
                 num_layers,
                 dropout_keep_prob,
                 residual=True,
                 batch_norm=True,
                 max_norm=True,
                 predict_14=False):
        """Creates the linear + relu model

        Args
        linear_size: integer. number of units in each layer of the model
        num_layers: integer. number of bilinear blocks in the model
        residual: boolean. Whether to add residual connections
        batch_norm: boolean. Whether to use batch normalization
        max_norm: boolean. Whether to clip weights to a norm of 1
        batch_size: integer. The size of the batches used during training
        learning_rate: float. Learning rate to start with
        summaries_dir: String. Directory where to log progress
        predict_14: boolean. Whether to predict 14 instead of 17 joints
        dtype: the data type to use to store internal variables
        """
        super(Pose3DBase, self).__init__()

        # There are in total 17 joints in H3.6M and 16 in MPII (and therefore in stacked
        # hourglass detections). We settled with 16 joints in 2d just to make models
        # compatible (e.g. you can train on ground truth 2d and test on SH detections).
        # This does not seem to have an effect on prediction performance.
        self.human_2d_size = 16 * 2

        # In 3d all the predictions are zero-centered around the root (hip) joint, so
        # we actually predict only 16 joints. The error is still computed over 17 joints,
        # because if one uses, e.g. Procrustes alignment, there is still error in the
        # hip to account for!
        # There is also an option to predict only 14 joints, which makes our results
        # directly comparable to those in https://arxiv.org/pdf/1611.09010.pdf
        self.human_3d_size = 14 * 3 if predict_14 else 16 * 3

        self.input_size = self.human_2d_size
        self.output_size = self.human_3d_size

        self.linear_size = linear_size
        self.num_layers = num_layers
        self.residual = residual
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_norm = batch_norm
        self.max_norm = max_norm
        self.predict_14 = predict_14

        self.lin = Linear3DBase(self.linear_size,
                                self.human_2d_size,
                                self.dropout_keep_prob,
                                max_norm=self.max_norm,
                                is_output_layer=False)
        self.lint = [TwoLinear3DBase(self.linear_size,
                                     self.linear_size,
                                     self.dropout_keep_prob,
                                     max_norm=self.max_norm,
                                     residual=self.residual) for _ in range(self.num_layers)]
        self.lout = Linear3DBase(self.human_3d_size,
                                 self.linear_size,
                                 self.dropout_keep_prob,
                                 max_norm=self.max_norm,
                                 is_output_layer=True)


    def call(self, enc_in, training=True):
        y_out = self.lin(enc_in, training=training)
        for layer in self.lint:
            y_out = layer(y_out, training=training)
        y_out = self.lout(y_out, training=training)
        return y_out



class PoseBase(tf.keras.Model):
    def __init__(self, units=1024, input_size=32, output_size=48):
        super(PoseBase, self).__init__()

        self.linear_size = units
        self.input_size = input_size
        self.output_size = output_size

        with tf.name_scope('linear_model'):
            self.w1 = tf.Variable(
                shape=(input_size, units),
                initial_value=kaiming(shape=(input_size, units)),
                # initial_value = next(w_gen),
                trainable=True,
                name='w1'
            )
            self.b1 = tf.Variable(
                shape=(units,),
                initial_value=kaiming(shape=(units,)),
                # initial_value = next(w_gen),
                trainable=True,
                name='b1'
            )
            # gamma1 = next(w_gen)
            # beta1 = next(w_gen)
            # mean1 = next(w_gen)
            # variance1 = next(w_gen)
            self.btn1 = keras.layers.BatchNormalization(
                # gamma_initializer=lambda _, dtype=None: gamma1,
                # beta_initializer=lambda _, dtype=None: beta1,
                # moving_mean_initializer=lambda _, dtype=None: mean1,
                # moving_variance_initializer=lambda _, dtype=None: variance1,
                name='batch_normalization'
            )

            with tf.name_scope('two_linear_0'):
                self.w2_0 = tf.Variable(
                    shape=(units, units),
                    initial_value=kaiming(shape=(units, units)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='w2_0'
                )
                self.b2_0 = tf.Variable(
                    shape=(units,),
                    initial_value=kaiming(shape=(units,)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='b2_0'
                )
                # gamma20 = next(w_gen)
                # beta20 = next(w_gen)
                # mean20 = next(w_gen)
                # variance20 = next(w_gen)
                self.btn2_0 = keras.layers.BatchNormalization(
                    # gamma_initializer=lambda _, dtype=None: gamma20,
                    # beta_initializer=lambda _, dtype=None: beta20,
                    # moving_mean_initializer=lambda _, dtype=None: mean20,
                    # moving_variance_initializer=lambda _, dtype=None: variance20,
                    name='batch_normalization10'
                )
                self.w3_0 = tf.Variable(
                    shape=(units, units),
                    initial_value=kaiming(shape=(units, units)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='w3_0'
                )
                self.b3_0 = tf.Variable(
                    shape=(units,),
                    initial_value=kaiming(shape=(units,)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='b3_0'
                )
                # gamma30 = next(w_gen)
                # beta30 = next(w_gen)
                # mean30 = next(w_gen)
                # variance30 = next(w_gen)
                self.btn3_0 = keras.layers.BatchNormalization(
                    # gamma_initializer=lambda _, dtype=None: gamma30,
                    # beta_initializer=lambda _, dtype=None: beta30,
                    # moving_mean_initializer=lambda _, dtype=None: mean30,
                    # moving_variance_initializer=lambda _, dtype=None: variance30,
                    name='batch_normalization20'
                )

            with tf.name_scope('two_linear_1'):
                self.w2_1 = tf.Variable(
                    shape=(units, units),
                    initial_value=kaiming(shape=(units, units)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='w2_1'
                )
                self.b2_1 = tf.Variable(
                    shape=(units,),
                    initial_value=kaiming(shape=(units,)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='b2_1'
                )
                # gamma21 = next(w_gen)
                # beta21 = next(w_gen)
                # mean21 = next(w_gen)
                # variance21 = next(w_gen)
                self.btn2_1 = keras.layers.BatchNormalization(
                    # gamma_initializer=lambda _, dtype=None: gamma21,
                    # beta_initializer=lambda _, dtype=None: beta21,
                    # moving_mean_initializer=lambda _, dtype=None: mean21,
                    # moving_variance_initializer=lambda _, dtype=None: variance21,
                    name='batch_normalization11'
                )
                self.w3_1 = tf.Variable(
                    shape=(units, units),
                    initial_value=kaiming(shape=(units, units)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='w3_1'
                )
                self.b3_1 = tf.Variable(
                    shape=(units,),
                    initial_value=kaiming(shape=(units,)),
                    # initial_value = next(w_gen),
                    trainable=True,
                    name='b3_1'
                )
                # gamma31 = next(w_gen)
                # beta31 = next(w_gen)
                # mean31 = next(w_gen)
                # variance31 = next(w_gen)
                self.btn3_1 = keras.layers.BatchNormalization(
                    # gamma_initializer=lambda _, dtype=None: gamma31,
                    # beta_initializer=lambda _, dtype=None: beta31,
                    # moving_mean_initializer=lambda _, dtype=None: mean31,
                    # moving_variance_initializer=lambda _, dtype=None: variance31,
                    name='batch_normalization21'
                )

            self.w4 = tf.Variable(
                shape=(units, output_size),
                initial_value=kaiming(shape=(units, output_size)),
                # initial_value = next(w_gen),
                trainable=True,
                name='w4'
            )
            self.b4 = tf.Variable(
                shape=(output_size,),
                initial_value=kaiming(shape=(output_size,)),
                # initial_value = next(w_gen),
                trainable=True,
                name='b4'
            )

    def call(self, inputs, training=True):
        y_out = inputs

        w1 = tf.clip_by_norm(self.w1, 1)
        y_out = tf.matmul(y_out, w1) + self.b1
        y_out = self.btn1(y_out, training=training)
        y_out = tf.nn.relu(y_out)

        res1 = y_out

        w2_0 = tf.clip_by_norm(self.w2_0, 1)
        y_out = tf.matmul(y_out, w2_0) + self.b2_0
        y_out = self.btn2_0(y_out, training=training)
        y_out = tf.nn.relu(y_out)

        w3_0 = tf.clip_by_norm(self.w3_0, 1)
        y_out = tf.matmul(y_out, w3_0) + self.b3_0
        y_out = self.btn3_0(y_out, training=training)
        y_out = tf.nn.relu(y_out)

        y_out = res1 + y_out

        res2 = y_out

        w2_1 = tf.clip_by_norm(self.w2_1, 1)
        y_out = tf.matmul(y_out, w2_1) + self.b2_1
        y_out = self.btn2_1(y_out, training=training)
        y_out = tf.nn.relu(y_out)

        w3_1 = tf.clip_by_norm(self.w3_1, 1)
        y_out = tf.matmul(y_out, w3_1) + self.b3_1
        y_out = self.btn3_1(y_out, training=training)
        y_out = tf.nn.relu(y_out)

        y_out = res2 + y_out

        w4 = tf.clip_by_norm(self.w4, 1)
        y_out = tf.matmul(y_out, w4) + self.b4

        return y_out



class Pose3DVae(keras.Model):
    """ Model with the 3dpose prediction + VAE filter """
    def __init__(self, latent_dim, enc_dim, dec_dim, efficient_net=None, use_2d=False):
        super(Pose3DVae, self).__init__()
        self.use_2d = use_2d
        self.human_3d_size = 48
        self.human_2d_size = 32

        vae_input = self.human_3d_size

        if use_2d:
            vae_input += self.human_2d_size

        self.use_effnet = False
        if efficient_net is not None:
            self.use_effnet = True
            if efficient_net == 0:
                self.effnet = efn.EfficientNetB1(weights='imagenet',
                                                 include_top=False,
                                                 pooling='max',
                                                 input_shape=EFFICIENT_NET_INPUT_SHAPE)
            elif efficient_net == 1:
                self.effnet = efn.EfficientNetB0(weights='imagenet',
                                                 include_top=False,
                                                 pooling='max',
                                                 input_shape=EFFICIENT_NET_INPUT_SHAPE)
            else:
                raise Exception("Only B0 or B1 are valid values for Efficient Net")

            vae_input += self.effnet.layers[-1].output_shape[1]

        if use_2d or efficient_net is not None:
            self.concat = tf.keras.layers.Concatenate(axis=1)

        self.pose3d = PoseBase()
        self.vae = VAE(input_size=vae_input,
                       latent_dim=latent_dim,
                       enc_dim=enc_dim,
                       dec_dim=dec_dim)


    def call(self, inputs, frame_inputs=None, training=True):
        out_2d3d = self.pose3d(inputs, training=training)
        out1 = out_2d3d

        if self.use_2d:
            out1 = self.concat([inputs, out1])

        if self.use_effnet:
            out1e = self.effnet(frame_inputs, training=training)
            out1 = self.concat([out1, out1e])

        out2 = self.vae(out1, training=training)

        return out_2d3d, out2
