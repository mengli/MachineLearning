# Import all of our packages

import prettytensor as pt
import tensorflow as tf
from deconv import deconv2d
from cifar import Cifar

dim1 = 32  # first dimension of input data
dim2 = 32  # second dimension of input data
dim3 = 3  # third dimension of input data (colors)
batch_size = 64  # size of batches to use (per GPU)
hidden_size = 256  # size of hidden (z) layer to use
num_epochs = 100000  # number of epochs to run


def encoder(X):
    '''Create encoder network.
    Args:
        x: a batch of flattened images [batch_size, 32*32]
    Returns:
        A tensor that expresses the encoder network
            # The transformation is parametrized and can be learned.
            # returns network output, mean, setd
    '''
    lay_end = (pt.wrap(X).
               reshape([batch_size, dim1, dim2, dim3]).
               conv2d(5, 16, stride=2).
               conv2d(5, 32, stride=2).
               flatten())
    z_mean = lay_end.fully_connected(hidden_size, activation_fn=None)
    z_log_sigma_sq = lay_end.fully_connected(hidden_size, activation_fn=None)
    return z_mean, z_log_sigma_sq


def generator(Z):
    '''Create generator network.
        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        x: a batch of vectors to decode
    Returns:
        A tensor that expresses the generator network
    '''
    return (pt.wrap(Z).
            fully_connected(8 * 8 * 32).reshape([batch_size, 8, 8, 32]).  # (128, 4 4, 256)
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(1, dim3, stride=1, activation_fn=tf.sigmoid).
            flatten()
            )


def discriminator(D_I):
    ''' A encodes
    Create a network that discriminates between images from a dataset and
    generated ones.
    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    descrim_conv = (pt.wrap(D_I).  # This is what we're descriminating
                    reshape([batch_size, dim1, dim2, dim3]).
                    conv2d(5, 16, stride=1).
                    conv2d(5, 32, stride=2).
                    flatten()
                    )
    lth_layer = descrim_conv.fully_connected(1024, activation_fn=tf.nn.elu)  # this is the lth layer
    D = lth_layer.fully_connected(1, activation_fn=tf.nn.sigmoid)  # this is the actual discrimination
    return D, lth_layer


def inference(x):
    """
    Run the models. Called inference because it does the same thing as tensorflow's cifar tutorial
    """
    z_p = tf.random_normal((batch_size, hidden_size), 0, 1)  # normal dist for GAN
    eps = tf.random_normal((batch_size, hidden_size), 0, 1)  # normal dist for VAE

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with tf.variable_scope("enc"):
            z_x_mean, z_x_log_sigma_sq = encoder(x)  # get z from the input
        with tf.variable_scope("gen"):
            z_x = tf.add(z_x_mean,
                         tf.mul(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps))  # grab our actual z
            x_tilde = generator(z_x)
        with tf.variable_scope("dis"):
            d_tilde_p, l_x_tilde = discriminator(x_tilde)
        with tf.variable_scope("gen", reuse=True):
            x_p = generator(z_p)
        with tf.variable_scope("dis", reuse=True):
            d_x, l_x = discriminator(x)  # positive examples
        with tf.variable_scope("dis", reuse=True):
            d_x_p, _ = discriminator(x_p)

        kernel_grad_grid = put_kernels_on_grid(tf.transpose(tf.reshape(x_tilde, [-1, 3, 32, 32]), [2, 3, 1, 0]), (8, 8))
        tf.summary.image("gen_images", kernel_grad_grid, max_outputs=1)

        return z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p, d_tilde_p


def loss(x, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, d_tilde_p, dim1, dim2, dim3):
    """
    Loss functions for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    ### We don't actually use SSE (MSE) loss for anything (but maybe pretraining)
    SSE_loss = tf.reduce_mean(tf.square(x - x_tilde))  # This is what a normal VAE uses

    # We clip gradients of KL divergence to prevent NANs
    KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + z_x_log_sigma_sq
                                                 - tf.square(z_x_mean)
                                                 - tf.exp(z_x_log_sigma_sq), 1)) / dim1 / dim2 / dim3
    tf.summary.scalar("KL_loss", KL_loss)

    # Discriminator Loss
    D_loss = tf.reduce_mean(-1. * (tf.log(d_x) +
                                   tf.log(1.0 - d_x_p)))
    tf.summary.scalar("D_loss", D_loss)

    # Generator Loss
    G_loss = tf.reduce_mean(-1. * (tf.log(d_x_p)))  # +
    # tf.log(tf.clip_by_value(1.0 - d_x,1e-5,1.0))))
    tf.summary.scalar("G_loss", G_loss)
    # Lth Layer Loss - the 'learned similarity measure'
    LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde)) / dim1 / dim2 / dim3
    tf.summary.scalar("LL_loss", LL_loss)

    return SSE_loss, KL_loss, D_loss, G_loss, LL_loss


def put_kernels_on_grid(kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    ch = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, ch]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, ch]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8


def learning_rate(global_step):
    starter_learning_rate = 0.001
    learning_rate_1 = tf.train.exponential_decay(
        starter_learning_rate, global_step, num_epochs * 0.2, 0.1, staircase=True)
    learning_rate_2 = tf.train.exponential_decay(
        learning_rate_1, global_step, num_epochs * 0.4, 0.5, staircase=True)
    decayed_learning_rate = tf.train.exponential_decay(
        learning_rate_2, global_step, num_epochs * 0.6, 0.8, staircase=True)
    tf.summary.scalar('learning_rate', decayed_learning_rate)
    return decayed_learning_rate


def main(_):
    cifar10 = Cifar()
    cifar10.ReadDataSets(one_hot=True)

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    opt_D = tf.train.AdamOptimizer(learning_rate(global_step), epsilon=1.0)
    opt_G = tf.train.AdamOptimizer(learning_rate(global_step), epsilon=1.0)
    opt_E = tf.train.AdamOptimizer(learning_rate(global_step), epsilon=1.0)

    x = tf.placeholder(tf.float32, [None, 3072])

    image_grid = put_kernels_on_grid(tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [2, 3, 1, 0]), (8, 8))
    tf.summary.image("images", image_grid, max_outputs=1)

    z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p, d_tilde_p = inference(x)

    # Calculate the loss for this tower
    SSE_loss, KL_loss, D_loss, G_loss, LL_loss = loss(x, x_tilde, z_x_log_sigma_sq,
                                                      z_x_mean, d_x, d_x_p, l_x, l_x_tilde, d_tilde_p, dim1,
                                                      dim2, dim3)  # specify loss to parameters
    params = tf.trainable_variables()
    E_params = [i for i in params if 'enc' in i.name]
    G_params = [i for i in params if 'gen' in i.name]
    D_params = [i for i in params if 'dis' in i.name]

    # Calculate the losses specific to encoder, generator, decoder
    L_e = KL_loss + LL_loss
    L_g = LL_loss + G_loss
    L_d = D_loss

    tf.summary.scalar('L_e', L_e)
    tf.summary.scalar('L_g', L_g)
    tf.summary.scalar('L_d', L_d)

    # Reuse variables for the next tower.
    tf.get_variable_scope().reuse_variables()

    # Calculate the gradients for the batch of data on this CIFAR tower.
    grads_e = opt_E.compute_gradients(L_e, var_list=E_params)
    grads_g = opt_G.compute_gradients(L_g, var_list=G_params)
    grads_d = opt_D.compute_gradients(L_d, var_list=D_params)

    # apply the gradients with our optimizers
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    train_G = opt_G.apply_gradients(grads_g, global_step=global_step)
    train_D = opt_D.apply_gradients(grads_d, global_step=global_step)

    epoch = 0

    # Start the Session
    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)

    sess.run(tf.global_variables_initializer())

    while epoch < num_epochs:
        batch = cifar10.train.next_batch(batch_size)

        _, _, _, D_err, G_err, KL_err, SSE_err, LL_err, d_fake, d_real, summary = sess.run([
            train_E, train_G, train_D,
            D_loss, G_loss, KL_loss, SSE_loss, LL_loss,
            d_x_p, d_x, merged,

        ],
            {
                x: batch[0],
            }
        )

        train_writer.add_summary(summary, epoch)

        print('Epoch: ' + str(epoch))
        epoch += 1


if __name__ == '__main__':
    tf.app.run(main=main)
