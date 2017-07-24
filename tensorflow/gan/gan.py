"""Generative Adversarial Networks

An example of distribution approximation using Generative Adversarial Networks in TensorFlow.
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns

sns.set(color_codes=True)
np.random.seed(688)

RANDOM_PORTION = 0.01
HIDDEN_SIZE = 16
BATCH_SIZE = 256
EPOCH = 15000
SAMPLE_RATE = 50


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = .5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def sample(self, N):
        return np.linspace(self._low, self._high, N) + np.random.random(N) * RANDOM_PORTION


def generator(hidden_size):
    model = Sequential()

    model.add(Dense(hidden_size, activation='softplus', batch_input_shape=(BATCH_SIZE, 1), init='normal', name="g0"))
    model.add(Dense(1, init='normal', name="g1"))

    return model


def discriminator(hidden_size):
    model = Sequential()

    model.add(Dense(hidden_size * 2, activation='tanh', batch_input_shape=(BATCH_SIZE, 1), init='normal', name="d0"))
    model.add(Dense(hidden_size * 2, activation='tanh', init='normal', name="d1"))
    model.add(Dense(hidden_size * 2, activation='tanh', init='normal', name="d2"))
    model.add(Dense(1, activation='sigmoid', init='normal', name="d3"))

    return model


def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


def export_animation(anim_frames):
    i = 0
    for t_data, g_data in anim_frames:
        f, ax = plt.subplots(figsize=(12, 8))
        f.suptitle('Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-2, 10)
        ax.set_ylim(0, 1.2)
        sns.distplot(t_data, hist=False, rug=True, color='r', label='Target Data', ax=ax)
        sns.distplot(g_data, hist=False, rug=True, color='g', label='Generated Data', ax=ax)
        f.savefig("images/frame_" + str(i) + ".png")
        print "Frame index: ", i * SAMPLE_RATE
        f.clf()
        plt.close()
        i += 1

# Generate mp4 from images:
# avconv -r 10 -i frame_%d.png -b:v 1000k gan.mp4
# convert -delay 20 -loop 0 output/decision_*.png myimage.gif

def train(_):
    anim_frames = []
    with tf.variable_scope('GAN'):
        G = generator(HIDDEN_SIZE)
        D = discriminator(HIDDEN_SIZE)

        Z = G.input
        X = D.input
        tf.summary.histogram("target", X)

        D1 = D(X)
        G_train = G(Z)
        tf.summary.histogram("generated", G_train)
        D2 = D(G_train)

        loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
        loss_g = tf.reduce_mean(-tf.log(D2))

        tf.summary.scalar("loss_d", loss_d)
        tf.summary.scalar("loss_g", loss_g)

        g_params = G.trainable_weights
        d_params = D.trainable_weights

        opt_g = optimizer(loss_g, g_params)
        opt_d = optimizer(loss_d, d_params)

    with tf.Session() as session:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train', session.graph)

        session.run(tf.global_variables_initializer())

        for step in xrange(EPOCH):
            # update discriminator
            x = DataDistribution().sample(BATCH_SIZE)
            gen = GeneratorDistribution(-2, 10)
            z = gen.sample(BATCH_SIZE)
            _, _, summary = session.run([loss_d, opt_d, merged], {
                X: np.reshape(x, (BATCH_SIZE, 1)),
                Z: np.reshape(z, (BATCH_SIZE, 1))
            })

            # update generator
            z = gen.sample(BATCH_SIZE)
            _,  _, summary = session.run([loss_g, opt_g, merged], {
                X: np.reshape(x, (BATCH_SIZE, 1)),
                Z: np.reshape(z, (BATCH_SIZE, 1))
            })

            G_gen = session.run([G_train], {
                X: np.reshape(x, (BATCH_SIZE, 1)),
                Z: np.reshape(z, (BATCH_SIZE, 1))
            })

            train_writer.add_summary(summary, step)

            if step % SAMPLE_RATE == 0:
                anim_frames.append((x, G_gen))
                print "step: ", step

        export_animation(anim_frames)


if __name__ == "__main__":
    tf.app.run(main=train)
