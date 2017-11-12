#!/usr/bin/env python

import scipy as scp
import scipy.misc
import matplotlib as mpl
import matplotlib.cm
import logging
import tensorflow as tf
import sys
import fcn8_vgg


def main(_):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
    img1 = scp.misc.imread("/Users/limeng/Downloads/kitti/data_road/training/image_2/uu_000000.png")
    with tf.Session() as sess:
        images = tf.placeholder("float")
        feed_dict = {images: img1}
        batch_images = tf.expand_dims(images, 0)

        vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path="/Users/limeng/Downloads/vgg16.npy")
        with tf.name_scope("content_vgg"):
            vgg_fcn.build(batch_images, debug=True, num_classes=2)

        print('Finished building Network.')

        logging.warning("Score weights are initialized random.")
        logging.warning("Do not expect meaningful results.")

        logging.info("Start Initializing Variabels.")

        init = tf.global_variables_initializer()
        sess.run(init)

        print('Running the Network')
        tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
        down, up = sess.run(tensors, feed_dict=feed_dict)

        down_color = color_image(down[0], 2)
        up_color = color_image(up[0], 2)

        scp.misc.imsave('fcn8_downsampled.png', down_color)
        scp.misc.imsave('fcn8_upsampled.png', up_color)


def color_image(image, num_classes=20):
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))


if __name__ == '__main__':
    tf.app.run(main=main)
