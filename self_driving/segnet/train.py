"""Train SegNet with camvid dataset.

nohup python -u -m self_driving.segnet.train > self_driving/segnet/output.txt 2>&1 &

"""

import os
import tensorflow as tf
from utils import camvid
import segnet_vgg

LOG_DIR = 'save'
EPOCH = 6000
BATCH_SIZE = 4
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960
IMAGE_CHANNEL = 3
NUM_CLASSES = 32
INITIAL_LEARNING_RATE = 0.0001

image_dir = "/usr/local/google/home/limeng/Downloads/camvid/train.txt"
val_dir = "/usr/local/google/home/limeng/Downloads/camvid/val.txt"


def loss(logits, labels):
    logits = tf.reshape(logits, [-1, NUM_CLASSES])
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE, global_step, EPOCH * 0.2, 0.9, staircase=True)
        tf.summary.scalar('total_loss', total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(total_loss, global_step=global_step)


def main(_):
    image_filenames, label_filenames = camvid.get_filename_list(image_dir)
    val_image_filenames, val_label_filenames = camvid.get_filename_list(val_dir)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # config = tf.ConfigProto(device_count = {'GPU': 0})
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            sess = tf.InteractiveSession(config=config)

            train_data = tf.placeholder(tf.float32,
                                        shape=[BATCH_SIZE,
                                               IMAGE_HEIGHT,
                                               IMAGE_WIDTH,
                                               IMAGE_CHANNEL],
                                        name='train_data')
            train_labels = tf.placeholder(tf.int64,
                                          shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1],
                                          name='train_labels')
            is_training = tf.placeholder(tf.bool, name='is_training')

            images, labels = camvid.CamVidInputs(image_filenames,
                                                 label_filenames,
                                                 BATCH_SIZE)
            val_images, val_labels = camvid.CamVidInputs(val_image_filenames,
                                                         val_label_filenames,
                                                         BATCH_SIZE)

            logits = segnet_vgg.inference(train_data, is_training, NUM_CLASSES)
            total_loss = loss(logits, train_labels)
            train_op = train(total_loss)
            check_op = tf.add_check_numerics_ops()

            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('train', sess.graph)
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            checkpoint_path = os.path.join(LOG_DIR, "segnet.ckpt")

            sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(EPOCH):
                image_batch, label_batch = sess.run([images, labels])
                feed_dict = {
                    train_data: image_batch,
                    train_labels: label_batch,
                    is_training: True
                }
                _, _, _, summary = sess.run([train_op, total_loss, check_op, merged_summary_op],
                                            feed_dict=feed_dict)
                if i % 10 == 0:
                    print("Start validating...")
                    val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                    loss_value = total_loss.eval(feed_dict={train_data: val_images_batch,
                                                            train_labels: val_labels_batch,
                                                            is_training: True})
                    print("Epoch: %d, Loss: %g" % (i, loss_value))
                    saver.save(sess, checkpoint_path)
                # write logs at every iteration
                summary_writer.add_summary(summary, i)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
