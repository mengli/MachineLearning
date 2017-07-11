import os
import tensorflow as tf
from utils import camvid
import segnet_vgg

LOG_DIR = 'save'
EPOCH = 5000
BATCH_SIZE = 5
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_CHANNEL = 3
NUM_CLASSES = 11
INITIAL_LEARNING_RATE = 0.0001

image_dir = "/usr/local/google/home/limeng/Downloads/camvid/data/train.txt"
val_dir = "/usr/local/google/home/limeng/Downloads/camvid/data/val.txt"


def loss(logits, labels):
    logits = tf.reshape(logits, [-1, NUM_CLASSES])
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss):
    tf.summary.scalar('total_loss', total_loss)
    optimizer = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(total_loss, global_step=global_step)


def train():
    image_filenames, label_filenames = camvid.get_filename_list(image_dir)
    val_image_filenames, val_label_filenames = camvid.get_filename_list(val_dir)

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config = config)

        train_data = tf.placeholder(tf.float32,
                                    shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        train_labels = tf.placeholder(tf.int64,
                                      shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        is_training = tf.placeholder(tf.bool, name='is_training')

        images, labels = camvid.CamVidInputs(image_filenames,
                                             label_filenames,
                                             BATCH_SIZE)
        val_images, val_labels = camvid.CamVidInputs(val_image_filenames,
                                                     val_label_filenames,
                                                     BATCH_SIZE)

        logits = segnet_vgg.inference(train_data, is_training)
        total_loss = loss(logits, train_labels)
        train_op = train(total_loss)

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('train', sess.graph)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        sess.run(tf.global_variables_initializer())

        for i in range(EPOCH):
            image_batch ,label_batch = sess.run([images, labels])
            feed_dict = {
                train_data: image_batch,
                train_labels: label_batch,
                is_training: True
            }
            _, summary = sess.run([train_op, total_loss, merged_summary_op],
                                  feed_dict=feed_dict)
            if i % 10 == 0:
                print("Start validating...")
                val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                loss_value = total_loss.eval(feed_dict={train_data: val_images_batch,
                                                        train_labels: val_labels_batch,
                                                        is_training: False})
                print("Epoch: %d, Loss: %g" % (i, loss_value))
                if not os.path.exists(LOG_DIR):
                    os.makedirs(LOG_DIR)
                checkpoint_path = os.path.join(LOG_DIR, "model.ckpt")
                saver.save(sess, checkpoint_path)
            # write logs at every iteration
            summary_writer.add_summary(summary, i)
