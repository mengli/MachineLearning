# nohup python -u -m self_driving.steering.train > self_driving/steering/output.txt 2>&1 &

import os
import tensorflow as tf
from utils import udacity_data
import model

LOG_DIR = 'save'
EPOCH = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
STEP_PER_EPOCH = udacity_data.NUM_TRAIN_IMAGES / BATCH_SIZE


def loss(pred, labels):
    train_vars = tf.trainable_variables()
    norm = tf.add_n([tf.nn.l2_loss(v) for v in train_vars])
    # create a summary to monitor L2 norm
    tf.summary.scalar('L2 Normalization', norm)
    losses = tf.reduce_mean(tf.square(tf.subtract(pred, labels)))
    # create a summary to monitor loss
    tf.summary.scalar('Loss', losses)
    return norm, losses, losses + norm * 0.00001


def train(total_loss):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # create a summary to monitor total loss
    tf.summary.scalar('Total Loss', total_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    return optimizer.minimize(total_loss, global_step=global_step)


def main(_):
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)

        x_image = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name="x_image")
        y_label = tf.placeholder(tf.float32, shape=[None, 1], name="y_label")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        y_pred = model.inference(x_image, keep_prob)
        norm, losses, total_loss = loss(y_pred, y_label)
        train_op = train(total_loss)

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('train', sess.graph)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        checkpoint_path = os.path.join(LOG_DIR, "steering.ckpt")

        sess.run(tf.global_variables_initializer())

        udacity_data.read_data()

        for epoch in range(EPOCH):
            for i in range(STEP_PER_EPOCH):
                steps = epoch * STEP_PER_EPOCH + i

                xs, ys = udacity_data.load_train_batch(BATCH_SIZE)

                _, summary = sess.run([train_op, merged_summary_op],
                                      feed_dict={x_image: xs, y_label: ys, keep_prob: 0.6})

                if i % 10 == 0:
                    xs, ys = udacity_data.load_val_batch(BATCH_SIZE)
                    loss_value = losses.eval(feed_dict={x_image: xs, y_label: ys, keep_prob: 0.6})
                    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, steps, loss_value))

                # write logs at every iteration
                summary_writer.add_summary(summary, steps)

                if i % 32 == 0:
                    if not os.path.exists(LOG_DIR):
                        os.makedirs(LOG_DIR)
                    saver.save(sess, checkpoint_path)


if __name__ == '__main__':
    tf.app.run(main=main)
