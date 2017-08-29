"""Evaluate SegNet.

nohup python -u -m self_driving.steering.evaluate > self_driving/segnet/output.txt 2>&1 &

"""

import os
import tensorflow as tf
from utils import udacity_data

LOG_DIR = 'save'
BATCH_SIZE = 128
EPOCH = udacity_data.NUM_VAL_IMAGES / BATCH_SIZE
OUTPUT = "steering_out.txt"


def main(_):
    udacity_data.read_data(shuffe=False)
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)

        saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, "steering.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

        graph = tf.get_default_graph()
        x_image = graph.get_tensor_by_name("x_image:0")
        y_label = graph.get_tensor_by_name("y_label:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        logits = tf.get_collection("logits")[0]

        if os.path.exists(OUTPUT):
            os.remove(OUTPUT)

        for epoch in range(EPOCH):
            image_batch, label_batch = udacity_data.load_val_batch(BATCH_SIZE)
            feed_dict = {
                x_image: image_batch,
                y_label: label_batch,
                keep_prob: 0.6
            }
            prediction = sess.run([logits], feed_dict)
            with open(OUTPUT, 'a') as out:
                for batch in range(BATCH_SIZE):
                    out.write("%s %.10f\n" % (udacity_data.val_xs[epoch * BATCH_SIZE + batch],
                                            prediction[0][batch]))


if __name__ == '__main__':
    tf.app.run(main=main)
