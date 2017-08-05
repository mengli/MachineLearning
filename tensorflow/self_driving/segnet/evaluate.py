"""Evaluate SegNet.

nohup python -u -m self_driving.segnet.evaluate > self_driving/segnet/output.txt 2>&1 &

"""

import os
import tensorflow as tf
from utils import camvid
from scipy import misc

LOG_DIR = 'save'
BATCH_SIZE = 4
EPOCH = 25
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960
IMAGE_CHANNEL = 3
NUM_CLASSES = 32

test_dir = "/usr/local/google/home/limeng/Downloads/camvid/val.txt"

colors = [
    [64, 128, 64],  # Animal
    [192, 0, 128],  # Archway
    [0, 128, 192],  # Bicyclist
    [0, 128, 64],  # Bridge
    [128, 0, 0],  # Building
    [64, 0, 128],  # Car
    [64, 0, 192],  # CartLuggagePram
    [192, 128, 64],  # Child
    [192, 192, 128],  # Column_Pole
    [64, 64, 128],  # Fence
    [128, 0, 192],  # LaneMkgsDriv
    [192, 0, 64],  # LaneMkgsNonDriv
    [128, 128, 64],  # Misc_Text
    [192, 0, 192],  # MotorcycleScooter
    [128, 64, 64],  # OtherMoving
    [64, 192, 128],  # ParkingBlock
    [64, 64, 0],  # Pedestrian
    [128, 64, 128],  # Road
    [128, 128, 192],  # RoadShoulder
    [0, 0, 192],  # Sidewalk
    [192, 128, 128],  # SignSymbol
    [128, 128, 128],  # Sky
    [64, 128, 192],  # SUVPickupTruck
    [0, 0, 64],  # TrafficCone
    [0, 64, 64],  # TrafficLight
    [192, 64, 128],  # Train
    [128, 128, 0],  # Tree
    [192, 128, 192],  # Truck_Bus
    [64, 0, 64],  # Tunnel
    [192, 192, 0],  # VegetationMisc
    [0, 0, 0],  # Void
    [64, 192, 0]  # Wall
]


def color_mask(tensor, color):
    return tf.reduce_all(tf.equal(tensor, color), 3)


def one_hot(labels):
    color_tensors = tf.unstack(colors)
    channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
    one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
    return one_hot_labels


def rgb(logits):
    softmax = tf.nn.softmax(logits)
    argmax = tf.argmax(softmax, 3)
    color_map = tf.constant(colors, dtype=tf.float32)
    n = color_map.get_shape().as_list()[0]
    one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    one_hot_matrix = tf.reshape(one_hot, [-1, n])
    rgb_matrix = tf.matmul(one_hot_matrix, color_map)
    rgb_tensor = tf.reshape(rgb_matrix, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return tf.cast(rgb_tensor, tf.float32)


def main(_):
    test_image_filenames, test_label_filenames = camvid.get_filename_list(test_dir)
    index = 0

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            sess = tf.InteractiveSession(config=config)

            images, labels = camvid.CamVidInputs(test_image_filenames,
                                                 test_label_filenames,
                                                 BATCH_SIZE,
                                                 shuffle=False)

            saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, "segnet.ckpt.meta"))
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

            graph = tf.get_default_graph()
            train_data = graph.get_tensor_by_name("train_data:0")
            train_label = graph.get_tensor_by_name("train_labels:0")
            is_training = graph.get_tensor_by_name("is_training:0")
            logits = tf.get_collection("logits")[0]

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(EPOCH):
                image_batch, label_batch = sess.run([images, labels])
                feed_dict = {
                    train_data: image_batch,
                    train_label: label_batch,
                    is_training: True
                }
                prediction = rgb(logits)
                pred = sess.run([prediction], feed_dict)[0]
                for batch in range(BATCH_SIZE):
                    misc.imsave('output/segnet_camvid/decision_%d.png' % index, pred[batch])
                    misc.imsave('output/segnet_camvid/train_%d.png' % index, image_batch[batch])
                    index += 1

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
