import os
import tensorflow as tf
from utils import kitti
import segnet_vgg

LOG_DIR = 'save'
EPOCH = 5000
N_cl = 2
UU_TRAIN_SET_SIZE = 98 - 9
UU_TEST_SET_SIZE = 9

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
sess = tf.InteractiveSession(config = config)

kitti_data = kitti.Kitti()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

sqrt_diff = tf.reduce_mean(tf.square(tf.subtract(segnet_vgg.y_, segnet_vgg.y)))
norm =  + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
loss = sqrt_diff + norm
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

# op to write logs to Tensorboard
summary_writer = tf.summary.FileWriter('train', sess.graph)

for i in range(EPOCH):
    xs, ys = kitti_data.next_batch(i % UU_TRAIN_SET_SIZE)
    train_step.run(feed_dict={segnet_vgg.x: xs, segnet_vgg.y_: ys, segnet_vgg.keep_prob: 0.8})
    if i % 10 == 0:
        xs, ys = kitti_data.load_val_batch()
        loss_value = loss.eval(feed_dict={segnet_vgg.x: xs,
                                          segnet_vgg.y_: ys,
                                          segnet_vgg.keep_prob: 1.0})
            print("Epoch: %d, Loss: %g" % (i, loss_value))
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        checkpoint_path = os.path.join(LOG_DIR, "model.ckpt")
        filename = saver.save(sess, checkpoint_path)

    # write logs at every iteration
    summary = merged_summary_op.eval(
        feed_dict={segnet_vgg.x: xs, segnet_vgg.y_: ys, segnet_vgg.keep_prob: 1.0})
    summary_writer.add_summary(summary, i)
