import os
import tensorflow as tf
import driving_data
import model

LOG_DIR = 'save'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(
    tf.subtract(
        model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

# op to write logs to Tensorboard
summary_writer = tf.summary.FileWriter('train', sess.graph)

epochs = 50
batch_size = 200

for epoch in range(epochs):
    for i in range(int(driving_data.num_images / batch_size)):
        xs, ys = driving_data.load_train_batch(batch_size)
        train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
        if i % 10 == 0:
            xs, ys = driving_data.load_val_batch(batch_size)
            loss_value = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

        # write logs at every iteration
        summary = merged_summary_op.eval(
            feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        summary_writer.add_summary(summary, epoch * driving_data.num_images / batch_size + i)

        if i % batch_size == 0:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            checkpoint_path = os.path.join(LOG_DIR, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
