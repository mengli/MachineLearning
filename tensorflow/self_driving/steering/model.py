import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def inference(x_image, keep_prob, is_training=True):
    #first convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 24])
    b_conv1 = bias_variable([24])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1, 'relu_conv1')
    h_conv1_norm = tf.contrib.layers.batch_norm(h_conv1, is_training=is_training, trainable=True)

    #second convolutional layer
    W_conv2 = weight_variable([5, 5, 24, 36])
    b_conv2 = bias_variable([36])

    h_conv2 = tf.nn.relu(conv2d(h_conv1_norm, W_conv2, 2) + b_conv2, 'relu_conv2')
    h_conv2_norm = tf.contrib.layers.batch_norm(h_conv2, is_training=is_training, trainable=True)

    #third convolutional layer
    W_conv3 = weight_variable([5, 5, 36, 48])
    b_conv3 = bias_variable([48])

    h_conv3 = tf.nn.relu(conv2d(h_conv2_norm, W_conv3, 2) + b_conv3, 'relu_conv3')
    h_conv3_norm = tf.contrib.layers.batch_norm(h_conv3, is_training=is_training, trainable=True)

    #fourth convolutional layer
    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])

    h_conv4 = tf.nn.relu(conv2d(h_conv3_norm, W_conv4, 1) + b_conv4, 'relu_conv4')
    h_conv4_norm = tf.contrib.layers.batch_norm(h_conv4, is_training=is_training, trainable=True)

    #fifth convolutional layer
    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])

    h_conv5 = tf.nn.relu(conv2d(h_conv4_norm, W_conv5, 1) + b_conv5, 'relu_conv5')
    h_conv5_norm = tf.contrib.layers.batch_norm(h_conv5, is_training=is_training, trainable=True)

    #FCL 1
    W_fc1 = weight_variable([1152, 1164])
    b_fc1 = bias_variable([1164])

    h_conv5_flat = tf.reshape(h_conv5_norm, [-1, 1152])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1,  'relu_fc1')
    h_fc1_norm = tf.contrib.layers.batch_norm(h_fc1, is_training=is_training, trainable=True)
    h_fc1_drop = tf.nn.dropout(h_fc1_norm, keep_prob)

    #FCL 2
    W_fc2 = weight_variable([1164, 100])
    b_fc2 = bias_variable([100])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,  'relu_fc2')
    h_fc2_norm = tf.contrib.layers.batch_norm(h_fc2, is_training=is_training, trainable=True)
    h_fc2_drop = tf.nn.dropout(h_fc2_norm, keep_prob)

    #FCL 3
    W_fc3 = weight_variable([100, 50])
    b_fc3 = bias_variable([50])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3,  'relu_fc3')
    h_fc3_norm = tf.contrib.layers.batch_norm(h_fc3, is_training=is_training, trainable=True)
    h_fc3_drop = tf.nn.dropout(h_fc3_norm, keep_prob)

    #FCL 3
    W_fc4 = weight_variable([50, 10])
    b_fc4 = bias_variable([10])

    h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4,  'relu_fc4')
    h_fc4_norm = tf.contrib.layers.batch_norm(h_fc4, is_training=is_training, trainable=True)
    h_fc4_drop = tf.nn.dropout(h_fc4_norm, keep_prob)

    #Output
    W_fc5 = weight_variable([10, 1])
    b_fc5 = bias_variable([1])

    y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
    #y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
    tf.add_to_collection("logits", y)

    return y
