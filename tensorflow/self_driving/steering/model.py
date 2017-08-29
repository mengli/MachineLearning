import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def p_relu(x, name):
    alphas = tf.get_variable(name,
                             x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=x.dtype)
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg


def inference(x_image, keep_prob):
    #first convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 24])
    b_conv1 = bias_variable([24])

    h_conv1 = p_relu(conv2d(x_image, W_conv1, 2) + b_conv1, 'prelu_conv1')

    #second convolutional layer
    W_conv2 = weight_variable([5, 5, 24, 36])
    b_conv2 = bias_variable([36])

    h_conv2 = p_relu(conv2d(h_conv1, W_conv2, 2) + b_conv2, 'prelu_conv2')

    #third convolutional layer
    W_conv3 = weight_variable([5, 5, 36, 48])
    b_conv3 = bias_variable([48])

    h_conv3 = p_relu(conv2d(h_conv2, W_conv3, 2) + b_conv3, 'prelu_conv3')

    #fourth convolutional layer
    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])

    h_conv4 = p_relu(conv2d(h_conv3, W_conv4, 1) + b_conv4, 'prelu_conv4')

    #fifth convolutional layer
    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])

    h_conv5 = p_relu(conv2d(h_conv4, W_conv5, 1) + b_conv5, 'prelu_conv5')

    #FCL 1
    W_fc1 = weight_variable([1152, 1164])
    b_fc1 = bias_variable([1164])

    h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
    h_fc1 = p_relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1,  'prelu_fc1')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #FCL 2
    W_fc2 = weight_variable([1164, 100])
    b_fc2 = bias_variable([100])

    h_fc2 = p_relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,  'prelu_fc2')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    #FCL 3
    W_fc3 = weight_variable([100, 50])
    b_fc3 = bias_variable([50])

    h_fc3 = p_relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3,  'prelu_fc3')
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    #FCL 3
    W_fc4 = weight_variable([50, 10])
    b_fc4 = bias_variable([10])

    h_fc4 = p_relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4,  'prelu_fc4')
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

    #Output
    W_fc5 = weight_variable([10, 1])
    b_fc5 = bias_variable([1])

    #y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
    y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
    tf.add_to_collection("logits", y)

    return y
