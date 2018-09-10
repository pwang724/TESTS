from __future__ import print_function
import numpy as np
import tensorflow as tf


class Model(object):
    learning_rate = 0.001
    n_input = 0
    n_classes = 0
    X = []
    Y = []
    weights = []
    biases = []
    stride = 2

    loss_op = []
    train_op = []
    accuracy = []

    # output = tf.nn.conv1d(x, filter, stride=2, padding="VALID")
    # batch_size = 32
    # x = tf.placeholder(tf.float32, [batch_size, input_length, channels])
    # filter = tf.zeros([width, input_dim, output_dim])

    def __init__(self, sess, n_input, n_classes, activation_f = 'softmax'):
        self.sess = sess
        self.X = tf.placeholder("float", [None, n_input], name='x') #input is 1-D
        X_expanded = tf.expand_dims(self.X, axis = 2, name= 'expanddim')
        self.Y = tf.placeholder("float", [None, n_classes], name = 'y')

        width =[5, 5, 5]
        dim = [3, 6, 12]
        pool_size = [2, 2, 2]
        pool_stride = [1, 1, 1]

        layer_size = [100]

        self.filters_0 = tf.Variable(tf.random_normal([width[0], 1, dim[0]]))
        filters_1 = tf.Variable(tf.random_normal([width[1], dim[0], dim[1]]))
        filters_2 = tf.Variable(tf.random_normal([width[2], dim[1], dim[2]]))

        filter_biases_0 = tf.Variable(tf.random_normal([dim[0]]))
        filter_biases_1 = tf.Variable(tf.random_normal([dim[1]]))
        filter_biases_2 = tf.Variable(tf.random_normal([dim[2]]))

        with tf.variable_scope('conv0') as scope:
            conv = tf.nn.conv1d(X_expanded, filters= self.filters_0, stride = self.stride, padding='SAME')
            pre_activation = tf.nn.bias_add(conv, filter_biases_0)
            conv2 = tf.nn.relu(pre_activation)
            pool = tf.layers.max_pooling1d(conv2, pool_size= pool_size[0], strides= pool_stride[0], padding='same')
            # norm = tf.nn.lrn(pool, 4, bias=1.0, alpha = 1, beta = .5)

        with tf.variable_scope('local0') as scope:
            in_shp = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [-1, in_shp[1] * in_shp[2]]) #flatten non-batch-size dimensions

            # shape = tf.shape(pool)
            # reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2]]) #get shape of batch_size, and flatten other dimensions
            dim = reshape.get_shape()[1].value #get shape of other dimension
            weights = tf.Variable(tf.random_normal([dim, layer_size[0]]))
            biases = tf.Variable(tf.random_normal([layer_size[0]]))
            local0 = tf.nn.relu(tf.matmul(reshape, weights) + biases)

        with tf.variable_scope('local0') as scope:
            weights = tf.Variable(tf.random_normal([layer_size[0], n_classes]))
            biases = tf.Variable(tf.random_normal([n_classes]))
            logits = tf.add(tf.matmul(local0, weights), biases)

        with tf.variable_scope('out') as scope:
            self.pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)