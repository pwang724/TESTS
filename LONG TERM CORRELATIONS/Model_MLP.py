from __future__ import print_function
import numpy as np
import tensorflow as tf


class Model_MLP(object):
    learning_rate = 0.001
    n_input = 0
    n_classes = 0
    X = []
    Y = []
    weights = []
    biases = []

    loss_op = []
    train_op = []
    accuracy = []

    def __init__(self, sess, n_input, n_classes, hidden_size, activation_f = 'softmax'):
        self.sess = sess
        self.X = tf.placeholder("float", [None, n_input], name='x')
        self.Y = tf.placeholder("float", [None, n_classes], name = 'y')

        #weights
        layer_size = [n_input] + hidden_size + [n_classes]
        for i in range(len(layer_size)-1):
            self.weights.append(tf.Variable(tf.random_normal([layer_size[i], layer_size[i+1]])))
            self.biases.append(tf.Variable(tf.random_normal([layer_size[i+1]])))

        #model
        temp = tf.add(tf.matmul(self.X, self.weights[0]), self.biases[0])
        for i in range(1, len(layer_size)-2):
            temp = tf.add(tf.matmul(temp, self.weights[i]), self.biases[i])
            temp = tf.nn.relu(temp)
        logits = tf.add(tf.matmul(temp,self.weights[-1]),self.biases[-1])

        if activation_f == 'softmax':
            self.pred = tf.nn.softmax(logits)
        elif activation_f == 'sigmoid':
            self.pred = tf.nn.sigmoid(logits)

        #accuracy, loss, train
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        if activation_f == 'softmax':
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=self.Y))
        elif activation_f == 'sigmoid':
            self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=self.Y))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)