from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class Model(object):
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

    def __init__(self, sess, n_time, n_classes, n_hidden, activation_f='softmax'):
        self.sess = sess
        self.X = tf.placeholder("float", [None, n_time], name='x')
        self.Y = tf.placeholder("float", [None, n_classes], name='y')
        X_tensor = tf.expand_dims(self.X, axis=2) #last dimension is input dimension, n_input == 1
        X_timeslice_of_tensors = tf.unstack(X_tensor, axis=1)

        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, X_timeslice_of_tensors, dtype=tf.float32)

        weight_out = tf.Variable(tf.random_normal([n_hidden, n_classes]))
        bias_out = tf.Variable(tf.random_normal([n_classes]))

        logits = tf.add(tf.matmul(outputs[-1], weight_out), bias_out)

        if activation_f == 'softmax':
            self.pred = tf.nn.softmax(logits)
        elif activation_f == 'sigmoid':
            self.pred = tf.nn.sigmoid(logits)

        # accuracy, loss, train
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