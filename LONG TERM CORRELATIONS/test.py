import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

x = np.random.randint(0,10, size = (4,5))

batch_size, time_steps = x.shape

# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, timesteps, n_input)
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
X = tf.placeholder("float", [None, time_steps])
Y = tf.expand_dims(X, axis = 2)
Z = tf.transpose(
    Y,
    perm= [1, 0, 2],
    name='transpose',
    conjugate=False
)
ZZ = tf.unstack(Z, axis=0)

lstm_cell = rnn.BasicLSTMCell(5, forget_bias=1.0)

# Get lstm cell output
outputs, states = rnn.static_rnn(lstm_cell, ZZ, dtype=tf.float32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a = sess.run(Y, feed_dict= {X: x})
    print(x)
    print(a)
