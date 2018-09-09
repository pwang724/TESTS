import numpy as np
import tensorflow as tf
import Model as M
import MA as MA
import UsefulCommands as UC

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#input
temp_data = []
temp_label = []
n_min = 1
n_max = 10
timeseries_length = 200
for i in range(n_min, n_max + 1):
    data, label = MA.ma_same_constants(i, trials = 1000, length = timeseries_length)
    temp_data.append(data)
    temp_label.append(label)

data = np.concatenate(temp_data, axis=0)
labels = np.concatenate(temp_label, axis=0)
labels = labels.squeeze()
labels_onehot = np.zeros((len(labels), n_max))
labels_onehot[np.arange(len(labels)), labels.astype(int)-1] = 1

#training parameters
batch_size = 100
n_input = timeseries_length
n_classes = n_max - n_min + 1
training_epochs = 200
display_step = 10

#initialize
XB, YB = UC.make_batches(data, labels_onehot, batch_size)
tf.reset_default_graph()
sess = tf.InteractiveSession(graph=tf.Graph())
model = M.Model_MLP(sess, n_input, n_classes, [100, 50], 'softmax')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = len(XB)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = XB[i], YB[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([model.train_op, model.loss_op], feed_dict={model.X: batch_x,
                                                            model.Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            r = np.random.randint(0,total_batch-1)
            acc = model.accuracy.eval({model.X: XB[r], model.Y: YB[r]})
            print("Epoch:", '%04d' % (epoch+1),
                  "cost={:.9f}".format(avg_cost),
                  "Accuracy: %04.4f" % (acc))
    print("Optimization Finished!")

    #holdout batch
    print("Accuracy:", model.accuracy.eval({model.X: XB[-1], model.Y: YB[-1]}))