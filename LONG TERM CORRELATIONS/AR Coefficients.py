import time
import numpy as np
import tensorflow as tf
import Model_MLP as M
import MA as MA
import UsefulCommands as UC
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#input
n = 1
num_unique_coefficients = 1000
trials_per_set_of_coefficients = 100
timeseries_length = 500

start_time = time.time()

temp_data = []
temp_label = []
for i in range(num_unique_coefficients):
    data, label = MA.ma_random_coefficients(n, trials = trials_per_set_of_coefficients, length = timeseries_length)
    temp_data.append(data)
    temp_label.append(label)

data = np.concatenate(temp_data, axis=0)
labels = np.concatenate(temp_label, axis=0)
labels = labels[:,1:2]

# your code
elapsed_time = time.time() - start_time
print("Make input time: %4.4f" % elapsed_time)

#training parameters
batch_size = 100
n_input = timeseries_length
n_classes = 1
training_epochs = 100
display_step = 10

#initialize
XB, YB = UC.make_batches(data, labels, batch_size)
tf.reset_default_graph()
sess = tf.InteractiveSession(graph=tf.Graph())
model = M.Model_MLP(sess, n_input, n_classes, [100, 50], 'sigmoid')
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
            avg_cost += c / (total_batch * batch_size)
        # Display logs per epoch step
        if epoch % display_step == 0:
            acc = model.loss_op.eval({model.X: XB[-1], model.Y: YB[-1]})
            print("Epoch:", '%04d' % (epoch+1),
                  "cost={:.4f}".format(avg_cost*1000),
                  "Accuracy: %04.4f" % (acc))
    print("Optimization Finished!")

    #holdout batch
    # print("Accuracy:", model.accuracy.eval({model.X: XB[-1], model.Y: YB[-1]}))
    pred = sess.run(model.pred, {model.X: XB[-1]})

    a = pred[0:10,:]
    b = YB[-1][0:10,:]
    print(np.concatenate((a,b),axis=1))
    # print(pred[0:10,:])
    # print(YB[-1][0:10,:])