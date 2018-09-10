import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import random
import time
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

# the system input argument define which model selected
unit_name = sys.argv[1]
mode = sys.argv[2]

if mode == "32":
    rnn_size = 32
elif mode == "64":
    rnn_size = 64
elif mode == "128":
    rnn_size = 128
elif mode == "stack_32":
    rnn_size = 32
    
def save_model(sess):
    # a function to save model so that we can recover from the futher
    if not os.path.exists('../models/'):
        os.mkdir('../models/')
    saver = tf.train.Saver()
    saver.save(sess, '../models/pixel_{}_{}.checkpoint'.format(unit_name, mode))
    print('model saved to ../models/pixel_{}_{}.checkpoint'.format(unit_name, mode))
    
def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    # a function to save learning curves
    if not os.path.exists('../models/'):
        os.mkdir('../models/')
    np.save("../models/pixel_{}_{}_train_error_list.npy".format(unit_name, mode), train_error_list)
    np.save("../models/pixel_{}_{}_test_error_list.npy".format(unit_name, mode), test_error_list)
    np.save("../models/pixel_{}_{}_train_loss_list.npy".format(unit_name, mode), train_loss_list)
    np.save("../models/pixel_{}_{}_test_loss_list.npy".format(unit_name, mode), test_loss_list)


# a function to binarize data

def binarize(data, threshold=0.1):
    data = (threshold < data).astype('float32')
    return data

# define some hyperparamters
hm_epochs = 500
n_classes = 10
batch_size = 512
chunk_size = 1
num_chunks = 784

# read the data
data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# binarize the data
X_train_original = binarize(mnist.train.images)
y_train_original = mnist.train.labels

X_vali_original = binarize(mnist.validation.images)
y_vali_original = mnist.validation.labels

X_test_original = binarize(mnist.test.images)
y_test_original = mnist.test.labels

# a function used to reshape data
def reshape_data(X_data):
    X_data = X_data.reshape([-1,num_chunks,chunk_size])
    y_data = X_data.transpose([1,0,2])[1:].reshape([-1, chunk_size])
    return X_data,y_data

# reshape data
X_train, y_train = reshape_data(X_train_original)
X_vali, y_vali = reshape_data(X_vali_original)
X_test, y_test = reshape_data(X_test_original)

# define some place holder
x = tf.placeholder('float', [None, num_chunks, chunk_size])
y = tf.placeholder('float')


# define the recurrent neural network
def recurrent_neural_network(x):
    # get the shape of x
    x_shape = tf.shape(x)[0]
    # define some weights for RNN
    rnn_sts_layer = {'weight':tf.Variable(tf.random_normal([rnn_size,1])),\
                     'bias':tf.Variable(tf.random_normal([1]))}

    # reshape the data
    before = tf.transpose(x,[1,0,2])
    before = tf.reshape(before,[-1,chunk_size])
    before = tf.split(before,num_or_size_splits=num_chunks,axis=0) 

    # define RNN cell
    if unit_name == "GRU":
        RNN_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_size)
    elif unit_name == "LSTM":
        RNN_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size)

    if mode == "stack_32":
        stack_rnn = []
        for i in range(3):
            stack_rnn.append(tf.nn.rnn_cell.GRUCell(num_units=rnn_size))
        RNN_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
    
    
    outputs, states = tf.contrib.rnn.static_rnn(RNN_cell,before,dtype=tf.float32)

    # calculate the pixels
    pixels = []
    for this_pixel in range(len(outputs)-1):
        pix = tf.add(tf.matmul(outputs[this_pixel],rnn_sts_layer['weight']),rnn_sts_layer['bias'])
        pixels.append(pix)
    # concat the pixels and output
    final_pixels = tf.concat(pixels,axis=0)
    rnn_output = tf.reshape(final_pixels,[(num_chunks-1) * x_shape,-1])

    return rnn_output



# define the predictions
prediction = recurrent_neural_network(x)
prediction_sig = tf.nn.sigmoid(prediction)


loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(loss)
correct = tf.equal(tf.round(prediction_sig),y) 
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
# this is a function used to calculate loss and error for all data set
def get_error_and_loss(sess):    
    vali_loss = sess.run(loss, feed_dict={x: X_vali, y: y_vali})
    vali_error = 1 - accuracy.eval({x: X_vali, y: y_vali})
    
    test_loss = sess.run(loss, feed_dict={x: X_test, y: y_test})
    test_error = 1 - accuracy.eval({x: X_test, y: y_test})
    
    train_loss = sess.run(loss, feed_dict={x: X_train, y: y_train})
    train_error = 1 - accuracy.eval({x: X_train, y: y_train})

    return test_error, test_loss, vali_error, vali_loss, train_error, train_loss

# list to store learning curves
train_error_list = []
test_error_list = []

train_loss_list = []
test_loss_list = []

# define the lowest validation loss, in order to perform early stop
lowest_vali_loss = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        t_ref = time.time()
        # random batch and train
        for epo in range(20):
            epoch_num = np.random.choice(len(X_train_original), batch_size, replace=False)
            epoch_x = X_train_original[epoch_num]

            epoch_x, epoch_y = reshape_data(epoch_x)

            _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
        # get loss and error
        test_error, test_loss, vali_error, vali_loss, train_error, train_loss = get_error_and_loss(sess)
        # store loss and error
        train_error_list.append(train_error)
        test_error_list.append(test_error)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        # early stop
        if epoch % 25 == 24:
            if vali_loss < lowest_vali_loss:
                lowest_vali_error = vali_error
                print("lowest vali error = {}".format(lowest_vali_error))
                save_model(sess)
            
        print('{}/{}, test error = {:.6f}, test loss = {:.6f}, train error = {:.6f}, train loss = {:.6f}, {:.2f}'.format(epoch, hm_epochs, test_error, test_loss, train_error, train_loss, time.time() - t_ref))
    print('test error = {:.6f}, test loss = {:.6f}, train error = {:.6f}, train loss = {:.6f}'.format(test_error, test_loss, train_error, train_loss))
    
    # calculate loss and error, save if the loss if the lowest
    test_error, test_loss, vali_error, vali_loss, train_error, train_loss = get_error_and_loss(sess)
    if vali_loss < lowest_vali_loss:
        lowest_vali_error = vali_error
        print("lowest vali error = {}".format(lowest_vali_error))
        save_model(sess)
    # save learning curve
    save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list)