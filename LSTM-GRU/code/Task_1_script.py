
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
    saver.save(sess, '../models/RNN_{}_{}.checkpoint'.format(unit_name, mode))
    print('model saved to ../models/RNN_{}_{}.checkpoint'.format(unit_name, mode))
    
def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    # a function to save learning curves
    if not os.path.exists('../models/'):
        os.mkdir('../models/')
    np.save("../models/RNN_{}_{}_train_error_list.npy".format(unit_name, mode), train_error_list)
    np.save("../models/RNN_{}_{}_test_error_list.npy".format(unit_name, mode), test_error_list)
    np.save("../models/RNN_{}_{}_train_loss_list.npy".format(unit_name, mode), train_loss_list)
    np.save("../models/RNN_{}_{}_test_loss_list.npy".format(unit_name, mode), test_loss_list)


# a function to binarize data
def binarize(data, threshold=0.1):
    data = (threshold < data).astype('float32')
    return data

# define some hyperparamters
hm_epochs = 1000
n_classes = 10
batch_size = 1024
chunk_size = 1
n_chunks = 784

# read the data
data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# binarize the data
X_train = binarize(mnist.train.images)
y_train = mnist.train.labels

X_vali = binarize(mnist.validation.images)
y_vali = mnist.validation.labels

X_test = binarize(mnist.test.images)
y_test = mnist.test.labels

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

# define the recurrent neural network
def recurrent_neural_network(x):
    # define some weight for RNN
    before_layer = {'weights': tf.Variable(tf.random_normal([chunk_size,rnn_size])),
             'biases': tf.Variable(tf.random_normal([rnn_size]))}

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size,rnn_size])),
             'biases': tf.Variable(tf.random_normal([rnn_size]))}

    final_output_layer = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    # reshape data
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    before = tf.matmul(x, before_layer['weights']) + before_layer['biases']

    linear_x = tf.split(before, n_chunks, 0)
    # define RNN cells by the system input arguments
    if unit_name == "GRU":
        RNN_cell = tf.contrib.rnn.GRUCell(rnn_size)
    elif unit_name == "LSTM":
        RNN_cell = tf.contrib.rnn.LSTMCell(rnn_size)

    if mode == "stack_32":
        stack_rnn = []
        for i in range(3):
            stack_rnn.append(RNN_cell)
        RNN_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn)


    outputs, states = tf.contrib.rnn.static_rnn(RNN_cell, linear_x, dtype=tf.float32)
    # get the output by the weights
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    output = tf.nn.relu(output)
    final_output = tf.matmul(output, final_output_layer['weights']) + final_output_layer['biases']

    return final_output

# define prediction, loss, optimizer and accuracy
prediction = recurrent_neural_network(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(loss)
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


# this is a function used to calculate loss and error for all data set
def get_error_and_loss(sess):
    
    vali_error = []
    vali_loss = []
    
    for num in range(5):
        vali_loss.append(sess.run(loss, feed_dict={x: X_vali[num*1000:(num+1)*1000].reshape((-1, n_chunks, chunk_size)), y: y_vali[num*1000:(num+1)*1000]}))
        vali_error.append(1 - accuracy.eval({x: X_vali[num*1000:(num+1)*1000].reshape((-1, n_chunks, chunk_size)), y: y_vali[num*1000:(num+1)*1000]}))
        
    vali_error = np.mean(vali_error)
    vali_loss = np.mean(vali_loss)
    
    
    test_error = []
    test_loss = []
    
    for num in range(10):
        test_loss.append(sess.run(loss, feed_dict={x: X_test[num*1000:(num+1)*1000].reshape((-1, n_chunks, chunk_size)), y: y_test[num*1000:(num+1)*1000]}))
        test_error.append(1 - accuracy.eval({x: X_test[num*1000:(num+1)*1000].reshape((-1, n_chunks, chunk_size)), y: y_test[num*1000:(num+1)*1000]}))
        
    test_error = np.mean(test_error)
    test_loss = np.mean(test_loss)
    
    
    train_error = []
    train_loss = []
    
    for num in range(55):
        train_loss.append(sess.run(loss, feed_dict={x: X_train[num*1000:(num+1)*1000].reshape((-1, n_chunks, chunk_size)), y: y_train[num*1000:(num+1)*1000]}))
        train_error.append(1 - accuracy.eval({x: X_train[num*1000:(num+1)*1000].reshape((-1, n_chunks, chunk_size)), y: y_train[num*1000:(num+1)*1000]}))
        
    train_error = np.mean(train_error)
    train_loss = np.mean(train_loss)
    
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
        for epo in range(50):
            epoch_num = np.random.choice(len(X_train), batch_size, replace=False)
            epoch_x = X_train[epoch_num]
            epoch_y = y_train[epoch_num]

            epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

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
        if epoch % 50 == 49:
            if vali_loss < lowest_vali_loss:
                lowest_vali_error = vali_error
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