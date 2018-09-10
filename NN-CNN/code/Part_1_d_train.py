from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time

if not os.path.exists('../models/'):
    os.mkdir('../models/')
    
# a function to save model
def save_model(sess):
    tf.train.Saver().save(sess, '../models/TF_CNN.checkpoint')
    print("model saved to {}".format('../models/TF_CNN.checkpoint'))
    
    
# a function to save learning curve such as error rate and loss for both training and test set

def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    np.save("../models/TF_CNN_train_error_list.npy", train_error_list)
    np.save("../models/TF_CNN_test_error_list.npy", test_error_list)
    np.save("../models/TF_CNN_train_loss_list.npy", train_loss_list)
    np.save("../models/TF_CNN_test_loss_list.npy", test_loss_list)
    print("data saved to {}".format('../models/'))

    
# import data and reshape data if needed

data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels

X_vali = mnist.validation.images
y_vali = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

hm_epochs = int(sys.argv[1])

#define a place holder for x and y

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# define some function for model

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# define the network
# define some weights

W_conv1 = weights([3, 3, 1, 32])
b_conv1 = biases([32])

X_train_reshape = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(X_train_reshape, W_conv1) + b_conv1)
h_pool1 = max_pooling(h_conv1)

W_conv2 = weights([3, 3, 32, 64])
b_conv2 = biases([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pooling(h_conv2)

W_fc1 = weights([7 * 7 * 64, 1024])
b_fc1 = biases([1024])

flatted_feature = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(flatted_feature, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weights([1024, 10])
b_fc2 = biases([10])

y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# define the cross entropy loss

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# define accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# define some list to store learning curves
train_error_list = []
train_loss_list = []
test_error_list = []
test_loss_list = []

# a function to calculate error rate and loss for training and testing dataset

def get_error_and_loss(sess):
     # the data are splited into pieces to avoid OOM
    vali_error = []
    vali_loss = []
    # evaluate validation set
    for num in range(5):
        vali_loss.append(sess.run(loss, feed_dict={x: X_vali[num*1000:(num+1)*1000], y: y_vali[num*1000:(num+1)*1000], keep_prob: 1.0}))
        vali_error.append(1 - accuracy.eval({x: X_vali[num*1000:(num+1)*1000], y: y_vali[num*1000:(num+1)*1000], keep_prob: 1.0}))
        
    vali_error = np.mean(vali_error)
    vali_loss = np.mean(vali_loss)
    
    # evaluate testing set
    test_error = []
    test_loss = []
    
    for num in range(10):
        test_loss.append(sess.run(loss, feed_dict={x: X_test[num*1000:(num+1)*1000], y: y_test[num*1000:(num+1)*1000], keep_prob: 1.0}))
        test_error.append(1 - accuracy.eval({x: X_test[num*1000:(num+1)*1000], y: y_test[num*1000:(num+1)*1000], keep_prob: 1.0}))
        
    test_error = np.mean(test_error)
    test_loss = np.mean(test_loss)
    
    # evaluate training set
    train_error = []
    train_loss = []
    
    for num in range(55):
        train_loss.append(sess.run(loss, feed_dict={x: X_train[num*1000:(num+1)*1000], y: y_train[num*1000:(num+1)*1000], keep_prob: 1.0}))
        train_error.append(1 - accuracy.eval({x: X_train[num*1000:(num+1)*1000], y: y_train[num*1000:(num+1)*1000], keep_prob: 1.0}))
        
    train_error = np.mean(train_error)
    train_loss = np.mean(train_loss)
    
    return test_error, test_loss, vali_error, vali_loss, train_error, train_loss


# training
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(hm_epochs):
        for i in range(120):
            # get batch data
            batch_X_train, batch_y_train = mnist.train.next_batch(128)
            # update model to minimise loss
            sess.run(train_step, feed_dict={x: batch_X_train, y: batch_y_train, keep_prob: 0.5})
        # evaluate the error rate and loss and store in the list
        test_error, test_loss, vali_error, vali_loss, train_error, train_loss = get_error_and_loss(sess)
        train_error_list.append(train_error)
        test_error_list.append(test_error)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print("{}/{}, train error = {:.6f}, test error = {:.6f},. train_loss = {:.6f}, test loss = {:.6f}".format(epoch, hm_epochs, train_error, test_error, train_loss, test_loss))
    # saving model and data
    save_model(sess)
    save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list)
