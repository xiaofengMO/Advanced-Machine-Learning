from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time

if not os.path.exists('../models/'):
    os.mkdir('../models/')

# a function to save model
def save_model(sess):
    tf.train.Saver().save(sess, '../models/TF_NN_256.checkpoint')
    print("model saved to {}".format('../models/TF_NN_256.checkpoint'))
    
    
# a function to save learning curve such as error rate and loss for both training and test set
def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    np.save("../models/TF_NN_256_train_error_list.npy", train_error_list)
    np.save("../models/TF_NN_256_test_error_list.npy", test_error_list)
    np.save("../models/TF_NN_256_train_loss_list.npy", train_loss_list)
    np.save("../models/TF_NN_256_test_loss_list.npy", test_loss_list)
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

# define the network
#define a place holder for x and y

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# define some weights

W_1 = tf.Variable(tf.random_normal([784, 256]))
b_1 = tf.Variable(tf.random_normal([256]))
y_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([256, 256]))
b_2 = tf.Variable(tf.random_normal([256]))
y_2 = tf.nn.relu(tf.matmul(x, W_1) + b_1)


W_3 = tf.Variable(tf.random_normal([256, 10]))
b_3 = tf.Variable(tf.random_normal([10]))

y_pred = tf.matmul(y_2, W_3) + b_3
# define the cross entropy loss

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# define accuracy

corrects = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))


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
        vali_loss.append(sess.run(loss, feed_dict={x: X_vali[num*1000:(num+1)*1000], y: y_vali[num*1000:(num+1)*1000]}))
        vali_error.append(1 - accuracy.eval({x: X_vali[num*1000:(num+1)*1000], y: y_vali[num*1000:(num+1)*1000]}))
        
    vali_error = np.mean(vali_error)
    vali_loss = np.mean(vali_loss)
    
    # evaluate testing set
    test_error = []
    test_loss = []
    
    for num in range(10):
        test_loss.append(sess.run(loss, feed_dict={x: X_test[num*1000:(num+1)*1000], y: y_test[num*1000:(num+1)*1000]}))
        test_error.append(1 - accuracy.eval({x: X_test[num*1000:(num+1)*1000], y: y_test[num*1000:(num+1)*1000]}))
        
    test_error = np.mean(test_error)
    test_loss = np.mean(test_loss)
    # evaluate training set
    
    train_error = []
    train_loss = []
    
    for num in range(55):
        train_loss.append(sess.run(loss, feed_dict={x: X_train[num*1000:(num+1)*1000], y: y_train[num*1000:(num+1)*1000]}))
        train_error.append(1 - accuracy.eval({x: X_train[num*1000:(num+1)*1000], y: y_train[num*1000:(num+1)*1000]}))
        
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
            sess.run(train_step, feed_dict={x: batch_X_train, y: batch_y_train})
            
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
