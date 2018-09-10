from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
import os, sys, time
# import dataset with one-hot class encoding
data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# a function to save model
def save_model(numy_data):
    np.save('../models/NP_Linear.npy', numy_data)
    print("model saved to {}".format('../models/NP_Linear.npy'))
    
# a function to save learning curve such as error rate and loss for both training and test set
def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    np.save("../models/NP_Linear_train_error_list.npy", train_error_list)
    np.save("../models/NP_Linear_test_error_list.npy", test_error_list)
    np.save("../models/NP_Linear_train_loss_list.npy", train_loss_list)
    np.save("../models/NP_Linear_test_loss_list.npy", test_loss_list)
    print("data saved to {}".format('../models/'))

# define some function for model
def add_bias_term(a):
    b = np.ones((a.shape[0],a.shape[1]+1))
    b[:,:-1] = a
    return b

# a softmax function
def softmax(a):
    a = np.exp(a)
    sum_a = sum(a.T)
    a = np.divide(a.T, sum_a).T
    return a

# a function to calculate accuracy

def accuracy(logits, labels):
    return np.sum(np.equal(np.argmax(logits, 1),np.argmax(labels, 1)))/len(np.argmax(logits, 1))

# a function to predict using the neural network
def predict(X, W):
    return np.matmul(X, W)

# a function to calculate cross entropy loss

def cross_entropy(y_pred, y_train):
    a = softmax(y_train)*np.log(softmax(y_pred)) + softmax(1-y_train)*np.log(1-softmax(y_pred))
    return (-sum(sum(a)))/len(y_train)

# define data and reshape if needed
X_train = mnist.train.images
X_train = add_bias_term(X_train)
y_train = mnist.train.labels

X_vali = mnist.validation.images
X_vali = add_bias_term(X_vali)
y_vali = mnist.validation.labels

X_test = mnist.test.images
X_test = add_bias_term(X_test)
y_test = mnist.test.labels

hm_epochs = int(sys.argv[1])

# define the weights, and some hyperparameters
W = np.random.normal(0,0.1, (785,10))
batch_size = 128
lr = 0.001

# define some list to store learning curves
train_error_list = []
train_loss_list = []
test_error_list = []
test_loss_list = []


for epoch in range(hm_epochs):
    for i in range(400):
        # random select from training set
        epoch_num = np.random.choice(len(X_train), batch_size, replace=False)
        epoch_x = X_train[epoch_num]
        epoch_y = y_train[epoch_num]

        # predict using the current weight
        epoch_y_pred = predict(epoch_x, W)
            
        # calculate the error from te prediction and the true labels
        error = softmax(epoch_y_pred) - softmax(epoch_y)
        
        # use the error to calculate a gradient for weights
        dW = np.matmul(epoch_x.T, error)
        
        # learning rate decrease every training step
        lr *= 0.99995
        
        # update weight according to the gradient to minimise loss
        W = W - lr * dW
    
    
    # after some training steps evaluate the training and testing error and loss
    y_train_pred = predict(X_train, W)
    y_test_pred = predict(X_test, W)

    train_error = 1 - accuracy(y_train_pred, y_train)
    test_error = 1 - accuracy(y_test_pred, y_test)

    train_loss = cross_entropy(y_train_pred, y_train)
    test_loss = cross_entropy(y_test_pred, y_test)

    # store the learning curves in list
    train_error_list.append(train_error)
    test_error_list.append(test_error)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    

    print("{}/{}, train error = {:.6f}, test error = {:.6f},. train_loss = {:.6f}, test loss = {:.6f}".format(epoch, hm_epochs, train_error, test_error, train_loss, test_loss))

# save the model and data after training
save_model(W)
save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list)
    
    
