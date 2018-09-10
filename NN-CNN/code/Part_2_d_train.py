import numpy as np
from numpy.linalg import inv
import sys
import random
import time
from tensorflow.examples.tutorials.mnist import input_data
import os, sys, time
import matplotlib.pyplot as plt

data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# a function to save model

def save_model(numy_data):
    np.save('../models/NP_NN_256.npy', numy_data)
    print("model saved to {}".format('../models/NP_NN_256.npy'))
    
# a function to save learning curve such as error rate and loss for both training and test set

def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    np.save("../models/NP_NN_256_train_error_list.npy", train_error_list)
    np.save("../models/NP_NN_256_test_error_list.npy", test_error_list)
    np.save("../models/NP_NN_256_train_loss_list.npy", train_loss_list)
    np.save("../models/NP_NN_256_test_loss_list.npy", test_loss_list)
    print("data saved to {}".format('../models/'))
    
    
# define data and reshape if needed

X_train = mnist.train.images
y_train = mnist.train.labels

X_validation = mnist.validation.images
y_validation = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

# define some function for model

# a softmax function

def softmax(a):
    a = np.exp(a)
    sum_a = sum(a.T)
    a = np.divide(a.T, sum_a).T
    return a

# a RELU function

def relu(z):
    g = np.maximum(0, z)
    return g

# a function to calculate function for RELU

def relu_gradient(z):
    g = (z >= 0).astype(int)
    return g

# a function to calculate cross entropy loss

def cross_entropy(y_pred, y_train):
    a = softmax(y_train)*np.log(softmax(y_pred)) + softmax(1-y_train)*np.log(1-softmax(y_pred))
    return (-sum(sum(a)))/len(y_train)

# a function to calculate accuracy

def accuracy(y_pred, y_train):
    a = softmax(y_pred)
    b = softmax(y_train)
    return sum(np.equal(np.argmax(a, axis=1), np.argmax(b, axis=1)))/b.shape[0]

# a function to initialize the weight randomly, a bias term was added

def random_init(L_in, L_out, c=0.12):
    return (np.random.rand(L_out, 1 + L_in) * 2 * c) - c

# a function to predict using the neural network
def predict(X_train, Theta):
    #forward propagate using weights Theta
    a1 = np.append(np.ones((X_train.shape[0], 1)), X_train, axis=1)
    z2 = np.matmul(a1, Theta[0].T)
    a2 = relu(z2)

    z3 = np.matmul(np.append(np.ones((X_train.shape[0], 1)), a2, axis=1), Theta[1].T)
    a3 = relu(z3)

    z4 = np.matmul(np.append(np.ones((X_train.shape[0], 1)), a3, axis=1), Theta[2].T)
#     a4 = softmax(z4)
    return z4

# a function to calculate neural network gradient

def calc_gradient(X_train, y_train, Theta, lambda_v, m):
    # forward propagate, notice there is a bias term added to training set
    a1 = np.append(np.ones((X_train.shape[0], 1)), X_train, axis=1)
    z2 = np.matmul(a1, Theta[0].T)
    a2 = relu(z2)

    z3 = np.matmul(np.append(np.ones((X_train.shape[0], 1)), a2, axis=1), Theta[1].T)
    a3 = relu(z3)

    z4 = np.matmul(np.append(np.ones((X_train.shape[0], 1)), a3, axis=1), Theta[2].T)
    a4 = softmax(z4)
    
    # calculate error
    delta_4 = (a4 - softmax(y_train)).T
    
    #backward propagate
    delta_3 = (np.matmul(Theta[2].T, delta_4).T * relu_gradient(np.append(np.ones((z3.shape[0], 1)), z3, axis=1)))[:,1:]
    delta_2 = (np.matmul(Theta[1].T, delta_3.T).T * relu_gradient(np.append(np.ones((z2.shape[0], 1)), z2, axis=1)))[:,1:]

    D_1 = np.matmul(delta_2.T, a1)
    D_2 = np.matmul(delta_3.T, np.append(np.ones((m, 1)), a2, axis=1))
    D_3 = np.matmul(delta_4, np.append(np.ones((m, 1)), a3, axis=1))

    T1 = Theta[0][:, 1:]
    T2 = Theta[1][:, 1:]
    T3 = Theta[2][:, 1:]
    
    # change gradient accordingly
    Theta1_grad = (1/m) * D_1 + (lambda_v/m) * np.append(np.zeros((Theta[0].shape[0], 1)), T1, axis=1)
    Theta2_grad = (1/m) * D_2 + (lambda_v/m) * np.append(np.zeros((Theta[1].shape[0], 1)), T2, axis=1)
    Theta3_grad = (1 / m) * D_3 + (lambda_v / m) * np.append(np.zeros((Theta[2].shape[0], 1)), T3, axis=1)
    return [Theta1_grad,Theta2_grad, Theta3_grad]




# define the weights, and some hyperparameters


all_layers = [X_train.shape[1]]

# define the model using two layer of 256 units

hidden_layer = [256, 256]

for i in hidden_layer:
    all_layers.append(i)

# append the layer using the output shape
all_layers.append(y_train.shape[1])

# get the input and output shapre

input_layer_size = X_train.shape[1]
num_labels = y_train.shape[1]

# Theta is the weight in neural network, which is randomly initialize for all inner layers
Theta = []
for i in range(0, len(all_layers)-1):
    Theta.append(random_init(all_layers[i], all_layers[i+1]))

# this is the regularization terms which is set to zero here, no use
lambda_v = 0

# set some parameters

batch_size = 128

# alpha here is the learning rate
alpha = 10

# total number of epoch are define by input argument
hm_epochs = int(sys.argv[1])


# define some list to store the learning curve
train_error_list = []
train_loss_list = []
test_error_list = []
test_loss_list = []

# training

for epoch in range(hm_epochs):
    for i in range(400):
        # random select from training set
        epoch_num = np.random.choice(len(X_train), batch_size, replace=False)
        epoch_x = X_train[epoch_num]
        epoch_y = y_train[epoch_num]
        
        # calculate gradient and change weights accordingly
        Theta_grad = calc_gradient(epoch_x, epoch_y, Theta, lambda_v, batch_size)
        for j in range(0, len(Theta)):
            Theta[j] -= alpha * (1 / batch_size) * Theta_grad[j]
            
        #learning rate decrease by every train step for model to converge
        alpha = alpha * 0.99999

    # get the accuracy and loss for both training and testing set after every fix training steps
    y_train_pred = predict(X_train, Theta)
    y_test_pred = predict(X_test, Theta)

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

# save model and learning curve

save_model(Theta)
save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list)
        