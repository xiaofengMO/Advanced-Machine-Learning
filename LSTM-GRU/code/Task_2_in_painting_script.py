#Import essential packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# specify model used by input arguments
unit_name = sys.argv[1]
mode = sys.argv[2]
pred_length = int(sys.argv[3])


if mode == "32":
    rnn_size = 32
elif mode == "64":
    rnn_size = 64
elif mode == "128":
    rnn_size = 128
elif mode == "stack_32":
    rnn_size = 32
    

# read the data
data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# a function to binarize data
def binarize(data, threshold=0.1):
    data = (threshold < data).astype('float32')
    
    return data

# binarize the data
X_train_original = binarize(mnist.train.images)
y_train_original = mnist.train.labels

X_vali_original = binarize(mnist.validation.images)
y_vali_original = mnist.validation.labels

X_test_original = binarize(mnist.test.images)
y_test_original = mnist.test.labels

# number of images
num_image_test = 100
# choose a random seed to fix the result
np.random.seed(2018) 

# get the random images
inpainting_select_ind = np.random.permutation(X_test_original.shape[0])[:num_image_test]
X_test_original = X_test_original[inpainting_select_ind]
y_test_original = y_test_original[inpainting_select_ind]
# mask the data
ground_truth_images = X_test_original.copy()
ground_truth_labels = y_test_original.copy()
masked_images = X_test_original.copy()
masked_images[:, -300:] = np.nan

# define some hyperparameters
num_images = masked_images.shape[0]
pixel_length = masked_images.shape[1]


n_classes = 2
batch_size = 512
prediction_start = 484
chunk_size = 1
num_chunks = 784
num_sample_image = 10



# define some place holder
x = tf.placeholder("float",[None,num_chunks,chunk_size])
y = tf.placeholder("float",[None,chunk_size])
# define the recurrent neural net in this cell

def recurrent_neural_network(x):
    # get the shape of x
    x_shape = tf.shape(x)[0]
    # define some weights for RNN
    rnn_to_pixel_layer = {'weight':tf.Variable(tf.random_normal([rnn_size,1])),\
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
        pix = tf.add(tf.matmul(outputs[this_pixel],rnn_to_pixel_layer['weight']),rnn_to_pixel_layer['bias'])
        pixels.append(pix)
    # concat the pixels and output
    final_pixels = tf.concat(pixels,axis=0)
    rnn_output = tf.reshape(final_pixels,[(num_chunks-1) * x_shape,-1])

    return rnn_output

# define the prediction
prediction = recurrent_neural_network(x)
prediction_eval = tf.nn.sigmoid(prediction) 
predict_result = tf.round(prediction_eval) 
# split for indexing
pixelwise_pred = tf.split(predict_result, num_or_size_splits=pixel_length-1, axis=0)

# allocate an array for reconstruction
random_reconstruction = np.zeros([num_image_test,num_sample_image,784])
image_uncertain_groudtruth = np.zeros([num_image_test,num_sample_image,784])
for this_sample in range(num_sample_image):
    random_reconstruction[:,this_sample,:prediction_start] = ground_truth_images[:,:prediction_start]
    image_uncertain_groudtruth[:,this_sample,:] = ground_truth_images[:]

# a functin to store data needed
def store_data(store_dict):
    if not os.path.exists('../models/'):
        os.mkdir('../models/')
    np.save("../models/inpainting_{}_{}_{}_store_dict.npy".format(unit_name, mode, pred_length), store_dict)
    print("saved to ../models/inpainting_{}_{}_{}_store_dict.npy".format(unit_name, mode, pred_length))
    
    
store_dict = {}

inpainting_loss_list = []
ground_truth_loss_list = []

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, '../models/pixel_{}_{}.checkpoint'.format(unit_name, mode))
    for i in range(pred_length):
        pixel_index = i + prediction_start - 1
        
        # define a inpainting and grund truth loss list for this prediction
        IP_loss_list = []
        GT_loss_list = []
        
        # start prediction
        for this_sample in range(num_sample_image):
            #print the process
            current_process = (i*(1/pred_length) + (this_sample+1)*(1/num_sample_image*(1/pred_length)))*100
            print('\rprocessing inpainting {:.2f} % finished'.format(current_process),end="",flush=True)
            
            # reshape data, get x and y for prediction
            this_input_pixel_x = np.reshape(random_reconstruction[:,this_sample,:],[-1, num_chunks, chunk_size])
            this_input_pixel_y = np.transpose(this_input_pixel_x,[1,0,2])   
            this_input_pixel_y = np.reshape(this_input_pixel_y,[-1, chunk_size])
            inpainting_dict = {x:this_input_pixel_x,y:this_input_pixel_y}
            # reshape data, get x and y for ground truth
            this_input_true_x = np.reshape(image_uncertain_groudtruth[:,this_sample,:],[-1, num_chunks, chunk_size])
            this_input_true_y = np.transpose(this_input_true_x,[1,0,2])    
            this_input_true_y = np.reshape(this_input_true_y,[-1,chunk_size])
            GT_dict = {x:this_input_true_x,y:this_input_true_y}
            # get the probability of the pixel
            
            # evaluate for inpainting and ground truth
            inpainting_prob = np.reshape(prediction_eval.eval(inpainting_dict),[num_chunks-1,num_image_test])
            this_inpainting_prob = inpainting_prob[pixel_index,:]
            Ground_truth_prob = np.reshape(prediction_eval.eval(GT_dict),[num_chunks-1,num_image_test]) 
            GT_pixel_prob = Ground_truth_prob[pixel_index,:]
            pred_GT_pixel_value = pixelwise_pred[pixel_index].eval(GT_dict)
            GT_pixel_value = image_uncertain_groudtruth[:,this_sample,pixel_index]

            # randomly matrix for random sample generation purposes
            random_matrix = np.random.uniform(low=0.0, high=1.0, size=this_inpainting_prob.shape)
            # get sample for this prediction subtract by the random matrix
            this_inpainting_pixel = this_inpainting_prob - random_matrix
            this_inpainting_pixel[this_inpainting_pixel >= 0] = 1.0
            this_inpainting_pixel[this_inpainting_pixel < 0] = 0.0

            random_reconstruction[:,this_sample,pixel_index+1] = this_inpainting_pixel[:]

            #calculate the cross-entropy loss for both inpainting ang ground truth
            inpainting_cross_entropy = -(np.multiply(this_inpainting_pixel, np.log(this_inpainting_prob)) + np.multiply((1 - this_inpainting_pixel), np.log(1 - this_inpainting_prob)))
            IP_loss_list.append(inpainting_cross_entropy)
            
            ground_truth_cross_entropy = -(np.multiply(GT_pixel_value, np.log(GT_pixel_prob)) + np.multiply((1 - GT_pixel_value), np.log(1 - GT_pixel_prob)))
            GT_loss_list.append(ground_truth_cross_entropy)
        # average the losses
        if i == 0:
            inpainting_loss_list = np.array(IP_loss_list)/pred_length
            ground_truth_loss_list = np.array(GT_loss_list)/pred_length
        else:
            inpainting_loss_list += np.array(IP_loss_list)/pred_length
            ground_truth_loss_list += np.array(GT_loss_list)/pred_length
        
    # save the loss for future uses
    inpainting_loss_list = np.array(inpainting_loss_list)
    ground_truth_loss_list = np.array(ground_truth_loss_list)
    store_dict["inpainting_loss_list"] = inpainting_loss_list
    store_dict["ground_truth_loss_list"] = ground_truth_loss_list
    store_dict["random_reconstruction"] = random_reconstruction
    store_dict["X_test_original"] = X_test_original
    store_dict["y_test_original"] = y_test_original

    store_data(store_dict)
