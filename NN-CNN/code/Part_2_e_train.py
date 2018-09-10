import numpy as np
import os
import math
import collections
import pickle
import os, sys, time
from tensorflow.examples.tutorials.mnist import input_data 
import numpy.matlib


# a function to save model

def save_model(class_dict):
    file_obj = open("../models/NP_CNN.obj","wb")
    pickle.dump(class_dict, file_obj)
    file_obj.close()
    print("model saved to {}".format('../models/NP_CNN.obj'))

# a function to load model

def load_model():
    file_obj = open("../models/NP_CNN.obj","rb")
    class_dict = pickle.load(file_obj)
    file_obj.close()
    print("loaded model {}".format('../models/NP_CNN.obj'))
    return class_dict

# a function to save learning curve such as error rate and loss for both training and test set

def save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list):
    np.save("../models/NP_CNN_train_error_list.npy", train_error_list)
    np.save("../models/NP_CNN_test_error_list.npy", test_error_list)
    np.save("../models/NP_CNN_train_loss_list.npy", train_loss_list)
    np.save("../models/NP_CNN_test_loss_list.npy", test_loss_list)
    print("data saved to {}".format('../models/'))

# a robust log function
def robust_log(input_arg):
    if input_arg >= math.pow(10,-300):
        return math.log(input_arg)
    else:
        return -70000

# define some class for CNN
    
# a class for linear later
class linear_layer():
    #Initialization
    def __init__(self,num_input,num_node):
        # num_input: dimension of the imput data
        # num_node: nodes of the current layer
        self.num_input = num_input
        self.num_node = num_node
        b = 0.01*np.ones((1,num_node))
        W_initial = np.random.normal(size=[num_input,num_node])
        self.W = np.concatenate((W_initial,b),axis=0)
        self.dldW = collections.defaultdict(float)
        self.x = collections.defaultdict(float)
        self.y = collections.defaultdict(float)
        self.dldb = collections.defaultdict(float)

    #forward pass
    def forward(self,x):
        x = np.asarray(x)
        xstar = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
        y = np.matmul(xstar,self.W)
        self.x = x
        self.y = y

        return y

    #backward pass
    def backward(self,dLdy):
        #dydx is the transpose(W)
        dydx = self.W
        #throw away the bias
        dydx = dydx[0:-1][:]
        #transpose to (n_node x n_input) size
        dydx = np.transpose(dydx)
        #compute dLdx to back propogate
        dLdx = np.matmul(dLdy,dydx)
        #Compute dldW to optimize
        dldW = np.matmul(np.transpose(self.x),dLdy)
        self.dldW = dldW
        self.dldb = np.matmul(np.ones((1,self.x.shape[0])),dLdy)

        return dLdx

    def update_func(self,learning_rate):
        self.W[:-1] = self.W[:-1] - learning_rate*self.dldW
        self.W[-1] = self.W[-1] - learning_rate*self.dldb


#RELU class is a class build for non-linearity for both forward and backward passing

class relu_layer():
    #Initialization
    def __init__(self,num_relu_nodes):
        self.num_relu_nodes = num_relu_nodes
        self.x = collections.defaultdict(float)
    
    def forward(self,x):
        x_array = np.array(x)
        self.x = x_array
        compare_matrix = np.zeros(np.shape(x_array))
        y = np.maximum(x_array,compare_matrix)

        return y

    def backward(self,dLdy):
        dydx = self.x
        dydx[dydx>0] = 1
        dydx[dydx<0] = 0
        #only propagate those which are selected
        dLdx = np.multiply(dLdy,dydx)

        return dLdx

#class for back propagation
class soft_max_cross_entropy_layer():
    def __init__(self,num_input,num_class):

        self.input = collections.defaultdict(float)
        self.target = np.zeros((1,num_class))
        self.output = np.zeros((1,num_class))
        self.loss = collections.defaultdict(float)

    #forward pass
    def forward(self,input,target):
        self.input = np.array(input)
        self.target = np.array(target)
        dim_max = np.reshape(np.amax(self.input,axis=1),[-1,1])
        input_adjusted = np.subtract(self.input,np.matlib.repmat(dim_max,1,self.input.shape[1]))
        self.output = self.softmax(input_adjusted)
        log_output = np.log(self.output+1e-10)
        self.loss = -np.mean(np.sum(np.multiply(self.target,log_output),axis=1),axis=0)

        return self.loss, self.output

    #backward passing
    def backward(self):
        dLdx = self.output - self.target

        return dLdx

    #softmax
    def softmax(self,input):
        np_input = np.array(input)
        num_dim = input.shape[1]
        exp_input = np.exp(np_input)
        norm_factor = np.matlib.repmat(np.reshape(np.sum(exp_input,axis=1),[-1,1]),1,num_dim)
        soft_max_output = np.divide(exp_input,norm_factor)

        return soft_max_output

#a function to calculate accuracy
def accuracy(y_pred, y_train):
    a = y_pred
    b = y_train
    return sum(np.equal(np.argmax(a, axis=1), np.argmax(b, axis=1)))/b.shape[0]


#class for convolution
class conv_layer_2d():
    # define the initialization function
    def __init__(self,num_kernel_size,num_stride,input_channel,output_channel,padding_mode):

        self.kernel_height = num_kernel_size[0]
        self.kernel_width = num_kernel_size[1]
        self.height_stride = num_stride[0]
        self.width_stride = num_stride[1]
        self.padding_mode = padding_mode
        self.input_height = 0
        self.input_width = 0
        self.input_height_padded = 0
        self.input_width_padded = 0
        self.input_channel = input_channel
        self.output_height = 0
        self.output_width = 0
        self.output_channel = output_channel
        # initialize weights
        self.weight = np.random.normal(size=[self.input_channel,self.kernel_height,self.kernel_width,self.output_channel])
        # for gradient variables
        self.dydw = 0
        self.dldw = 0
        self.dldx = 0
        # padding legnth
        self.pad_top = 0
        self.pad_bottom = 0
        self.pad_left = 0
        self.pad_right = 0

    def forward(self,data):

        assert len(data.shape) == 4
        num_data = data.shape[0]
        self.input_height = data.shape[1]
        self.input_width = data.shape[2]
        self.input_channel = data.shape[3]

        if self.padding_mode == 'SAME':     # similar to tensorflow, here allows different padding to different dimensions
            self.output_height = math.ceil((float)(self.input_height)/(float)(self.height_stride))
            self.output_width = math.ceil((float)(self.input_width)/(float)(self.width_stride))
        elif self.padding_mode == 'VALID':     # no padding for the 'VALID' option
            self.output_height = math.ceil((float)(self.input_height-self.kernel_height+1)/(float)(self.height_stride))
            self.output_width = math.ceil((float)(self.input_width-self.kernel_width+1)/(float)(self.width_stride))
        else:
            raise ValueError('The padding method cannot be recognized! Please check your codes!')

        # compute the padding length
        # height
        if(self.input_height%self.height_stride==0):
            pad_height = max(self.kernel_height-self.height_stride,0)
        else:
            pad_height = max(self.kernel_height - (self.input_height%self.height_stride), 0)
        # width
        if(self.input_width%self.width_stride==0):
            pad_width = max(self.kernel_width-self.width_stride,0)
        else:
            pad_width = max(self.kernel_width - (self.input_width%self.width_stride), 0)

        # compute the prev and after padding length and pad
        # height
        self.pad_top = pad_height//2
        self.pad_bottom = pad_height - self.pad_top
        # widths
        self.pad_left = pad_width//2
        self.pad_right = pad_width - self.pad_left
        # pad
        if self.padding_mode == 'SAME':
            data_padded = np.pad(data,((0,0),(self.pad_top,self.pad_bottom),(self.pad_left,self.pad_right),(0,0)),'constant',constant_values=0)
        else:
            data_padded = np.pad(data,((0,0),(0,0),(0,0),(0,0)),'constant',constant_values=0)   # to keep the formation the same
        self.input_height_padded = data_padded.shape[1]
        self.input_width_padded = data_padded.shape[2]
        # create the 6-dim tensor for forward computation 

        prod_tensor_inverse = np.zeros(shape=[self.output_channel,self.output_height,self.output_width, 
                                    self.input_height_padded,self.input_width_padded,self.input_channel])
        # get a dydw 6-dim tensor for backward computation 
        dydw_tensor_invere = np.zeros(shape=[self.output_height,self.output_width, num_data,
                                    self.kernel_height,self.kernel_width,self.input_channel])
        # get the transposed weight
        weight_trans = np.transpose(self.weight,(3,1,2,0))
        
        for conv_channel in range(self.output_channel):
            for conv_row_out in range(self.output_height):
                for conv_column_out in range(self.output_width):
                    this_start_row = conv_row_out*self.height_stride
                    this_end_row = this_start_row + self.kernel_height
                    this_start_col = conv_column_out*self.width_stride
                    this_end_col = this_start_col + self.kernel_width
                    prod_tensor_inverse[conv_channel,conv_row_out,conv_column_out,this_start_row:this_end_row,this_start_col:this_end_col] = weight_trans[conv_channel]
                    if conv_channel == 0:
                        dydw_tensor_invere[conv_row_out,conv_column_out] = data_padded[:,this_start_row:this_end_row,this_start_col:this_end_col,:]
        # store the dydw tensor for backwoard
        self.dydw = np.transpose(dydw_tensor_invere,(2,1,0,3,4,5))
        # store the dydx tensor
        self.dydx = np.transpose(prod_tensor_inverse,(0,2,1,3,4,5))
        # calculate the production tensor and perform tensor production to get the convolutional output
        prod_tensor = np.transpose(prod_tensor_inverse,(5,4,3,1,2,0))

        conv_data = np.tensordot(data_padded,prod_tensor,axes=([3,2,1],[0,1,2]))

        return conv_data

    def backward(self,dldy):
        num_data = dldy.shape[0]
        # transpose the dldy tensor for computing weight gradient
        dldy_inv_weight = np.transpose(dldy,(3,1,2,0))        
        # perform dldw computation and transpose to get the result
        dldw_inv  = np.tensordot(dldy_inv_weight,self.dydw,axes=([3,2,1],[0,1,2]))
        self.dldw = np.transpose(dldw_inv,(3,1,2,0))
        # perform dldx computation
        dldx_padded = np.tensordot(dldy,self.dydx,axes=([3,2,1],[0,1,2]))  

        self.dldx = dldx_padded[:,self.pad_top:self.pad_top+self.input_height,self.pad_left:self.pad_left+self.input_width,:]

        return self.dldx

    def update_func(self,learning_rate=1e-3):
        self.weight = self.weight - learning_rate*self.dldw

# class for max-pooling
class max_pooling_2d():

    def __init__(self,num_kernel_size,num_stride,padding_mode):
        self.kernel_height = num_kernel_size[0]
        self.kernel_width = num_kernel_size[1]
        self.height_stride = num_stride[0]
        self.width_stride = num_stride[1]
        self.padding_mode = padding_mode
        self.input_height = 0
        self.input_width = 0
        self.input_height_padded = 0
        self.input_width_padded = 0
        self.num_channel = 0
        # initialize gradient varaibles
        self.dydx = 0
        self.dldx = 0
        # initialize padding varaibles

        self.pad_top = 0
        self.pad_bottom = 0
        self.pad_left = 0
        self.pad_right = 0

    def forward(self,data):

        num_data = data.shape[0]
        self.input_height = data.shape[1]
        self.input_width = data.shape[2]
        self.num_channel = data.shape[3]
        # calculate the output size
        if self.padding_mode == 'SAME':     
            self.output_height = math.ceil((float)(self.input_height)/(float)(self.height_stride))
            self.output_width = math.ceil((float)(self.input_width)/(float)(self.width_stride))
        elif self.padding_mode == 'VALID':
            self.output_height = math.ceil((float)(self.input_height-self.kernel_height+1)/(float)(self.height_stride))
            self.output_width = math.ceil((float)(self.input_width-self.kernel_width+1)/(float)(self.width_stride))

        # calculate the padding length
        if(self.input_height%self.height_stride==0):
            pad_height = max(self.kernel_height-self.height_stride,0)
        else:
            pad_height = max(self.kernel_height - (self.input_height%self.height_stride), 0)

        if(self.input_width%self.width_stride==0):
            pad_width = max(self.kernel_width-self.width_stride,0)
        else:
            pad_width = max(self.kernel_width - (self.input_width%self.width_stride), 0)
        # pad length to apply
        self.pad_top = pad_height//2
        self.pad_bottom = pad_height - self.pad_top
        # padding the input data
        self.pad_left = pad_width//2
        self.pad_right = pad_width - self.pad_left
        # pad the data
        if self.padding_mode == 'SAME':
            data_padded = np.pad(data,((0,0),(self.pad_top,self.pad_bottom),(self.pad_left,self.pad_right),(0,0)),mode='constant',constant_values=0)
        else:
            data_padded = np.pad(data,((0,0),(0,0),(0,0),(0,0)),mode='constant',constant_values=0)
            
        
        self.input_height_padded = data_padded.shape[1]
        self.input_width_padded = data_padded.shape[2]
        # get the tensor of output for forward output
        inv_pooled_tensor = np.zeros(shape=[self.output_height,self.output_width,num_data,self.num_channel])
        # get the tensor of dydx_mask for backward compuatation
        dydx_mask_inv_tensor = np.zeros(shape=[self.output_height,self.output_width,num_data,self.num_channel,self.input_height_padded,self.input_width_padded])
        # calculate the pooling output tensor and the dydx tensor
        for cRow in range(self.output_height):
            for cCol in range(self.output_width):
                this_start_row = cRow*self.height_stride
                this_end_row = this_start_row + self.kernel_height
                this_start_col = cCol*self.width_stride
                this_end_col = this_start_col + self.kernel_width

                this_image_crop = data_padded[:,this_start_row:this_end_row,this_start_col:this_end_col,:]    # [num_data * kernel_height * kernel_width * num_channel]
                this_image_crop_inv = np.transpose(this_image_crop,(0,2,3,1))                                 # [num_data * num_channel * kernel_height * kernel_width]
                this_image_crop_inv_flatten = np.reshape(this_image_crop_inv,[num_data,self.num_channel,-1])        # [num_data * num_channel * (kernel_height * kernel_width)]
                inv_pooled_tensor[cRow,cCol] = np.amax(this_image_crop_inv_flatten,axis=2)        # [num_data * num_channel]

                ind_max_pixel = np.reshape(np.argmax(this_image_crop_inv_flatten,axis=2),[num_data,self.num_channel,1])

                this_one_hot_mask = np.zeros(shape=this_image_crop_inv_flatten.shape)
                this_one_hot_mask[:,:,ind_max_pixel] = 1
                this_one_hot_mask = np.reshape(this_one_hot_mask,[num_data,self.num_channel,self.kernel_height,self.kernel_width])   # [num_data * num_channel * kernel_height * kernel_width]

                dydx_mask_inv_tensor[cRow,cCol,:,:,this_start_row:this_end_row,this_start_col:this_end_col] = this_one_hot_mask[:]
        # store the dydx
        self.dydx = np.transpose(dydx_mask_inv_tensor,(2,0,1,4,5,3))
        # output the pooled tensor
        pooling_output = np.transpose(inv_pooled_tensor,(2,0,1,3))

        return pooling_output

    def backward(self,dldy):
        assert len(dldy.shape)==4
        num_data = dldy.shape[0]
        # first proceed dldy to a 6-dim tensor as dydx
        dldy_compute_inv = np.zeros(shape=[self.input_height_padded,self.input_width_padded,num_data,self.output_height,self.output_width,self.num_channel])
        dldy_compute_inv[:,:] = dldy
        dldy_compute = np.transpose(dldy_compute_inv,(2,3,4,0,1,5))     # [num_data * nHeight_out * nWidth_out * nHeight_padded * nWidth_padded * num_channel]

        dldx_tensor = np.einsum('...,...->...',dldy_compute,self.dydx)
        # sum-up to the dldx output
        self.dydx = np.sum(np.sum(dldx_tensor,axis=1),axis=1)

        return self.dydx

#calculate training errors
def get_error_loss_and_pred(inter_data, label):
    # predict and calculate the training and testing error, and loss
    for conv_layer in range(len(conv_layer_list)):
        inter_data = conv_layer_list[conv_layer].forward(inter_data)
        if (conv_layer+1) in pool_operationum_layers:
                this_pooling_ind = pool_operationum_layers.index(conv_layer+1)
                inter_data = pool_layer_list[this_pooling_ind].forward(inter_data)
    
    batch_image_out_shape = inter_data.shape
    
    inter_data = np.reshape(inter_data,[-1,flattenum_size])
    inter_data = linear_layer_1.forward(inter_data)
    inter_data = nonum_linear_layer_1.forward(inter_data)
    z_out = linear_layer_2.forward(inter_data)
    
    
    loss, softmax_predict = loss_layer.forward(z_out,label)
    error = 1 - accuracy(softmax_predict, label)
    return error, loss, softmax_predict


# import data and reshape if needed

data_dir = '../data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label = mnist.test.labels

train_image_feed = np.reshape(train_image,[-1,28,28,1])
test_image_feed = np.reshape(test_image,[-1,28,28,1])

num_data_train = train_image_feed.shape[0]
num_data_test = test_image_feed.shape[0]


conv_kernel_size_list = [(3,3),(3,3)]
# conv_stride_size_list = [(1,1),(1,1)]
conv_stride_size_list = [(2,2),(2,2)]
# pool_operationum_layers = [1,2]
pool_operationum_layers = []

pooling_kernel_size_list = [(2,2),(2,2)]
pooling_stride_size_list = [(2,2),(2,2)]
image_channels = [1,3,9]
flattenum_size = 7*7*image_channels[-1]
num_linear_node = 100
num_class = 10

#define some paramters
batch_size = 32
learning_rate = 1e-3
cnnum_learning_rate = 1e-4



# define model
conv_layer_list = []
pool_layer_list = []

# define convolutional layers and pooling layers using list
conv_layer_1 = conv_layer_2d(num_kernel_size=conv_kernel_size_list[0],
                                      num_stride=conv_stride_size_list[0],
                                      input_channel=image_channels[0],
                                      output_channel=image_channels[1],
                                      padding_mode='SAME')
conv_layer_list.append(conv_layer_1)

pool_layer_1 = max_pooling_2d(num_kernel_size=pooling_kernel_size_list[0],
                              num_stride=pooling_stride_size_list[0],
                              padding_mode='SAME')

pool_layer_list.append(pool_layer_1)

conv_layer_2 = conv_layer_2d(num_kernel_size=conv_kernel_size_list[1],
                                      num_stride=conv_stride_size_list[1],
                                      input_channel=image_channels[1],
                                      output_channel=image_channels[2],
                                      padding_mode='SAME')
conv_layer_list.append(conv_layer_2)

pool_layer_2 = max_pooling_2d(num_kernel_size=pooling_kernel_size_list[1],
                              num_stride=pooling_stride_size_list[1],
                              padding_mode='SAME')

pool_layer_list.append(pool_layer_2)

# define some linear and nonlinear layers
linear_layer_1 = linear_layer(flattenum_size,num_linear_node)
nonum_linear_layer_1 = relu_layer(num_linear_node)
linear_layer_2 = linear_layer(num_linear_node,num_class)
loss_layer = soft_max_cross_entropy_layer(num_class,num_class)


# lists to store learning curves
train_error_list = []
train_loss_list = []
test_error_list = []
test_loss_list = []

hm_epochs = 200

# training
for epoch in range(hm_epochs):
    t_ref = time.time()
    current_epoch_loss = 0
    # random select some data
    train_ind_rand = np.random.choice(num_data_train,size=num_data_train)
    for i in range(1000):
        print("\r{}, {}".format(i,1000), end="")
        # looping the random data for training, the random dataset was large enough so basicly the same idea as random batch.
        this_train_ind = train_ind_rand[i*batch_size:(i+1)*batch_size]
        this_train_image = train_image_feed[this_train_ind]
        this_train_label = train_label[this_train_ind]
        inter_data = this_train_image[:]
        
        #forward propagations
        for conv_layer in range(len(conv_layer_list)):
            inter_data = conv_layer_list[conv_layer].forward(inter_data)
            if (conv_layer+1) in pool_operationum_layers:
                this_pooling_ind = pool_operationum_layers.index(conv_layer+1)
                inter_data = pool_layer_list[this_pooling_ind].forward(inter_data)

        batch_image_out_shape = inter_data.shape

        inter_data = np.reshape(inter_data,[-1,flattenum_size])
        inter_data = linear_layer_1.forward(inter_data)
        inter_data = nonum_linear_layer_1.forward(inter_data)
        z_out = linear_layer_2.forward(inter_data)

        this_loss, _ = loss_layer.forward(z_out,this_train_label)
        current_epoch_loss += this_loss
        
        #backward propagations
        dldy = loss_layer.backward()

        
        dldy = linear_layer_2.backward(dldy)
        linear_layer_2.update_func(learning_rate=learning_rate)
        dldy = nonum_linear_layer_1.backward(dldy)

        dldy = linear_layer_1.backward(dldy)
        linear_layer_1.update_func(learning_rate=learning_rate)

        dldy = np.reshape(dldy,batch_image_out_shape)
        for conv_layer in range(len(conv_layer_list)):

            this_layer_number = len(conv_layer_list) - conv_layer
            if this_layer_number in pool_operationum_layers:
                this_pooling_ind = pool_operationum_layers.index(this_layer_number)
                dldy = pool_layer_list[this_pooling_ind].backward(dldy)

            conv_input = -(conv_layer + 1)
            dldy = conv_layer_list[conv_input].backward(dldy)
            conv_layer_list[conv_input].update_func(learning_rate=cnnum_learning_rate)

    # calculate error rate and loss for both training and testing set
    train_error, train_loss, y_train_pred = get_error_loss_and_pred(train_image_feed, train_label)
    test_error, test_loss, y_test_pred = get_error_loss_and_pred(test_image_feed, test_label)
    # store the learning curve
    train_error_list.append(train_error)
    test_error_list.append(test_error)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    print("\r{}/{}, train error = {:.6f}, test error = {:.6f}, train_loss = {:.6f}, test loss = {:.6f}, time = {:.2f}".format(epoch, hm_epochs, train_error, test_error, train_loss, test_loss, time.time() - t_ref))    

# a dictionary to store the objects for recovery    
class_dict = {}
class_dict["conv_layer_list"] = conv_layer_list
class_dict["pool_layer_list"] = pool_layer_list
class_dict["linear_layer_1"] = linear_layer_1
class_dict["nonum_linear_layer_1"] = nonum_linear_layer_1
class_dict["linear_layer_2"] = linear_layer_2
class_dict["loss_layer"] = loss_layer

# save model and learning curve
save_model(class_dict)
save_learning_curve(train_error_list, test_error_list, train_loss_list, test_loss_list)


