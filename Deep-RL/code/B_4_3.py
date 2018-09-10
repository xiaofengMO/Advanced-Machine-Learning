import gym
import random
from statistics import mean, stdev
import numpy as np
import collections
import tensorflow as tf
from collections import deque
# import scipy
# from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import os,sys,time
import math
import random
from XFLib_1 import *
import os
import subprocess
import re
import random

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# choose which game to run by system arguments
game_name = sys.argv[1]
# game_name = 'MsPacgirl'
# game_name = 'Boxing'

# print the game_name
print(game_name)

# get the file name for storing purposes
file_name = str(os.path.basename(__file__))[:-3]
# setup a path for this script by the file name and game name
path = "../models/{}/{}".format(file_name,game_name)
if not os.path.exists(path):
    os.makedirs(path)

# choose the size of buffer
buffer_size = 500000


# setup hyperparameter with respect to different games
if game_name == 'Pong':
    env = gym.make("Pong-v4")
    env_1 = gym.make("Pong-v4")
    num_of_actions = 6
    max_runs = 2000000
    Learning_rate = 1e-4 # 1e-7
    final_greedy_rate = 0.1
    discount = .99
    eval_start  = 10000
    greedy_stop_num = eval_start + 20
    greedy_stop_number = max_runs * 10
    train_start = 1
    render_flag = 0
    optimizer = "RMS"
    test_threshold = -18
    save_threshold = -21
elif game_name == 'MsPacgirl':
    env = gym.make("MsPacman-v4")
    env_1 = gym.make("MsPacman-v4")
    num_of_actions = 9
    max_runs = 2000000
    Learning_rate = 1e-4 # 1e-7
    final_greedy_rate = 0.1
    discount = .99
    eval_start  = 10000
    greedy_stop_num = eval_start + 20
    greedy_stop_number = max_runs * 10
    train_start = 1
    render_flag = 0
    optimizer = "RMS"
    test_threshold = 20
    save_threshold = 20
elif game_name == 'Boxing':
    env = gym.make("Boxing-v4")
    env_1 = gym.make("Boxing-v4")
    num_of_actions = 18
    max_runs = 2000000
    Learning_rate = 1e-4 # 1e-7
    final_greedy_rate = 0.1
    discount = .99
    eval_start  = 10000
    greedy_stop_num = eval_start + 20
    greedy_stop_number = max_runs * 10
    train_start = 1
    render_flag = 0
    optimizer = "RMS"
    test_threshold = -20
    save_threshold = -20
else:
    raise ValueError('Unidentified game mode')
    

# a function to save model
def save_model(sess, score):
    # make the path if ot exist
    if not os.path.exists(path):
        os.makedirs(path)
    # save model by the test performance, file name and game name, for early stop purposes
    save_name = "{}/{}-{}-*{}*{}".format(path,file_name,game_name,score,'.checkpoint')
    saver = tf.train.Saver()
    saver.save(sess, save_name)
    print("saved to {}".format(save_name))

def save_random_data(real_replay_buffer_N, real_replay_buffer_P, real_replay_buffer_Z, n=1, z=1, p=1):
    save_path = "./random_data/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if n:
        np.save("{}{}{}".format(save_path, game_name, "-N.npy"), list(real_replay_buffer_N))
        print("N saved")
    if z:
        np.save("{}{}{}".format(save_path, game_name, "-Z.npy"), list(real_replay_buffer_Z))
        print("Z saved")
    if p:
        np.save("{}{}{}".format(save_path, game_name, "-P.npy"), list(real_replay_buffer_P))
        print("P saved")

    print("random_data_saved to {}".format(save_path))
    
# a function to load relay buffers which is collected by random policy
def load_random_data(save_path = "./random_data/"):
    print("trying to recover buffer")
    real_replay_buffer_N = deque(maxlen=buffer_size)
    real_replay_buffer_P = deque(maxlen=buffer_size)
    real_replay_buffer_Z = deque(maxlen=buffer_size)
    
    try:
        N = np.load("{}{}{}".format(save_path, game_name, "-N.npy"))
        for i in N:
            real_replay_buffer_N.append(i)
        print("recover N sucess")
    except Exception as e:
        print("recover N fail " + str(e))
        pass
    
    try:
        P = np.load("{}{}{}".format(save_path, game_name, "-P.npy"))
        for i in P:
            real_replay_buffer_P.append(i)
        print("recover P sucess")
    except Exception as e:
        print("recover P fail " + str(e))
        pass
        
    try:
        Z = np.load("{}{}{}".format(save_path, game_name, "-Z.npy"))
        for i in Z:
            real_replay_buffer_Z.append(i)
        print("recover Z sucess")
    except Exception as e:
        print("recover Z fail " + str(e))
        pass
        
    
    return real_replay_buffer_N, real_replay_buffer_P, real_replay_buffer_Z

# a function to convert image from rgd to grey, since color is not useful, a grey image can simplify
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


print(test_threshold, save_threshold)


# some hyperparameters
loss_set = []
batch_size = 20
#hyperparameters for CNN
width = 6
height = 6
layer_1_unit = 16
layer_2_unit = 32

#size of the image
image_size = 28

# define tensorflow place holders
reward_place = tf.placeholder(tf.float32, [None])
obs_new = tf.placeholder(tf.float32, [None,image_size,image_size,4])
obs = tf.placeholder(tf.float32, [None,image_size,image_size,4])
act = tf.placeholder(tf.int32,[None])


# two dimensional convolutional
def convolution2D(DataIn,Weight):
    return tf.nn.conv2d(DataIn,Weight,strides = [1,2,2,1], padding = 'SAME')         

# a function to define the neural network model
def define_model(seed_num):
    tf.set_random_seed(seed_num)
    W1 = tf.Variable(tf.random_normal([width, height, 4, layer_1_unit]))
    b1 = tf.Variable(tf.random_normal([layer_1_unit]))

    y1 = tf.nn.relu(convolution2D(obs,W1)+b1)
    y1_new = tf.nn.relu(convolution2D(obs_new,W1)+b1)

    W2 = tf.Variable(tf.random_normal([width, height, layer_1_unit, layer_2_unit]))
    b2 = tf.Variable(tf.random_normal([layer_2_unit]))

    y2 = tf.nn.relu(convolution2D(y1,W2)+b2)
    y2_new = tf.nn.relu(convolution2D(y1_new,W2)+b2)

    # flatten the tensor before put into the fully connected layer
    flatten_shape = int(image_size/4)*int(image_size/4)*layer_2_unit
    W3 = tf.Variable(tf.random_normal([flatten_shape,256]))
    b3 = tf.Variable(tf.random_normal([256]))

    y3 = tf.nn.relu(tf.matmul(tf.reshape(y2,[-1,flatten_shape]),W3)+b3)
    y3_new = tf.nn.relu(tf.matmul(tf.reshape(y2_new,[-1,flatten_shape]),W3)+b3)

    W4 = tf.Variable(tf.random_normal([256,num_of_actions]))
    b4 = tf.Variable(tf.random_normal([num_of_actions]))

    y4 = tf.matmul(y3,W4) + b4
    y4_new = tf.matmul(y3_new,W4) + b4

    # target network to copy to
    W1_target = tf.Variable(tf.random_normal([width, height, 4, layer_1_unit]))
    b1_target = tf.Variable(tf.random_normal([layer_1_unit]))

    y1_target = tf.nn.relu(convolution2D(obs, W1_target) + b1_target)
    y1_new_target = tf.nn.relu(convolution2D(obs_new, W1_target) + b1_target)

    W2_target = tf.Variable(
        tf.random_normal([width, height, layer_1_unit, layer_2_unit]))
    b2_target = tf.Variable(tf.random_normal([layer_2_unit]))

    y2_target = tf.nn.relu(convolution2D(y1_target, W2_target) + b2_target)
    y2_new_target = tf.nn.relu(convolution2D(y1_new_target, W2_target) + b2_target)

    flatten_shape = int(image_size/4)*int(image_size/4)*layer_2_unit
    W3_target = tf.Variable(tf.random_normal([flatten_shape, 256]))
    b3_target = tf.Variable(tf.random_normal([256]))

    y3_target = tf.nn.relu(tf.matmul(tf.reshape(y2_target, [-1, flatten_shape]), W3_target) + b3_target)
    y3_new_target = tf.nn.relu(tf.matmul(tf.reshape(y2_new_target, [-1, flatten_shape]), W3_target) + b3_target)

    W4_target = tf.Variable(tf.random_normal([256, num_of_actions]))
    b4_target = tf.Variable(tf.random_normal([num_of_actions]))

    y4_target = tf.matmul(y3_target, W4_target) + b4_target
    y4_new_target = tf.matmul(y3_new_target, W4_target) + b4_target
    
    # global same variables for updating the target network
    global update_1, update_2, update_3, update_4, update_5, update_6, update_7, update_8

    # variables to update the network
    update_1 = W1_target.assign(W1)
    update_2 = b1_target.assign(b1)
    update_3 = W2_target.assign(W2)
    update_4 = b2_target.assign(b2)
    update_5 = W3_target.assign(W3)
    update_6 = b3_target.assign(b3)
    update_7 = W4_target.assign(W4)
    update_8 = b4_target.assign(b4)
    
    return y4,y4_new, y4_target, y4_new_target

# a function to update the target network
def update(sess):
    sess.run(update_1)
    sess.run(update_2)
    sess.run(update_3)
    sess.run(update_4)
    sess.run(update_5)
    sess.run(update_6)
    sess.run(update_7)
    sess.run(update_8)


def image_preprocess(image):
    # if the image is Pong, cut the image so that the noisy part do not affect the network
    if game_name == 'Pong':
        image = image[34:194,:]
    # process image to 28 by 28 grey image
    img = Image.fromarray(image, 'RGB').convert('L')
    img = img.resize((image_size,image_size),resample=Image.BILINEAR)
    image_trans = np.asarray(img, dtype=np.uint8)
    # binarize the image of pong to just two level for network to better understand the image
    if game_name == 'Pong':
        image_trans.setflags(write=True)
        image_trans[image_trans>90] = 255
        image_trans[image_trans<=90] = 0

    return image_trans

# a function to run the game with the current model for 100 times and average the performance
def test_process(sess, action_max):
    print("\nchecking performace")
    
    est = time_est(100)
    with sess.as_default():
        # lists to store the the episode length and returns
        score_list = []
        dis_rtn_list = []
        for i in range(100):
            old_obs = env_1.reset()
            t = 0
            frame_buffer = deque(maxlen=4)
            total_reward = 0
            discount_factor = 1
            discounted_value = 0
            count = 0
            observation_stack = []

            while 1:
                if count!= 0:
                    discount_factor *= 0.99
                # greedy policy
                if t > 3 and random.random() > final_greedy_rate:
                    action_dic = {obs:[old_obs],obs_new:np.zeros([1,image_size,image_size,4]),act:[0],reward_place:[reward]}
                    action = action_max.eval(action_dic)[0]
                else:
                    action = round(random.uniform(0, num_of_actions - 1))
                # take action and get the response from the action
                observation, reward, done, info = env_1.step(action)
                obs_frame = image_preprocess(observation)
                frame_buffer.append(obs_frame)

                if len(frame_buffer) == 4:
                    observation_stack = np.stack(list(frame_buffer),axis=0)
                    observation_stack = observation_stack.transpose([1,2,0])
                
                # reward either -1, 1 or 0
                reward = np.clip(reward,-1,1)
                total_reward += reward
                count += 1
                
                # save the observation to a variable
                old_obs = observation_stack
                t += 1
                # calculate the discounted returns
                discounted_value += discount_factor * reward

                if done:
                    # saving the reward and discouned rewards into list
                    score_list.append(total_reward)
                    dis_rtn_list.append(discounted_value)
                    break
            est.check()
    # return the mean of episode length and discounted rewards
    return np.mean(score_list), np.mean(dis_rtn_list)

# a function to collect samples undder random policy, positive, negative and zero rewards are stored in different array
# so when I do the training, there will be sufficient amount of data for different rewards.
def collect_random_func(num, LN, LZ, LP):
    P = []
    Z = []
    N = []
    score_list = []
    for i in range(10):
        old_obs = env.reset()
        t = 0
        frame_buffer = deque(maxlen=4)
        total_reward = 0
        discount_factor = 1
        discounted_value = 0
        count = 0
        observation_stack = []

        while 1:
            if count!= 0:
                discount_factor *= 0.99
            # random action
            action = round(random.uniform(0, num_of_actions - 1))
            # take action and get the response from the action
            observation, reward, done, info = env.step(action)
            obs_frame = image_preprocess(observation)
            frame_buffer.append(obs_frame)

            if len(frame_buffer) == 4:
                observation_stack = np.stack(list(frame_buffer),axis=0)
                observation_stack = observation_stack.transpose([1,2,0])
            
            # reward either -1, 1 or 0
            reward = np.clip(reward,-1,1)
            total_reward += reward
            count += 1
            # save the observation to a variable
            old_obs = observation_stack
            t += 1
            
            # calculate the discounted returns
            discounted_value += discount_factor * reward
            
            if t > 3:
                if reward == -1:
                    if LN < buffer_size:
                        N.append([old_obs,action,reward,observation_stack])
                elif reward == 1:
                    if LP < buffer_size:
                        P.append([old_obs,action,reward,observation_stack])
                elif reward == 0:
                    if LZ < buffer_size:
                        Z.append([old_obs,action,reward,observation_stack])
                else:
                    print("reward not true")
                    
            if done:
                score_list.append(total_reward)
                break

    return N, Z, P


# a class to check the current highest performance by looking into the saved models
# and to perform testing for evaluation.
class test_performance():
    # function to find the highest episode length in testing from the models
    def __init__(self):
        files = os.listdir(path)
        check_points = [float(i.split("*")[1]) for i in files if i.split("*")[-1] == ".checkpoint.data-00000-of-00001"]
        if len(check_points) > 0:
            self.highest_score = max(check_points)
        else:
            self.highest_score = -1000000
        print("\nhighest_score = {}\n".format(self.highest_score))
        
    # function to test the current models and return the performances
    def test(self, sess, action_max, assume_score):
        if 1:
            score_mean, ds_rtn_mean = test_process(sess, action_max)
            print("score_mean = {}, ds_rtn_mean = {}, final_score = {}".format(score_mean, ds_rtn_mean, score_mean))
            if score_mean >= save_threshold:
                save_model(sess, score_mean)
                if score_mean> self.highest_score:
                    self.highest_score = score_mean
                    print("\nhighest_score = {}\n".format(self.highest_score))
            return score_mean, ds_rtn_mean
        
# function for training
def training(seed_num):
    # define the greedy rate, initially 1, it can be changed in the training
    greedy_rate = 1
    
    # define the highes reward to be -21, it can be changed in the training
    highest_reward = -21
    
    # define the model
    y2, y2_new, y2_target, y2_target_new= define_model(seed_num)

    # get action by Q values
    action_max = tf.cast(tf.argmax(y2, axis=1), tf.int32)
    # calculate bellman loss
    data_amount = tf.shape(y2_new)[0]
    Q_old_index = tf.concat([tf.reshape(tf.range(0,limit=data_amount),[data_amount,1]), tf.reshape(act,[data_amount,1])],axis=1)
    old_q = tf.gather_nd(y2,Q_old_index)
    action_max_primary = tf.cast(tf.argmax(y2_new, axis=1), tf.int32)
    Q_new_index = tf.concat([tf.reshape(tf.range(0,limit=data_amount),[data_amount,1]), tf.reshape(action_max_primary,[data_amount,1])],axis=1)
    max_q_value_next = tf.gather_nd(y2_target_new, Q_new_index)
    change_in_q = (reward_place + discount * tf.stop_gradient(max_q_value_next) - old_q)
    loss = tf.reduce_mean(tf.square(change_in_q)/2)
    
    # optimizer can be changed by the setting
    if optimizer == "SGD":
        train_step = tf.train.GradientDescentOptimizer(learning_rate=Learning_rate).minimize(loss)
    elif optimizer == "ADAM":
        train_step = tf.train.AdamOptimizer(learning_rate=Learning_rate).minimize(loss)
    elif optimizer == "RMS":
        train_step = tf.train.RMSPropOptimizer(learning_rate=Learning_rate,decay=0.9,momentum=0.2,centered=True).minimize(loss)
    else:
        aaaaaaaaa

    l = 0
    
    # it is due to the input arguments for this to be a script just to collect random data or to train
    collect_random = int(sys.argv[2])
    
    # load the random data pre-stored, it contains data with postive, negative, and zero reward
    real_replay_buffer_N, real_replay_buffer_P, real_replay_buffer_Z = load_random_data()
        
        
    # mix the real_relay_buffer which is used for actual training with different reward datas
    # so that within the real_relay_buffer, the data are balanced. 
    # That is, there are same amount of data for positive, negative and zero rewards.
    try:
        N_idx = np.random.randint(len(real_replay_buffer_N), size=buffer_size)
        P_idx = np.random.randint(len(real_replay_buffer_P), size=buffer_size)
        Z_idx = np.random.randint(len(real_replay_buffer_Z), size=buffer_size)

        real_replay_buffer = np.array(real_replay_buffer_N)[N_idx,:]
        real_replay_buffer = np.append(real_replay_buffer, np.array(real_replay_buffer_P)[P_idx,:], axis=0)
        real_replay_buffer = np.append(real_replay_buffer, np.array(real_replay_buffer_Z)[Z_idx,:], axis=0)
        
        # the relay buffer which is the copy version of real_replay_buffer, this one is used for random batching
        replay_buffer = real_replay_buffer
    except:
        replay_buffer = []
        
    
    # a function to get random batch for training
    def random_batch():
        if game_name != "MsPacgirl":
            idx = np.random.randint(buffer_size * 3, size=batch_size)
        else:
            idx = np.random.randint(buffer_size * 2, size=batch_size)
        batch_x = replay_buffer[idx,:]
        return batch_x
    
    # if collect_random is true, this function to keep random random policy using multiprocessing 
    # untill all three relay buffers storing data with different rewards are filled, or stop by human interrupt
    # this will return after collection, no training will be proceed
    if collect_random:
        while 1:
            if len(real_replay_buffer_N) == len(real_replay_buffer_Z) == len(real_replay_buffer_P) == buffer_size:
                break
            if len(real_replay_buffer_Z) == len(real_replay_buffer_P) == buffer_size and game_name== "MsPacgirl":
                break    
            n=z=p=1
            if len(real_replay_buffer_N) >=buffer_size:
                n = 0
            if len(real_replay_buffer_Z) >=buffer_size:
                z = 0
            if len(real_replay_buffer_P) >=buffer_size:
                p = 0
                
            # home-made multiprocessing
            mp = MP(max_process=18)
            
            if len(real_replay_buffer_N) >= buffer_size and len(real_replay_buffer_Z) >= buffer_size:
                collection = 5000
            elif len(real_replay_buffer_Z) >= buffer_size:
                collection = 500
            else:
                collection = 50
            for num in range(collection):
                mp.give([num, len(real_replay_buffer_N), len(real_replay_buffer_Z), len(real_replay_buffer_P)], key=num)
            
            # run the multiprocess with collect_random_func
            mp.run(collect_random_func, mode=1, split=1)
            
            # collect the data after training
            return_dict = mp.get()
            for num in return_dict:
                N, Z, P = return_dict[num]
                for data in N:
                    real_replay_buffer_N.append(data)
                for data in Z:
                    real_replay_buffer_Z.append(data)
                for data in P:
                    real_replay_buffer_P.append(data)
                print("\r{} : {} : {}".format(len(real_replay_buffer_N),  len(real_replay_buffer_Z), len(real_replay_buffer_P) ), end="")

            print("\ncollection finished")
            print(np.array(real_replay_buffer_N).shape, np.array(real_replay_buffer_Z).shape, np.array(real_replay_buffer_P).shape)
            # save the random data
            save_random_data(real_replay_buffer_N, real_replay_buffer_P, real_replay_buffer_Z, n=n, z=z, p=p)
            
            
        return 0
    
    # here comes the training code
    with tf.Session() as sess:
        print()
        # name of the last checkpoint model
        model_name = "{}/{}-{}-*{}*{}".format(path,file_name,game_name,-999999,'.checkpoint.meta')
        print(model_name)
        
        # recover model and some data if possible, again, not used, all model start from begining
        if os.path.exists(model_name):
            model_name = model_name[:-5]
            print(model_name)
            print()
            saver = tf.train.Saver()
            saver.restore(sess, model_name)
            
            loss_array = list(np.load(path + "/" + game_name + "-loss.npy"))
            average_performance = list(np.load(path + "/" + game_name + "-performance.npy"))
            discounted_list = list(np.load(path + "/" + game_name + "-discounted_list.npy"))
            step_loss = list(np.load(path + "/" + game_name + "-step_loss.npy"))
            test_perform = list(np.load(path + "/" + game_name + "-test_perform.npy"))
            test_rtn = list(np.load(path + "/" + game_name + "-test_rtn.npy"))
            total_runs = len(step_loss)
            print("recover successful")
            l = step_loss[-1]
            
        # start the new model and define some list for data storing usage and initial global variables
        else:
            print("recover fail, starting from beginning")
            sess.run(tf.global_variables_initializer())
            loss_array = []
            average_performance = []
            discounted_list = []
            step_loss = []
            test_perform = []
            test_rtn = []
            total_runs = 0
            l = 0
         
        # define the class for test performance
        TP = test_performance()
        # define the class to tell me how long does the training take
        est = time_est(max_runs)
        buffer_ready = 0
        # a time reference for saving training data, do not 
        save_model_ref = time.time()
        for episode in range(max_runs):
            t_ref = time.time()
            # break after maximum run and save all data
            if total_runs > max_runs:
                np.save(path + "/" + game_name + "-loss.npy", loss_array)
                np.save(path + "/" + game_name + "-performance.npy", average_performance)
                np.save(path + "/" + game_name + "-discounted_list.npy", discounted_list)
                np.save(path + "/" + game_name + "-step_loss.npy", step_loss)
                np.save(path + "/" + game_name + "-test_perform.npy", test_perform)
                np.save(path + "/" + game_name + "-test_rtn.npy", test_rtn)
                save_model(sess, -999999)                
                break

            # reset enviorment and set some variables for training
            old_obs = env.reset()
            count = 0
            t = 0
            frame_buffer = deque(maxlen=4)
            observation_stack = []
            loss_set = []
            
            total_reward = 0
            discounted_value = 0
            discount_factor = 1
            buffer_ready = 1
            while 1:
                # the greedy rate can be controlled and vary
                greedy_rate = max(final_greedy_rate, 1 - (total_runs - eval_start)/(greedy_stop_num - eval_start))
                if total_runs == greedy_stop_number:
                    print("\ngreedy stop\n")
                
                if count!= 0:
                    discount_factor *= 0.99
                # env.render()
                
                total_runs+=1

                # update sess every 5000 steps
                if buffer_ready and total_runs > train_start and total_runs % 5000 == 0 and collect_random == 0:
                    update(sess)
                
                # select action
                if t > 3 and (random.random() > greedy_rate and total_runs < greedy_stop_number) and total_runs > eval_start and collect_random == 0:
                    action_dic = {obs:[old_obs],obs_new:np.zeros([1,image_size,image_size,4]),act:[0],reward_place:[reward]}
                    action = action_max.eval(action_dic)[0]
                else:
                    action = round(random.uniform(0, num_of_actions - 1))

                # get the observation, reward, and state of done, and information from the action
                observation, reward, done, info = env.step(action)
  
                # get image into frame stack
                obs_frame = image_preprocess(observation)
                frame_buffer.append(obs_frame)

                if len(frame_buffer) == 4:
                    observation_stack = np.stack(list(frame_buffer),axis=0)
                    observation_stack = observation_stack.transpose([1,2,0])
                
                # clip rewards
                reward = np.clip(reward,-1,1)
                total_reward += reward
                count += 1

                # the training process
                if buffer_ready and total_runs > train_start and len(replay_buffer) > 10 and collect_random == 0:
                    batch_x = random_batch()
                    a = list(batch_x[:,0])
                    b = list(batch_x[:,1])
                    c = list(batch_x[:,2])
                    d = list(batch_x[:,3])
                    # optimize by batch
                    sess.run(train_step, feed_dict={obs: a, obs_new:d,act:b, reward_place: c})
                    # calculate bellman loss
                    l = sess.run(loss, feed_dict={obs: a, obs_new:d,act:b, reward_place: c})
                    loss_set.append(l)
                    step_loss.append(l)
                    
                # store rawards seperately with respect to different rewards
                if t > 3:
                    if reward == -1:
                        real_replay_buffer_N.append([old_obs,action,reward,observation_stack])
                    elif reward == 1:
                        real_replay_buffer_P.append([old_obs,action,reward,observation_stack])
                    else:
                        real_replay_buffer_Z.append([old_obs,action,reward,observation_stack])
                
                # save the old observation
                old_obs = observation_stack
                t += 1

                # print some thing
                if total_runs % 20 == 0:
                    mean_loss = np.mean(loss_set)
                    print("\r{}, ep = {:>4}, r = {:>8}, l = {:>20}, rew= {:>4}, gr = {:.4f}, buffer = {} : {} : {}".format(seed_num, episode, total_runs, mean_loss, int(total_reward), greedy_rate, len(real_replay_buffer_N),  len(real_replay_buffer_Z), len(real_replay_buffer_P) ), end="")
                # calculate the discounted returns
                discounted_value += discount_factor * reward

                # get test performance every 50k
                if buffer_ready and total_runs % 50000 == 0 and total_runs > train_start and collect_random == 0:
                    score_mean, ds_rtn_mean = TP.test(sess, action_max, assume_score=1000)
                    test_perform.append(score_mean)
                    test_rtn.append(ds_rtn_mean)

                if done:
                    if buffer_ready and collect_random == 0 and total_runs > train_start:
                        # process the data after every "done" and store then into the buffer for training
                        # to avoid too much numpy array actions, which really slows the program
                        Z_idx = np.random.randint(len(real_replay_buffer_Z), size=buffer_size)
                        P_idx = np.random.randint(len(real_replay_buffer_P), size=buffer_size)
                        if game_name != "MsPacgirl":
                            N_idx = np.random.randint(len(real_replay_buffer_N), size=buffer_size)

                        real_replay_buffer = np.array(real_replay_buffer_Z)[Z_idx,:]
                        real_replay_buffer = np.append(real_replay_buffer, np.array(real_replay_buffer_P)[P_idx,:], axis=0)
                        if game_name != "MsPacgirl":
                            real_replay_buffer = np.append(real_replay_buffer, np.array(real_replay_buffer_N)[N_idx,:], axis=0)
                        
                        replay_buffer = real_replay_buffer
                        
                    
                    if total_runs < train_start or buffer_ready == 0 or collect_random == 1:
                        break

                    # store some data
                    discounted_list.append(discounted_value)
                    average_performance.append(total_reward)
                    mean_loss = np.mean(loss_set)
                    loss_array.append(mean_loss)
                    
                    # save data every 20 mins
                    if time.time() - save_model_ref > 20 * 60:
                        np.save(path + "/" + game_name + "-loss.npy", loss_array)
                        np.save(path + "/" + game_name + "-performance.npy", average_performance)
                        np.save(path + "/" + game_name + "-discounted_list.npy", discounted_list)
                        np.save(path + "/" + game_name + "-step_loss.npy", step_loss)
                        np.save(path + "/" + game_name + "-test_perform.npy", test_perform)
                        np.save(path + "/" + game_name + "-test_rtn.npy", test_rtn)
                        save_model(sess, -999999)
                        save_model_ref = time.time()
                    # put the new steps for averaging
                    if total_reward > highest_reward:
                        highest_reward = total_reward
                    
                    # print something
                    print("\r{}, episode = {:>3}, runs = {:>8}, loss = {:>20}, reward = {:>4}, high = {:>4}, time = {:.2f}, disc = {:.8f}, mean_disc = {:.8f}, buffer = {} : {} : {}".format(seed_num, episode, total_runs, mean_loss, int(total_reward), highest_reward, time.time()-t_ref, discounted_value, np.mean(discounted_list[-100:]), len(real_replay_buffer_N),  len(real_replay_buffer_Z), len(real_replay_buffer_P) ))

                    
                    print()
                    est.check(t)
                    print()
                    print()
                    break



if __name__ == "__main__":
    # start training
    training(round(random.uniform(0,3000)))

