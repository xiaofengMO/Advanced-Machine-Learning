import gym
import random
import numpy as np
import collections
import tensorflow as tf
import time
from collections import deque
import matplotlib.pyplot as plt
import os,sys

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# make the gym enviorment
env = gym.make("CartPole-v1")
env.reset()
# make the gym enviorment for testing purpose
env_1 = gym.make("CartPole-v1")
env.reset()


# define some fixed hyperparameters
dis_factor = 0.99
greedy_rate = 0.05
# get the file name for storing
file_name = str(os.path.basename(__file__))[:-3]
# file_name = "test"

# define some place holder
reward_place = tf.placeholder(tf.float32, [None])
obs_new = tf.placeholder(tf.float32, [None,4])
obs = tf.placeholder(tf.float32, [None,4])
act = tf.placeholder(tf.int32,[None])


# define the model
def define_model(random_seed, num_hidden_units):
    tf.set_random_seed(random_seed)
    W1 = tf.Variable(tf.random_normal([4, num_hidden_units]))
    b1 = tf.Variable(tf.random_normal([num_hidden_units]))

    y1 = tf.nn.relu(tf.matmul(obs, W1) + b1)
    y1_new = tf.nn.relu(tf.matmul(obs_new, W1) + b1)
    
    W2 = tf.Variable(tf.random_normal([num_hidden_units, 2]))
    b2 = tf.Variable(tf.random_normal([2]))

    y2 = tf.matmul(y1, W2) + b2
    y2_new = tf.matmul(y1_new, W2) + b2

    return y2,y2_new

# sample some data
def sampling_cart_pole(np_random_seed):
    np.random.seed(np_random_seed)
    sample_list = []
    for episode in range(2000):
        old_obs = env.reset()
        dis_return = 0
        dis_value = 1
        
        for length in range(1, 301):
            action = round(np.random.uniform())
            
            # get the observation, reward, and state of done, and information from the action
            observation, reward, done, info = env.step(action)
            if length > 1:
                sample_list.append([old_obs,action,reward,observation])
                
            # set the reward to be 0 on non-terminating steps and -1 on termination
            reward = -1 if done else 0

            # count the number of action token
            dis_return += dis_value * reward
            dis_value *= dis_factor 
            
            old_obs = observation
            if done or length == 300:
                break

    return np.array(sample_list)



# code for testing performace
def test_model(sess, action_max):
    perf_list = []
    dis_rtn_list = []
    with sess.as_default():
        for episode in range(20):
            old_obs = env_1.reset()
            dis_return = 0
            dis_value = 1            
            for length in range(1, 301):
                if length > 1 and np.random.uniform() > greedy_rate:
                    action_dic = {obs:[old_obs],obs_new:np.zeros([1,4]),act:[0],reward_place:[reward]}
                    action = action_max.eval(action_dic)[0]
                else:
                    action = round(np.random.uniform())

                # get the observation, reward, and state of done, and information from the action
                observation, reward, done, info = env_1.step(action)
                
                # set the reward to be 0 on non-terminating steps and -1 on termination
                reward = -1 if done else 0
                
                # calculate the dicounted return
                dis_return += dis_value * reward
                dis_value *= dis_factor

                old_obs = observation
                if done or length == 300:
                    perf_list.append(length)
                    dis_rtn_list.append(dis_return)
                    break
    return np.mean(perf_list), np.mean(dis_rtn_list)


def training(tf_random_seed, np_random_seed, mode, learning_rate, batch_size, optimizer, num_hidden_units):
    t_ref = time.time()
    print(tf_random_seed, np_random_seed, mode, learning_rate, batch_size, optimizer, num_hidden_units)
    sample_data = sampling_cart_pole(np_random_seed)

    # define a storing dictionary to store the data needed
    store_dict = {}
    store_dict["length_list"] = []
    store_dict["dis_return_list"] = []
    store_dict["test_dis_return"] = []
    store_dict["test_preformance_list"] = []
    store_dict["bellman_loss_list"] = []
    
    # a function to store data
    def store_data(store_dict):
        # make the "models" folder if not exists
        if not os.path.exists('../models/{}'.format(file_name)):
            os.mkdir('../models/{}'.format(file_name))
        # save the data to "models" folder
        model_save_path = '../models/{}/{}_{}_{}_{}_{}_{}_{}_store_dict.npy'.format(file_name, file_name, tf_random_seed, np_random_seed, mode, learning_rate, batch_size, optimizer)
        np.save(model_save_path, store_dict)
        print('model saved to {}'.format(model_save_path))

    # a function to save model so that we can recover from the futher
    def save_model(sess):
        # make the "models" folder if not exists
        if not os.path.exists('../models/{}'.format(file_name)):
            os.mkdir('../models/{}'.format(file_name))
        # save the data to "models" folder
        saver = tf.train.Saver()
        model_save_path = '../models/{}/{}_{}_{}_{}_{}_{}_{}.checkpoint'.format(file_name, file_name, tf_random_seed, np_random_seed, mode, learning_rate, batch_size, optimizer)
        saver.save(sess, model_save_path)
        # print('model saved to {}'.format(model_save_path))
    
    best_test_performance = 0

    # define model
    y2,y2_new = define_model(tf_random_seed, num_hidden_units)
    
    # get action by Q values
    action_max = tf.cast(tf.argmax(y2,axis=1),tf.int32)
    
    # calculate bellman loss
    data_amount = tf.shape(y2_new)[0]
    Q_old_index = tf.concat([tf.reshape(tf.range(0,limit=data_amount),[data_amount,1]), tf.reshape(act,[data_amount,1])],axis=1)
    old_q = tf.gather_nd(y2,Q_old_index)
    next_q_max = tf.reduce_max(y2_new, axis=1)

    change = (reward_place + dis_factor * tf.stop_gradient(next_q_max) - old_q)
    bellman_loss = tf.reduce_mean(tf.square(change)/2)

    # choose optimizer base on the selection
    if optimizer == "SGD":
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(bellman_loss)
    elif optimizer == "RMS":
        train_step =tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.9,momentum=0.2,centered=True).minimize(bellman_loss)
    elif optimizer == "ADAM":
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(bellman_loss)

    # numpy random seed define
    np.random.seed(np_random_seed)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        b_loss = 0
        for episode in range(2000):
            old_obs = env.reset()
            dis_return = 0
            dis_value = 1
            for length in range(1, 301):
                
                # select action base on the greedy method
                if length > 1 and np.random.uniform() > greedy_rate:
                    action_dic = {obs:[old_obs],obs_new:np.zeros([1,4]),act:[0],reward_place:[reward]}
                    action = action_max.eval(action_dic)[0]
                else:
                    action = round(np.random.uniform())

                # get the observation, reward, and state of done, and information from the action
                observation, reward, done, info = env.step(action)

                # set the reward to be 0 on non-terminating steps and -1 on termination
                reward = -1 if done else 0

                # calculate the dicounted return
                dis_return += dis_value * reward
                dis_value *= dis_factor
                # optimize the model
                sess.run(train_step, feed_dict={obs: [old_obs], obs_new:[observation],act:[action], reward_place: [reward]})
                # calculate bellman loss
                b_loss = sess.run(bellman_loss, feed_dict={obs: [old_obs], obs_new:[observation],act:[action], reward_place: [reward]})
                # save the old observation
                old_obs = observation
            
                if done or length == 300:
                    # test the model
                    test_performance, test_dis_return = test_model(sess, action_max)
                    if test_performance > best_test_performance:
                        best_test_performance = test_performance
                    # store the data in the list
                    store_dict["test_preformance_list"].append(test_performance)
                    store_dict["test_dis_return"].append(test_dis_return)
                    store_dict["bellman_loss_list"].append(b_loss)
                    print("\repisode {:>4}, length = {:>4}, loss = {:.4f}, dis return = {:.4f}, best test performance = {:>6}".format(episode, length, b_loss, dis_return, best_test_performance), end="")
                    store_dict["length_list"].append(length)
                    store_dict["dis_return_list"].append(dis_return)
                    break
                
    # store data
    store_data(store_dict)
    print(time.time() - t_ref)
    
if __name__ == "__main__":
    # run the model mutiple times with different hyperparameters
    number_of_random = 100

    for mode in ["non_linear"]:
        for learning_rate in [1e-5]:
            for batch_size in [128]:
                for optimizer in ["RMS"]:
                    for num_hidden_units in [100]:
                        for randomness in range(number_of_random):
                            print(randomness)
                            tf_random_seed = int(random.random() * 10000)
                            np_random_seed = int(random.random() * 10000)                            
                            training(tf_random_seed, np_random_seed, mode, learning_rate, batch_size, optimizer, num_hidden_units)