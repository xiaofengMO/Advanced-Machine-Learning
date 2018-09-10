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
binarized_act_place = tf.placeholder(tf.float32,[None,2])


# define the liear or nonlinear model base on the selection
def define_model(seed_num, num_hidden_units):
    if mode == "linear":
        tf.set_random_seed(seed_num)
        W1 = tf.Variable(tf.random_normal([4, 2]))
        b1 = tf.Variable(tf.random_normal([2]))

        y1 = tf.matmul(obs, W1) + b1
        y1_new = tf.matmul(obs_new, W1) + b1

        return y1,y1_new
    elif mode == "non_linear":
        tf.set_random_seed(seed_num)
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
                    action_dic = {obs: [old_obs], obs_new: np.zeros([1, 4]), act: [0], binarized_act_place: [[0, 0]],
                                  reward_place: [reward]}
                    action = action_max.eval(action_dic)[0]
                else:
                    action = round(np.random.uniform())

                # get the observation, reward, and state of done, and information from the action
                observation, reward, done, info = env_1.step(action)
                
                # set the reward to be 0 on non-terminating steps and -1 on termination
                reward = -1 if done else 0
                
                # count the number of action token
                dis_return += dis_value * reward
                dis_value *= dis_factor

                old_obs = observation
                if done or length == 300:
                    perf_list.append(length)
                    dis_rtn_list.append(dis_return)
                    break
    return np.mean(perf_list), np.mean(dis_rtn_list)

# a function for training
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

    # some numpy place holders for data
    old_obs = np.stack(sample_data[:,0],axis=0)
    action = np.stack(sample_data[:,1],axis=0)
    observation = np.stack(sample_data[:,3],axis=0)
    reward = np.stack(sample_data[:,2],axis=0)
    
    # define the model
    y2,y2_new = define_model(tf_random_seed, num_hidden_units)

    # get the action by Q values
    action_max = tf.cast(tf.argmax(y2,axis=1),tf.int32)
    action_max_input = tf.cast(action_max,tf.float32)
    
    # the following are for losses
    num_data = tf.shape(y2_new)[0]
    q_old_index = tf.concat([tf.reshape(tf.range(0,limit=num_data),[num_data,1]), tf.reshape(act,[num_data,1])],axis=1)
    old_q = tf.gather_nd(y2,q_old_index)
    next_q_max = tf.reduce_max(y2_new, axis=1)

    change = (reward_place + dis_factor * tf.stop_gradient(next_q_max) - old_q)
    bellman_loss = tf.reduce_mean(tf.square(change)/2)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y2,labels=binarized_act_place)
    
    # select an optimizer
    if optimizer == "SGD":
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    elif optimizer == "RMS":
        train_step =tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.9,momentum=0.2,centered=True).minimize(loss)
    elif optimizer == "ADAM":
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # numpy random seed
    np.random.seed(np_random_seed)
    b_loss = 0
    
    # start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n = np.shape(old_obs)[0]
        binarized_act = np.zeros((n,2))
        binarized_act[np.arange(n),action] = 1

        for epoch in range(1000):
            random_list = np.random.choice(n, batch_size, replace=False)
            for length in range(100):
                obs_input = old_obs[random_list]
                observation_input = observation[random_list]
                action_input = action[random_list]
                reward_input = reward[random_list]
                binarized_act_input = binarized_act[random_list]
                sess.run(train_step, feed_dict={obs: obs_input, obs_new:observation_input, act:action_input,binarized_act_place:binarized_act_input,reward_place: reward_input})

                l = sess.run(loss, feed_dict={obs: obs_input, obs_new:observation_input,act:action_input,binarized_act_place:binarized_act_input,reward_place: reward_input})
                print("\repoch {:>4}, length = {:>4}, loss = {:.4f}, best test performance = {:>6}".format(epoch, length, b_loss, best_test_performance), end="")
                
            b_loss = sess.run(bellman_loss, feed_dict={obs: old_obs, obs_new:observation,act:action, binarized_act_place:binarized_act,reward_place: reward})
            test_performance, test_dis_return = test_model(sess, action_max)
            
            # save the data in the list
            store_dict["test_preformance_list"].append(test_performance)
            store_dict["test_dis_return"].append(test_dis_return)
            store_dict["bellman_loss_list"].append(b_loss)
            
            # save model if the performance is better than the best performance
            if test_performance > best_test_performance:
                save_model(sess)
                best_test_performance = test_performance
                
    # store the data
    store_data(store_dict)
    print(time.time() - t_ref)
    
if __name__ == "__main__":
    # run the model mutiple times with different hyperparameters
    number_of_random = 1
    for randomness in range(number_of_random):
        print(randomness)
        tf_random_seed = int(random.random() * 10000)
        np_random_seed = int(random.random() * 10000)    
        for mode in ["linear", "non_linear"]:
            for learning_rate in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]:
                for batch_size in [128]:
                    for optimizer in ["SGD", "RMS", "ADAM"]:
                        for num_hidden_units in [100]:
                            training(tf_random_seed, np_random_seed, mode, learning_rate, batch_size, optimizer, num_hidden_units)