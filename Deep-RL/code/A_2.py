import gym
import random
import numpy as np
import os

# make the gym enviorment
env = gym.make("CartPole-v1")
env.reset()

# get the file name for storing
file_name = str(os.path.basename(__file__))[:-3]

# a function to store data
def store_data(store_dict):
    # make the "models" folder if not exists
    if not os.path.exists('../models/{}'.format(file_name)):
        os.mkdir('../models/{}'.format(file_name))
    # save the data to "models" folder
    np.save("../models/{}/{}_store_dict.npy".format(file_name, file_name), store_dict)
    print("saved to ../models/{}/{}_store_dict.npy".format(file_name, file_name)) 

# define a storing dictionary to store the data needed
store_dict = {}
store_dict["length_list"] = []
store_dict["dis_return_list"] = []


# define some hyperparameters
dis_factor = 0.99

# start the runing
for episode in range(100):
    
    # reset enviorment and some value before get into this episode
    env.reset()
    dis_return = 0
    dis_value = 1
    
    # start the episode with maximun length of 300
    for length in range(1, 301):

        # generate a random motion base on uniform distribution
        action = round(random.uniform(0,1))

        # get the observation, reward, and state of done, and information from the action
        observation, reward, done, info = env.step(action)
        
        # set the reward to be 0 on non-terminating steps and -1 on termination
        reward = -1 if done else 0
        
        # count the number of action token
        dis_return += dis_value * reward
        dis_value *= dis_factor
        
        # store data, print and break after "done"
        if done or length == 300:
            store_dict["length_list"].append(length)
            store_dict["dis_return_list"].append(dis_return)
            print(length,dis_return)
            break


# store data
store_data(store_dict)