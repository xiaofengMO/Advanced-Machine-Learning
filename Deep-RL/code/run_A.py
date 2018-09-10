import os, sys, time
import random


for i in range(8):
    name = "A_{}.py".format(i)
    print("running", name)
    cmd_str = "CUDA_VISIBLE_DEVICES=0 python {}".format(name)
    print(cmd_str)
    os.system(cmd_str)
    print("\n\n\n\n")