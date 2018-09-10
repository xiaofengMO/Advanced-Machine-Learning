import os, sys, time
import random

name_list = ["MsPacgirl", "Boxing", "Pong", ]

for name in name_list:
    print("running", name)
    cmd_str = "CUDA_VISIBLE_DEVICES=0 python B_4_7.py {} 0".format(name)
    print(cmd_str)
    os.system(cmd_str)
    print("\n\n\n\n")