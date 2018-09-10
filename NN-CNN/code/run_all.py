import os, sys, time

file_names = ["Part_1_a_train.py", "Part_1_b_train.py", "Part_1_c_train.py", "Part_1_d_train.py", "Part_2_b_train.py", "Part_2_c_train.py", "Part_2_d_train.py"]


for i in file_names:
    print("running {}".format(i))
    os.system("CUDA_VISIBLE_DEVICES=0 python {} {}".format(i, 1000))
