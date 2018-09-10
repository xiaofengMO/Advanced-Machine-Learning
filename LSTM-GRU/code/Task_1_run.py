import os
for unit_name in ["GRU", "LSTM"]:
    for mode in ["32", "64", "128", "stack_32"]:
        print("running {}, {}".format(unit_name, mode))
        os.system("CUDA_VISIBLE_DEVICES=0 python Task_1_script.py {} {}".format(unit_name, mode))