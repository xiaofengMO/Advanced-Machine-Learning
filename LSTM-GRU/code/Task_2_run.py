import os

for unit_name in ["GRU"]:
    for mode in ["32", "64", "128", "stack_32"]:
        print("running {}, {}".format(unit_name, mode))
        os.system("CUDA_VISIBLE_DEVICES=1 python Task_2_script.py {} {}".format(unit_name, mode))