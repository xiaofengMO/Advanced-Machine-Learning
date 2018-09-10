import os

for unit_name in ["GRU"]:
    for mode in ["32", "64", "128", "stack_32"]:
        for pred_length in [1, 10, 28, 300]:
            print("running {}, {}, {}".format(unit_name, mode, pred_length))
            os.system("CUDA_VISIBLE_DEVICES=1 python Task_2_in_painting_script.py {} {} {}".format(unit_name, mode, pred_length))