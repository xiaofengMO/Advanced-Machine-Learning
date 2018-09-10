All files ended with ".py" are for training, the jupyter notebooks are for recovery.

The XFLib_1.py is my home-made script for multiprocessing, I use it to collect random data

########################   model recover   ###################################
The notebook "B_recover_model.ipynb" is used to recover the three best models for Problem B sequentially, there is a render_flag at the very beginnig of the notebook which controls whether to render, you can read the comment for detail usage.

The notebook "A_recover_models.ipynb" is used to recover models from Problem A, there is a render_flag at the very beginnig of the notebook which controls whether to render, you can read the comment for detail usage.

This notebooks are ready to run, just run them to recover the model and calculate the test performance.

##############################################################################
Notebook "B_random_and_untrained_performance.ipynb" was used to get the performance of random and untrained models in Problem B

Notebook "B_data_recover.ipynb" was used to recover the data and some image for reports

Notebook "A_data_recover.ipynb" was used to recover the data and some image for reports
##############################################################################


Training files:

A_1.py for Problem A 1
A_2.py for Problem A 2
A_3.py for Problem A 3
A_4.py for Problem A 4
A_5.py for Problem A 5
A_6.py for Problem A 6
A_7.py for Problem A 7
A_8.py for Problem A 8

B_4_3.py for Problem B 3 and 4

###############################################################################
Runing files

file "run_A.py" provide a way to run training files in Problem A, but you can always just run them.


If you want to run B_4_3.py, you might need to install some package which was used in XFLib_1.py (home-made multiprocessing library, and sadly it does not work properly in Windows, I believe linux and mac os are fine), and the way to run it is in "run_B.py". 
