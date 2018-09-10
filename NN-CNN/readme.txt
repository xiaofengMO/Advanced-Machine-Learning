    Training and recovering code are located in "code" folder

    File end with "train.py", all training file require a input argument : number of epoch training. 
    For example, training can be initialized using "python Part_1_a_train.py 1000" to train model for part 1.a, with a 1000 epoch.
    For Part 2 e, CNN, the input argument was not required, the epoch was fixed to 200.
    
    To recover model, simply using the jupyter notebooks which end with "recover", the result are already saved in the notebook, if you want to run it again, just run all cells. Some cells are used to produce confusion matrix, and some to print training and test error. The confusion matrix code was from scikit-learn. Notice !!! You may need to kill the kernel after running one notebook to release gpu, for another notebook to run properly.
    
    Since the recover notebook has the similar structure, all comment will be written in training files, since the recover notebooks' code have similar model structures as the training codes.