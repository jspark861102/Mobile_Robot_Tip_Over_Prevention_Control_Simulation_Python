import numpy as np
from for_gen import generator

def multipleA(A,cur_state, final_state):
    multiple_A = np.eye(A.shape[0])       
        
    for i in generator(cur_state ,final_state+1):
        multiple_A = np.matmul(A[:,:,i], multiple_A)

    return multiple_A