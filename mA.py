import numpy as np

def multipleA(A,cur_state, final_state):
    multiple_A = np.eye(A.shape[0])       
        
    for i in range(cur_state ,final_state+1):
        multiple_A = np.matmul(A[:,:,i], multiple_A)

    return multiple_A