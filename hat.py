import numpy as np
import math
import mA

def A_hat(A,cur_state,final_state):
    result = mA.multipleA(A,cur_state, final_state-1)
    return result


def B_hat(A,B,cur_state,final_state):
    n = A.shape[0]
    m = B.shape[1]

    del_fc = final_state - cur_state
    result = np.zeros((n,m,del_fc))
    for i in range(del_fc, 0, -1):
        if i == del_fc:
            result[:,:,i-1] = B[:,:,final_state-1]
        else:
            result[:,:,i-1] =np.matmul( mA.multipleA(A,cur_state+i,final_state-1), B[:,:,cur_state + i -1] )
    return result

