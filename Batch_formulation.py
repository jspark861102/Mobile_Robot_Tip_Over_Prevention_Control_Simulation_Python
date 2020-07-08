import numpy as np
from mA import multipleA
from scipy.linalg import block_diag
from for_gen import generator

def BF(A, B, N, Q, R, cur_state):
    #parameter size
    n = A.shape[0]
    m = B.shape[1]

    #Sx{n*N,n}
    Sx = np.zeros((A.shape[0],A.shape[0],N))
    Sx = A[:,:,cur_state]
    for i in generator(1, N):
        Sx = np.append(Sx, multipleA(A,cur_state,cur_state+i), axis =0)

    #Su{n*N,m*N}
    Su = np.zeros((n*N,m*N))
    for i in generator(1, N+1):
        for j in generator(1, N+1):    
            if i-j < 0: #upper than diagonal 
                At = np.zeros((n,n))
            elif i-j == 0: #diagonal
                At = np.eye(n)
            else: #i-j>0
                At = multipleA(A,cur_state+1,cur_state+(i-j))            
            
            Su[(i-1)*n : i*n , (j-1)*m : j*m] = np.matmul(At, B[:,:,cur_state + (j-1)] )
    
    #Qb{n*N,n*N}
    Q_dummy = Q.copy()
    for i in generator(2,N+1):    
        Qb = block_diag(Q_dummy,Q)    
        Q_dummy = Qb
    
    #Rb{m*N,m*N}
    R_dummy = R.copy()
    for i in generator(2,N+1):    
        Rb = block_diag(R_dummy,R)
        R_dummy = Rb

    return Sx, Su, Qb, Rb