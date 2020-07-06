import numpy as np
import math
from scipy.linalg import block_diag
from mA import multipleA
import hat

# QP constraints
def constraints(cur_state, xr, ddxr, x_tilt_current, x_tilt_m1, N, A, B, h2, D, L, T, switch_zmp, switch_input, thr_input, switch_state, thr_state,mu):
    n = A.shape[0]
    m = B.shape[1]

    #zmp matrix
    h = h2
    g = 9.81
    P = 1/(T*T)*np.array([ [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0] ])

   ############################### backward finite difference for zmp ###############################
    if switch_zmp == 1:
        #G0{2*n*N, m*N}        
        Bhat = np.zeros((N, n, m, N))  
        for b_hat_num in range(1, N+1):    
            #in writing Bi(k+j), in code Bj(k+i), 
            #(1,2,3,4), 1:j, (2,3):B(:,:), 4:b_hat_num
            Bhat[b_hat_num-1, :, :, :b_hat_num] = hat.B_hat(A,B,cur_state, cur_state+b_hat_num)            

        G01 = np.zeros((n*N, m*N))
        for i in range(1, N+1):    
            for j in range(1, N+1):    
                if i-j == 0: #diagonal                    
                    G01[(i-1)*n : i*n, (j-1)*m : j*m] = np.matmul( P, Bhat[i-1,:,:,j-1] )                    
                     
                elif i-j == 1: #one lower than diagonal                     
                    G01[(i-1)*n : i*n, (j-1)*m : j*m] = np.matmul( P, Bhat[i-1,:,:,j-1] ) + np.matmul( P, -2*Bhat[i-1-1,:,:,j-1] )

                elif i-j < 0: #upper diagonal is 0                    
                    G01[(i-1)*n : i*n, (j-1)*m : j*m] = 0

                else: #lower diagonal                    
                    G01[(i-1)*n : i*n, (j-1)*m : j*m] = np.matmul(P, Bhat[i-1,:,:,j-1]) + np.matmul(P, -2*Bhat[i-1-1,:,:,j-1] + Bhat[i-1-2,:,:,j-1])                      
                    
        #G0 = np.append(G01,-G01, axis=0)
        G0 = G01.copy()

        #E0{2*n*N,n}
        E01 = np.zeros((n*N,n))
        for i in range(1, N+1):    
            if i == 1:                
                E01[(i-1)*n : i*n,:] = np.matmul( P, hat.A_hat(A, cur_state, cur_state+i) ) + np.matmul( P, -2*np.eye(n) )
                
            elif i == 2:
                E01[(i-1)*n : i*n,:] = np.matmul( P, hat.A_hat(A, cur_state, cur_state+i) ) + np.matmul( P, -2*hat.A_hat(A,cur_state,cur_state+(i-1)) + np.eye(n) )

            else:
                E01[(i-1)*n : i*n,:] = np.matmul( P, hat.A_hat(A, cur_state, cur_state+i) ) + np.matmul( P, -2*hat.A_hat(A,cur_state,cur_state+(i-1)) + hat.A_hat(A,cur_state,cur_state+(i-2)) )

        #E0 = np.append(-E01, -(-E01), axis=0)
        E0 = -E01.copy()


        #W0{2*n*N,1}
        #w0 = np.zeros(2*n*N)        
        w0u = np.zeros(n*N)
        w0l = np.zeros(n*N)
        for i in range(1, N+1):    
            w0u[(i-1)*n : i*n] = np.amin([ np.matmul(np.array([ [np.abs(np.cos(x_tilt_current[2]+xr[2,cur_state])), np.abs(np.sin(x_tilt_current[2]+xr[2,cur_state])), 0], 
                                                                [np.abs(np.sin(x_tilt_current[2]+xr[2,cur_state])), np.abs(np.cos(x_tilt_current[2]+xr[2,cur_state])), 0],
                                                                [                                                0,                                                 0, 1] ]),  np.array([D/2,L/2,0])*g/h), mu*g], axis=0) - np.array([ddxr[0,cur_state+i], ddxr[1,cur_state+i], 0])
                                     
        for i in range(1, N+1):    
            w0l[(i-1)*n : i*n] = - np.amin([ np.matmul(np.array([ [np.abs(np.cos(x_tilt_current[2]+xr[2,cur_state])), np.abs(np.sin(x_tilt_current[2]+xr[2,cur_state])), 0], 
                                                                  [np.abs(np.sin(x_tilt_current[2]+xr[2,cur_state])), np.abs(np.cos(x_tilt_current[2]+xr[2,cur_state])), 0],
                                                                  [                                                0,                                                 0, 1] ]),  np.array([D/2,L/2,0])*g/h), mu*g], axis=0) - np.array([ddxr[0,cur_state+i], ddxr[1,cur_state+i], 0])

        w0u = w0u - np.append(np.matmul(P, x_tilt_m1), np.zeros(n*(N-1)), axis=0)
        w0l = w0l - np.append(np.matmul(P, x_tilt_m1), np.zeros(n*(N-1)), axis=0)    
        #w0 = np.append(w01, w02, axis=0)        
        ##############################################################          
        
    else:
        G0 = np.array([])
        E0 = np.array([])
        w0u = np.array([])
        w0l = np.array([])
    
    #############################################################################################

    ############################### state limit ###############################
    if switch_state == 1:
        #state limit
        E22 = np.zeros((n*N,n))
        for i in range(1, N+1):    
            E22[(i-1)*n : i*n,:] = multipleA(A, cur_state, cur_state + (i-1))    

        #E2 = np.append(-E22, -(-E22), axis=0)
        E2 = -E22.copy()

        G22 = np.zeros((n*N, m*N))       
        Bhat = np.zeros((N, n, m, N))  
        for i in range(1, N+1):    
            #in writing Bi(k+j), in code Bj(k+i), 
            #(1,2,3,4), 1:j, (2,3):B(:,:), 4:b_hat_num
            Bhat[i-1, :, :, :i] = hat.B_hat(A,B,cur_state, cur_state+i)                        
            for j in range(1, i+1):     
                G22[(i-1)*n : i*n, (j-1)*m : j*m] = Bhat[i-1,:,:,j-1]
        
        #G2 = np.append(G22, -G22, axis=0)
        G2 = G22.copy()
        
        w2u = np.zeros(n*N)
        w2l = np.zeros(n*N)
        for i in range(1, N+1):
            w2u[(i-1)*n : i*n] =  thr_state[:3]
        
        for i in range(1, N+1):
            w2l[(i-1)*n : i*n] =  -thr_state[3:]
        
        #w2 = np.append(w21, w22, axis=0)
    else:
        G2 = np.array([])
        E2 = np.array([])
        w2u = np.array([])
        w2l = np.array([])
    
    #############################################################################################

    ############################### input limit ###############################
    if switch_input == 1:
        #input limit                
        w1u = thr_input[:3]
        w1l = -thr_input[3:]
        for i in range(2, N+1):    
            w1u = np.append(w1u, thr_input[:3], axis=0)
            w1l = np.append(w1l, -thr_input[3:], axis=0)

    else:        
        w1u = np.array([])
        w1l = np.array([])
    #############################################################################################

    #constraint completion
    G = np.append(G0, G2).reshape(-1,m*N)
    E = np.append(E0, E2).reshape(-1,n)
    wu = np.append(w0u, w2u).reshape(-1)
    wl = np.append(w0l, w2l).reshape(-1)
    wu_input = w1u
    wl_input = w1l    

    return G, E, wu, wl, wu_input, wl_input

