import numpy as np
from ctr import dlqr
import math
import copy
import ref_trajectory_mecanum
import time
from for_gen import generator
from data_plot import dataplot


###################################################################################################################################
##############################################  Parameters  #######################################################################
###################################################################################################################################

#############__Simulation Design Parameters__################
#model, 1:nonlinear model, 2:linear model           #########
model_switch = 2                                    #########
#reference trajectory, 1:wv, 2:xy                   #########
ref_traj_switch = 2                                 #########
#zmp constraint, 1:on 0:off                         #########
switch_zmp = 1                                      #########
#friction constraint, 1:on 0:off                    ######### 
switch_slip = 0                                     #########
#input constraint, 1:on 0:off                       #########
switch_input = 1                                    #########
#state constraint, 1:on 0:off                       #########
switch_state = 0                                    #########
#############################################################

#input constraint
mag = 2.0
thr_input_plus = np.array([mag, mag, mag]) 
thr_input_minus = np.array([-mag, -mag, -mag ])
thr_input = np.concatenate((thr_input_plus, -thr_input_minus))

#state constraint
thr_state_plus =  np.array([ 0.008,  0.008,  0.008])
thr_state_minus = np.array([-0.008, -0.008, -0.008])
thr_state = np.concatenate((thr_state_plus, -thr_state_minus))

#slip constraint
mu = np.array([0.46, 0.46, 0])
#mu = np.array([0.1, 0.1, 0])

#############__Simulation Design Parameters__#############
# MPC parameters
N = 15 # Batch size
Qs = 5 # state weight
Rs = 1 # input weight

#initial state : [x y theta]
if ref_traj_switch == 1:
    x0 = np.array([0.1, 0.2, math.pi/8])
elif ref_traj_switch == 2:
    x0 = np.array([0.0, 0.0, 0])

#wheel base & width
D = 0.5
L = 0.3

#z value from global origin to link2 mass center
h2 = 0.6

g = 9.81
T = 0.01                                            
dt = copy.copy(T)                                   

###################################################################################################################################
#########################################  Reference trajectory  ##################################################################
###################################################################################################################################

#reference trajectory xr=[x_ref, y_ref, theta_ref], ur=[dx_ref, dy_ref, dtheta_ref]
if ref_traj_switch == 1:
    v_max = 0.3
    road_width = 0.6
    t, xr, ur, ddxr = ref_trajectory_mecanum.ref_trajectory_mecanum_wv(L, v_max, road_width,dt,T)    
elif ref_traj_switch == 2:
    t, xr, ur, ddxr = ref_trajectory_mecanum.ref_trajectory_mecanum_xy(T)

#State Space equation with reference trajectory
A = np.zeros((3,3,len(t)))
B = np.zeros((3,3,len(t)))
for i in range(0, len(t)):    
    A[:,:,i]= np.eye(3) + np.array([ [0, 0, (-ur[0][i]*np.sin(xr[2][i]) - ur[1][i]*np.cos(xr[2][i]))*T],
                                     [0, 0, ( ur[0][i]*np.cos(xr[2][i]) - ur[1][i]*np.sin(xr[2][i]))*T], 
                                     [0, 0,                                                          0] ])
    B[:,:,i]= np.array([ [np.cos(xr[2][i])*T, -np.sin(xr[2][i])*T, 0],
                         [np.sin(xr[2][i])*T,  np.cos(xr[2][i])*T, 0],
                         [0,                   0,                  T] ])

#MPC parameters
n = A.shape[0]
m = B.shape[1]
Q = Qs*np.eye(n)
R = Rs*np.eye(m)

#zmp constraint
Df = D *np.ones(len(t)-N)
Dr = D *np.ones(len(t)-N)
Lf = L *np.ones(len(t)-N)
Lr = L *np.ones(len(t)-N)

###################################################################################################################################
############################################  dLQR Simulation  ####################################################################
###################################################################################################################################

#initial parameters
x_tilt_lqr_current = x0 - xr[:,0]
u_tilt_lqr = np.zeros(m)
k_lqr_set = np.zeros((m,n, len(t)-N))
s_lqr_set = np.zeros((m,n, len(t)-N))
e_lqr_set = np.zeros((m,n, len(t)-N))
u_tilt_lqr_set = np.zeros((m,len(t)-N))
x_tilt_lqr_set = np.zeros((n,len(t)-N))
u_lqr =  np.zeros((m,len(t)-N))
x_state_lqr = np.zeros((n,len(t)-N))
dx_state_lqr = np.zeros((n,len(t)-N))
tactime = np.zeros(len(t)-N)

#############__Start Simulation!!!__#############
for k in generator(0, len(t)-N):   
    starttime = time.time()

    #dLQR    
    k_lqr_set[:,:,k], s_lqr_set[:,:,k], e_lqr_set[:,:,k] = dlqr(A[:,:,k],B[:,:,k],Q,R)
    u_tilt_lqr = -np.matmul(k_lqr_set[:,:,k], x_tilt_lqr_current)
    
    #Applying input to the plant 
    u_tilt_lqr_set[:,k] = u_tilt_lqr.copy()    
    x_tilt_lqr_set[:,k] = x_tilt_lqr_current.copy()     
    
    u_lqr[:,k] = u_tilt_lqr + ur[:,k]
    x_state_lqr[:,k] = x_tilt_lqr_current + xr[:,k]    
    
    if model_switch == 2: #linear model    
        if k < len(t)-N-1:  
            x_tilt_lqr_current_next = np.matmul(A[:,:,k], x_tilt_lqr_current) + np.matmul(B[:,:,k], u_tilt_lqr)   
            dx_state_lqr[:,k+1] = ( ( x_tilt_lqr_current_next+xr[:,k+1] ) - x_state_lqr[:,k] )/T  #backward derivative
            x_tilt_lqr_current = x_tilt_lqr_current_next.copy()         

    tactime[k] = time.time() - starttime    

#Calculate ddx_state
ddx_state_lqr = np.zeros((n,len(t)-N))
ddx_state_lqr[:,1:] = (dx_state_lqr[:,1:] - dx_state_lqr[:,:-1])/T    #backward derivative

###################################################################################################################################
###############################################  Data Plot  #######################################################################
###################################################################################################################################
dataplot(t, N, Lf, Lr, Df, Dr, h2, xr, x_state_lqr, ur, u_lqr, x_tilt_lqr_set, thr_state_plus, thr_state_minus, u_tilt_lqr_set, thr_input_plus, thr_input_minus, ddx_state_lqr, mu, tactime, switch_state, switch_input, switch_slip)
    









































###################################################################################################################################
############################################  MPC Simulation  #####################################################################
###################################################################################################################################

#initial parameters
x_tilt_lqr_current = x0 - xr[:,0]
u_tilt_lqr = np.zeros(m)
k_lqr_set = np.zeros((m,n, len(t)-N))
s_lqr_set = np.zeros((m,n, len(t)-N))
e_lqr_set = np.zeros((m,n, len(t)-N))
u_tilt_lqr_set = np.zeros((m,len(t)-N))
x_tilt_lqr_set = np.zeros((n,len(t)-N))
u_lqr =  np.zeros((m,len(t)-N))
x_state_lqr = np.zeros((n,len(t)-N))
dx_state_lqr = np.zeros((n,len(t)-N))


#############__Start Simulation!!!__#############
for k in generator(0, len(t)-N):   
    
    #LQR
    k_lqr_set[:,:,k], s_lqr_set[:,:,k], e_lqr_set[:,:,k] = control.lqr(A[:,:,k],B[:,:,k],Q,R)
    u_tilt_lqr = -np.matmul(x_tilt_lqr_current, k_lqr_set[:,:,k])
    
    #Applying input to the plant 
    u_tilt_lqr_set[:,k] = u_tilt_lqr.copy()    
    x_tilt_lqr_set[:,k] = x_tilt_lqr_current.copy()     
    
    u_lqr[:,k] = u_tilt_lqr + ur[:,k]
    x_state_lqr[:,k] = x_tilt_lqr_current + xr[:,k]    
    
    if model_switch == 2: #linear model    
        if k < len(t)-N-1:  
            x_tilt_lqr_current_next = np.matmul(A[:,:,k], x_tilt_lqr_current) + np.matmul(B[:,:,k], u_tilt_lqr)   
            dx_state_lqr[:,k+1] = ( ( x_tilt_lqr_current_next+xr[:,k+1] ) - x_state_lqr[:,k] )/T  #backward derivative
            x_tilt_lqr_current = x_tilt_lqr_current_next.copy()  
        
    

#Calculate ddx_state
ddx_state_lqr = np.zeros((n,len(t)-N))
ddx_state_lqr[:,1:] = (dx_state_lqr[:,1:] - dx_state_lqr[:,:-1])/T    #backward derivative
