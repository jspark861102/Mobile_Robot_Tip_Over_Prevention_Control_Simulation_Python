import numpy as np
from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
import math
import random
import copy
import ref_trajectory_mecanum
import Batch_formulation
from QP_constraints import constraints
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

T = 0.01
dt = copy.copy(T)

# Design Parameters
N = 15 # Batch size
Qs = 5
Rs = 1

#1:nonlinear model, 2:linear model
model_switch = 2

#reference trajectory 1:wv, 2:xy
ref_traj_switch = 2

#1:on 0:off zmp constraint
switch_zmp = 1

# %1:on, 0:off slip constraint
# switch_slip = 1;
mu = np.array([0.46, 0.46, 0])

#initial state : [x y theta]
# x0 = [0 0 0]'
if ref_traj_switch == 1:
    x0 = np.array([0.1, 0.2, math.pi/8])
elif ref_traj_switch == 2:
    x0 = np.array([0.0, 0.0, 0])

#1:on 0:off input constraint
switch_input = 1
mag = 2.0
thr_input_plus = np.array([mag, mag, mag]) 
thr_input_minus = np.array([-mag, -mag, -mag ])
thr_input = np.concatenate((thr_input_plus, -thr_input_minus))

#1:on 0:off state constraint
switch_state = 0
thr_state_plus =  np.array([ 0.008,  0.008,  0.008])
thr_state_minus = np.array([-0.008, -0.008, -0.008])
thr_state = np.concatenate((thr_state_plus, -thr_state_minus))

#system parameters
#mass
g = 9.81
#wheel base
D = 0.5
#width
L = 0.3
#z value from global origin to link2 mass center
h2 = 0.6

#reference trajectory xr=[x_ref, y_ref, theta_ref], ur=[dx_ref, dy_ref, dtheta_ref]
if ref_traj_switch == 1:
    v_max = 0.3
    road_width = 0.6
    t, xr, ur, ddxr = ref_trajectory_mecanum.ref_trajectory_mecanum_wv(L, v_max, road_width,dt,T)    
elif ref_traj_switch == 2:
    t, xr, ur, ddxr = ref_trajectory_mecanum.ref_trajectory_mecanum_xy(T)


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

#Optimization
x0_tilt = x0 - xr[:,0]
x_tilt_current = x0_tilt; 
dx_state = np.zeros((3,len(t)-N))

u_tilt = np.zeros(m)
u_tilt_set = np.zeros((m,len(t)-N))
x_tilt_set = np.zeros((n,len(t)-N))
u =  np.zeros((m,len(t)-N))
x_state = np.zeros((n,len(t)-N))
#x_tilt_m1_dummy = np.zeros(3)

#define QP
QP_MPC = SQProblem(m*N, n*N)
options = Options()
QP_MPC.setOptions(options)
#QP_MPC.printOptions()

for k in range(0, len(t)-N):   
    #MPC formulation
    cur_state = k
    Sx, Su, Qb, Rb = Batch_formulation.BF(A, B, N, Q, R, cur_state)

    H = np.matmul(np.matmul(Su.T, Qb), Su) + Rb
    F = np.matmul(np.matmul(Sx.T, Qb), Su) 
    Y = np.matmul(np.matmul(Sx.T, Qb), Sx)    

    #constraints
    if k == 0 or k == 1:
        x_tilt_m1 = x0_tilt.copy()
    else:
        x_tilt_m1 = x_tilt_m1_dummy.copy()
    
    G, E, wu, wl, wu_input, wl_input = constraints(cur_state, xr, ddxr, x_tilt_current, x_tilt_m1, N, A, B, h2, D, L, T, switch_zmp, switch_input, thr_input, switch_state, thr_state, mu);
    
    #QP    
    nWSR = np.array([100])
    if k == 0:
        QP_MPC.init(H, 2*np.matmul( F.T, x_tilt_current), G, wl_input, wu_input, wl+np.matmul(E, x_tilt_current), wu+np.matmul(E, x_tilt_current), nWSR)        
    else:
        QP_MPC.hotstart(H, 2*np.matmul( F.T, x_tilt_current), G, wl_input, wu_input, wl+np.matmul(E, x_tilt_current), wu+np.matmul(E, x_tilt_current), nWSR)        

    QP_MPC.getPrimalSolution(u_tilt)
    print('k =',k)
    print('u_tilt =',u_tilt)
    
    #Simulation
    print(u_tilt)
    print(u_tilt_set[:,k])
    u_tilt_set[:,k] = u_tilt.copy()    
    x_tilt_set[:,k] = x_tilt_current.copy() 
    x_tilt_m1_dummy = x_tilt_current.copy()  
    
    u[:,k] = u_tilt + ur[:,k]
    x_state[:,k] = x_tilt_current + xr[:,k]

    if model_switch == 1: #nonlinear model
        if k < len(t)-N-1:  
            dx_state[:,k+1] = np.array([u[0,k]*np.cos(x_state[2,k]) - u[1,k]*np.sin(x_state[2,k]), u[0,k]*np.sin(x_state[2,k]) + u[1,k]*np.cos(x_state[2,k]), u[2,k]])  #backward derivative    
            x_state_next = x_state[:,k] + dx_state[:,k+1]*T    #backward derivative
            x_tilt_current = x_state_next - xr[:,k+1]       
    
    elif model_switch == 2: #linear model    
        if k < len(t)-N-1:  
            x_tilt_current_next = np.matmul(A[:,:,k], x_tilt_current) + np.matmul(B[:,:,k], u_tilt)   
            dx_state[:,k+1] = ( ( x_tilt_current_next+xr[:,k+1] ) - x_state[:,k] )/T  #backward derivative
            x_tilt_current = x_tilt_current_next.copy()  

#zmp        
ddx_state = np.zeros((n,len(t)-N))
ddx_state[:,1:] = (dx_state[:,1:] - dx_state[:,:-1])/T    #backward derivative

# local zmp
J = np.zeros((n,n,len(t)-N))
dJ = np.zeros((n,n,len(t)-N))
ddv = np.zeros((n,len(t)-N))

for i in range(0, len(t)-N):
    J[:,:,i] = np.array([ [np.cos(x_state[2,i]), -np.sin(x_state[2,i]),  0],
                          [np.sin(x_state[2,i]),  np.cos(x_state[2,i]),  0],
                          [                   0,                     0,  1] ])
    dJ[:,:,i] =np.array([ [-np.sin(x_state[2,i])*dx_state[2,i], -np.cos(x_state[2,i])*dx_state[2,i],  0],
                          [ np.cos(x_state[2,i])*dx_state[2,i], -np.sin(x_state[2,i])*dx_state[2,i],  0],
                          [                                  0,                                   0,  0] ])
    ddv[:,i] = np.matmul( np.linalg.inv(J[:,:,i]) , ddx_state[:,i] - np.matmul(dJ[:,:,i], u[:,i]) )

du = np.zeros((m,len(t)-N))
du[:,1:] = (u[:,1:]-u[:,:-1])/T;


######################################################### plot ############################################################################
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(xr[0,:],xr[1,:], color = 'red', label = 'reference trajectory')
plt.plot(x_state[0,:],x_state[1,:], color = 'blue', linestyle = '--', label = 'real trajectory')
'''
plt.plot(xr[0,0],xr[1,0], color = 'red', linestyle = 'o')
plt.plot(xr[0,-1],xr[1,-1], color = 'red', linestyle = '^')
plt.plot(x_state[0,0],x_state[1,0], color = 'blue', linestyle = 'o')
plt.plot(x_state[0,-1],x_state[1,-1], color = 'black', linestyle = '^')
'''
#plt.tight_layout()
plt.title('trajectory')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')


plt.subplot(2,2,2)
plt.plot(t,ur[0,:], color = 'blue', label = 'vx ref')
plt.plot(t[:-N],u[0,:], color = 'blue', linestyle = '--', label = 'vx real')
plt.plot(t,ur[1,:], color = 'red', label = 'vy ref')
plt.plot(t[:-N],u[1,:], color = 'red', linestyle = '--', label = 'vy real')
plt.plot(t,ur[2,:], color = 'black', label = 'w ref')
plt.plot(t[:-N],u[2,:], color = 'black', linestyle = '--', label = 'w real')
#plt.tight_layout()
plt.title('input')
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('u')
plt.legend(loc='upper right')

plt.subplot(2,2,3)
plt.plot(t[:-N],x_tilt_set[0,:], color = 'blue', label = 'x')
plt.plot(t[:-N],x_tilt_set[1,:], color = 'red', label = 'y')
plt.plot(t[:-N],x_tilt_set[2,:], color = 'black', label = 'theta')
if switch_state == 1:
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_state_plus[0], thr_state_plus[0]]), color = 'blue', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_state_plus[1], thr_state_plus[1]]), color = 'red', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_state_plus[2], thr_state_plus[2]]), color = 'black', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_state_minus[0], thr_state_minus[0]]), color = 'blue', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_state_minus[1], thr_state_minus[1]]), color = 'red', linestyle = '--')    
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_state_minus[2], thr_state_minus[2]]), color = 'black', linestyle = '--')
#plt.tight_layout()
plt.title('x tilt')
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('x tilt')
plt.legend(loc='upper right')

plt.subplot(2,2,4)
plt.plot(t[:-N],u_tilt_set[0,:], color = 'blue', label = 'vx')
plt.plot(t[:-N],u_tilt_set[1,:], color = 'black', label = 'vy')
plt.plot(t[:-N],u_tilt_set[2,:], color = 'red', label = 'w')
if switch_input == 1:
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_input_plus[0], thr_input_plus[0]]), color = 'blue', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_input_minus[0], thr_input_minus[0]]), color = 'blue', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_input_plus[1], thr_input_plus[1]]), color = 'red', linestyle = '--')
    plt.plot(np.array([t[0], t[-N]]), np.array([thr_input_minus[1], thr_input_minus[1]]), color = 'red', linestyle = '--')
#plt.tight_layout()
plt.title('u tilt')
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('u tilt')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(t[:-N], ddx_state[0,:], color = 'blue', label = 'ddx')
plt.plot(t[:-N], ddx_state[1,:], color = 'red', label = 'ddy')
plt.plot(t[:-N], (np.abs(np.cos(x_state[2,:]))* D/2 + np.abs(np.sin(x_state[2,:]))* L/2)*g/h2, color = 'blue', linestyle = '--', label = 'zmpx constraint')
plt.plot(t[:-N], (np.abs(np.sin(x_state[2,:]))* D/2 + np.abs(np.cos(x_state[2,:]))* L/2)*g/h2, color = 'red', linestyle = '--', label = 'zmpy constraint')
plt.plot(t[:-N], mu[0]*g *np.ones(len(t)-N), color = 'black', linestyle = ':', label = 'slip constraint')
plt.plot(t[:-N], (np.abs(np.cos(x_state[2,:]))*-D/2 + np.abs(np.sin(x_state[2,:]))*-L/2)*g/h2, color = 'blue', linestyle = '--')
plt.plot(t[:-N], (np.abs(np.sin(x_state[2,:]))*-D/2 + np.abs(np.cos(x_state[2,:]))*-L/2)*g/h2, color = 'red', linestyle = '--')
plt.plot(t[:-N], -mu[0]*g *np.ones(len(t)-N), color = 'black', linestyle = ':')
plt.plot(t[:-N],  mu[1]*g *np.ones(len(t)-N), color = 'black', linestyle = ':')
plt.plot(t[:-N], -mu[1]*g *np.ones(len(t)-N), color = 'black', linestyle = ':')
#plt.tight_layout()
plt.title('constraint')
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('zmp')
plt.legend(loc='upper right')

plt.show()


