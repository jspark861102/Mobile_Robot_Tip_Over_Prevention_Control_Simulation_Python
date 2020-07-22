import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def dataplot(t, N, L, D, h2, xr, x_state, ur, u, x_tilt_set, thr_state_plus, thr_state_minus, u_tilt_set, thr_input_plus, thr_input_minus, ddx_state, mu, tactime, switch_state, switch_input):
    g = 9.81

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

    plt.figure(3)
    plt.plot(t[:-N], tactime)
    #plt.tight_layout()
    plt.title('tactime')
    plt.grid(True)
    plt.xlabel('t(sec)')
    plt.ylabel('tactime')

    plt.show()

