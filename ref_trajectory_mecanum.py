import numpy as np
import math
from for_gen import generator

def ref_trajectory_mecanum_wv(L, v_max, road_width,dt,T):
    ###### reference trajectory
    circle_r = (road_width - L*math.cos(math.pi/4)) / (1-math.cos(math.pi/4))    
    t = np.arange(0, math.pi *circle_r/(2*abs(v_max)) +0.5, dt)
    ##############################

    ######make input reference
    num = 20
    v_ref = v_max * np.concatenate((-np.cos(np.array(math.pi * np.arange(0,1+1/num,1/num)))/2+0.5, 1*np.ones((len(t) - 2*(num+1))), +np.cos(math.pi * np.arange(0,1+1/num,1/num))/2+0.5))
    w_ref = 1*(v_max/circle_r) * np.concatenate((-np.cos(math.pi * np.arange(0,1+1/num,1/num))/2+0.5, 1*np.ones(len(t) -2*(num+1)), +np.cos(math.pi * np.arange(0,1+1/num,1/num))/2+0.5))

    t = np.arange(0, math.pi *circle_r/(2*abs(v_max)) +0.5 +3, dt) 
    ##############################

    ######local_coordinate
    v_ref = np.concatenate((np.zeros(150), v_ref, np.zeros(150)))
    w_ref = np.concatenate((np.zeros(150), w_ref, np.zeros(150)))
    vy_ref = v_ref*0.8;
    # w_ref=zeros(size(w_ref));
    # vy_ref = v_ref*0.5.*sin(2*pi*t);
    # vy_ref = zeros(size(v_ref));
    ##############################

    ######make state reference, global coordinate
    theta_ref = np.zeros(len(t))
    for i in generator(0, len(t)-1):
        theta_ref[i+1] = theta_ref[i] + w_ref[i+1] * dt    

    dx_ref = v_ref*np.cos(theta_ref) - vy_ref*np.sin(theta_ref)
    dy_ref = v_ref*np.sin(theta_ref) + vy_ref*np.cos(theta_ref)
    x_ref = np.zeros(len(t))
    y_ref = np.zeros(len(t))
    for i in generator(0, len(t)-1):
        x_ref[i+1] = x_ref[i] + dx_ref[i+1] * dt
        y_ref[i+1] = y_ref[i] + dy_ref[i+1] * dt

    ddx_ref = np.zeros(len(t))
    ddy_ref = np.zeros(len(t))
    ddtheta_ref = np.zeros(len(t))
    ddx_ref[1:len(dx_ref)] = (dx_ref[1:]-dx_ref[:-1])/T
    ddy_ref[1:len(dx_ref)] = (dy_ref[1:]-dy_ref[:-1])/T
    ddtheta_ref[1:len(dx_ref)] = (w_ref[1:]-w_ref[:-1])/T
    ##############################

    ######define state and input reference
    xr = np.vstack([x_ref, y_ref, theta_ref])
    ur = np.vstack([v_ref, vy_ref, w_ref])

    ddxr = np.vstack([ddx_ref, ddy_ref, ddtheta_ref])
    ##############################

    return t, xr, ur, ddxr 


def ref_trajectory_mecanum_xy(T):
    ######reference trajectory 
    tb = np.arange(0, 10+T, T)
    dx_ref = -1.0*2*math.pi/10*np.cos(2*math.pi*tb/10)
    dy_ref = -1.0*4*math.pi/10*np.cos(4*math.pi*tb/10)

    tz=3
    t = np.arange(0, 10+tz+T, T)
    num=19
    dx_ref = np.concatenate((np.zeros(int((tz*(1/T)-2*(num+1)/2)/2)), 1.0*2*math.pi/10*(np.cos(2*math.pi*np.arange(0,0.5,1/num))/2-0.5), dx_ref, 1.0*2*math.pi/10*(-np.cos(2*math.pi*np.arange(0,0.5,1/num))/2-0.5), np.zeros(int((tz*(1/T)-2*(num+1)/2)/2))))
    dy_ref = np.concatenate((np.zeros(int((tz*(1/T)-2*(num+1)/2)/2)), 1.0*4*math.pi/10*(np.cos(2*math.pi*np.arange(0,0.5,1/num))/2-0.5), dy_ref, 1.0*4*math.pi/10*(-np.cos(2*math.pi*np.arange(0,0.5,1/num))/2-0.5), np.zeros(int((tz*(1/T)-2*(num+1)/2)/2))))

    if T == 0.01:
        theta_ref = np.concatenate((np.arctan2(dy_ref[141],dx_ref[141])*np.ones(141),  np.arctan2(dy_ref[141:-141],dx_ref[141:-141]),  np.arctan2(dy_ref[1160], dx_ref[1160]*np.ones(141))))
    elif T == 0.1:
        theta_ref = np.concatenate((np.arctan2(dy_ref[6],dx_ref[6])*np.ones(6),  np.arctan2(dy_ref[6:-6],dx_ref[6:-6]),  np.arctan2(dy_ref[125], dx_ref[125]*np.ones(6))))
    elif T == 0.05:
            theta_ref = np.concatenate((np.arctan2(dy_ref[21],dx_ref[21])*np.ones(21),  np.arctan2(dy_ref[21:-21],dx_ref[21:-21]),  np.arctan2(dy_ref[240], dx_ref[240]*np.ones(21))))

    # theta_ref = zeros(1,length(dx_ref));
    # theta_ref = atan2(dy_ref,dx_ref);
    theta_ref = np.unwrap(theta_ref)

    x_ref  = np.zeros(len(t))
    y_ref  = np.zeros(len(t))
    for i in generator(0, len(t)-1):
        x_ref[i+1] = x_ref[i] + dx_ref[i+1] * T
        y_ref[i+1] = y_ref[i] + dy_ref[i+1] * T


    ddx_ref  = np.zeros(len(t))
    ddy_ref  = np.zeros(len(t))
    ddx_ref[1:] = (dx_ref[1:]-dx_ref[:-1])/T
    ddy_ref[1:] = (dy_ref[1:]-dy_ref[:-1])/T

    dtheta_ref = np.zeros(len(t))
    dtheta_ref[1:] = (theta_ref[1:]-theta_ref[:-1])/T
    # dtheta_ref = (dx_ref*ddy_ref - dy_ref*ddx_ref) / (dx_ref*dx_ref +  dy_ref*dy_ref)

    ddtheta_ref = np.zeros(len(t))
    ddtheta_ref[1:] = (dtheta_ref[1:]-dtheta_ref[:-1])/T
    ##############################

    ######define state and input reference
    xr = np.vstack([x_ref, y_ref, theta_ref])
    ur = np.vstack([dx_ref, dy_ref, dtheta_ref])

    ddxr = np.vstack([ddx_ref, ddy_ref, ddtheta_ref])
    ##############################
    
    return t, xr, ur, ddxr

