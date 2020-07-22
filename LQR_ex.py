import numpy as np
import control

A = np.array([ [0, 1],
               [-1, -2] ])

B = np.array([[0, 1], [1, 0]]).reshape(-1,2)

Q = 1*np.eye(2)
R = 0.1*np.eye(2)

K, S, E = control.lqr(A,B,Q,R)

print(K)
print(S)
print(E)