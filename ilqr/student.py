import numpy as np
from scipy import linalg
from numpy import sin, cos

class CartPole:
    def __init__(self, env, x=None,max_linear_velocity=2, max_angular_velocity=np.pi/3):
        if x is None:
            x = np.zeros(2)
        self.env = env
        
    def getA(self, u, x=None):
        if x is None:
            x = self.x
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        force = u
        dt = self.env.tau
        A = np.eye(4)

        A[0,1]= dt
        A[1,2]= ((gravity)/(length*((4/3)-(masspole/total_mass))))
        A[2,3]= dt
        A[3,2]= ((gravity)/(length*((4/3)-(masspole/total_mass)))) 
        
        return A
        
    def getB(self, x=None):
        if x is None:
            x = self.x
        
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        dt = self.env.tau

        B = np.zeros((4,1))

        B[1]= 1/ total_mass
        B[3]=-1/(length*((4/3)-(masspole/total_mass)))

        return B



def lqr(A, B, T=100):
    K = np.zeros((1,4))
    Q = np.eye(4)
    R = np.eye(1)
    P = np.zeros((4,4))
    for t in range(T):
        K = -linalg.inv(R+B.T@P@B)@B.T@P@A
        P = (Q + K.T@R@K + (A+B@K).T@P@(A+B@K))
    return K
    
