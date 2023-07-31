import numpy as np

iX = 0                      # position in X
iXv = 1                     # Velocity in X
iY = 2                      # Position in Y
iYv = 3                      # velocity in Y

NUM_VARS = iYv + 1


class KF:
    """
    A class to track a signal using Kalman filter

    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    predict(dt:float)
        computes state estimate and covariance estimate.
    """
    def __init__(self, acc_x_dir : float,
                       acc_y_dir : float,
                       std_acc : float,
                       x_std_meas : float,
                       y_std_meas : float,
                       ) -> None:

        self._x = np.zeros(NUM_VARS)

        self.u = np.array([[acc_x_dir],[acc_y_dir]])

        # Process Noise Magnitude
        self.std_acc = std_acc

    

        # mean of state Gaussian R.V
        self._x[iX]   = 0
        self._x[iXv]  = 0
        self._x[iY]   = 0
        self._x[iYv]  = 0
        
        # covariance of state GRV
        self._P = np.eye(NUM_VARS)

        self.R = np.array([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
    
    def predict(self, dt: float) -> None:
        # x = F * x 
        # P = F P Ft + G Gt a
        # F = np.eye(NUM_VARS)
        

        #  Process Noise Covariance 
        Q = np.array([
            [(dt**4)/4,  (dt**3)/2, 0 , 0],
            [(dt**3)/2, dt**2, 0, 0],
            [0,0,(dt**4)/4, (dt**3)/2],
            [0,0,(dt**3)/2,dt**2 ]
            ]) * self.std_acc **2
        
        # state transition model 
        F = np.array([
                      [1, dt, 0, 0 ], 
                      [0,  1, 0, 0 ],
                      [0,  0, 1, dt],
                      [0,  0, 0, 1 ]
                      ])
        
       

        # control-input model 
        G = np.array([
            [0.5*dt**2, 0],
            [dt,        0],
            [0, 0.5*dt**2],
            [0,        dt]
        ])


        new_x = F.dot(self._x.reshape((4,1))) + G.dot(self.u)

        new_P = F.dot(self._P).dot(F.T) + Q

        self._x = new_x
        self._P = new_P

    def update(self,measured_x: float, measured_y: float):
        # y = z - Hx
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + ky
        # P = (I -kH) * P

        # self._x = np.reshape(self._x,(4,1))

        H = np.array([[1,0,0,0],[0,0,1,0]]).reshape((2,4))

        # Measurements with noise
        z = np.array([measured_x,measured_y]).reshape((2,1)) 
        
        # Innovation Residual
        y = z - H.dot(self._x.reshape((4,1)))
        
        # Innovation Covariance 
        S = H.dot(self._P).dot(H.T) + self.R

        # Kalman Gain
        K = (self._P).dot(H.T).dot(np.linalg.inv(S))
        

        kdoty = K.dot(y)

        # Updated x (State)
        new_x = (self._x).reshape((4,1))+ kdoty

        # Updated Covariance
        new_P = (np.eye(4) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x
        

    @property
    def pos(self):
        return (self._x[0],self._x[2])

    @property
    def vel(self):
        return self._x[1]

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x
