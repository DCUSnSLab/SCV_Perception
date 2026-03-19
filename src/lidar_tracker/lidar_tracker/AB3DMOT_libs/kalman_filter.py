import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D, info):
        # bbox3D는 이제 Numpy 배열 [h, w, l, x, y, z, ry, score] 입니다.
        
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=7)       
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])

        self.kf.P[7:,7:] *= 1000. 	# state uncertainty
        self.kf.P *= 10.
        self.kf.Q[7:,7:] *= 0.01

        # [수정된 부분 1] 객체 변환 없이 배열을 바로 사용 (앞 7개 값: h,w,l,x,y,z,ry)
        self.kf.x[:7] = bbox3D[:7].reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info 

    def update(self, bbox3D, info): 
        """ 
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1          
        self.still_first = False
        self.info = info              
        
        # [수정된 부분 2] 객체 변환 없이 배열을 바로 사용
        self.kf.update(bbox3D[:7])

    def predict(self):       
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.first_continuing_hit = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7, ))