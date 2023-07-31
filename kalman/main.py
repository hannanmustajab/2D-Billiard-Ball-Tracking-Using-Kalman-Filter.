# load libraries
from cmath import sqrt
import cv2
import numpy as np
from Kalman import KF
from cvzone.ColorModule import ColorFinder
import cvzone
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 


# load video
# Initialize the stream
ball__list = ['ball3','ball4']
cap = cv2.VideoCapture(f'{ball__list[1]}.mov')

# get fps
fps = cap.get(cv2.CAP_PROP_FPS)  

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Duration
duration = frame_count/fps


# extract frames and track object
hsvVals = {'hmin': 0, 'smin': 197, 'vmin': 181,
           'hmax': 179, 'smax': 255, 'vmax': 255}


# Variables
positionListX = []
positionListY = []
predListX = []
predListY = []


# color finder
# Create color finder object
myColorFinder = ColorFinder(False)



# 
"""
Create Kalman Filter object with the following parameters:
    @acc_x_dir : 0
    @acc_y_dir : 0
    @std_acc : 1
    @x_std_meas : 0.1
    @y_std_meas : 0.1
"""
kf = KF(acc_x_dir=0.1,acc_y_dir=0.1,std_acc=1 ,x_std_meas=3,y_std_meas=3)

errorListX = []
errorListY = []


# List to store means and covariances.
mus = []
covs = []

# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and filename.
# out = cv2.VideoWriter('v2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    # grab the image
    success, img = cap.read()
    
    # If the stream ends, break the loop.
    if not success: break
    
    # find color of the ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=1)
    
    # Add text to the feed 
    cv2.putText(img, 'Predicted', (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(img, 'Actual', (100,55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # Start Kalman Predict
    kf.predict(dt=1) 
    
    if contours:
        posX = contours[0]['center'][0]
        positionListX.append(posX)
        posY = contours[0]['center'][1]
        positionListY.append(posY)

        # Update Covariance Matrix and Mean
        covs.append(kf.cov)
        mus.append(kf.mean)

        # Update Position ( Measurement step )
        kf.update(measured_x=posX,measured_y=posY)
        
        # Show predicted position
        cv2.circle(img, (int(kf.pos[0][0]),int(kf.pos[1][0])), 4, (255, 255, 0), 5)

        # # show 95% confidence values
        # cv2.circle(img, (int(kf.mean[0] - np.sqrt(kf.cov[0][0])),int(kf.mean[2] - np.sqrt(kf.cov[2][2]))), 3, (255, 121, 0), 5)

        # cv2.circle(img, (int(kf.mean[0] + np.sqrt(kf.cov[0][0])),int(kf.mean[2] + np.sqrt(kf.cov[2][2]))), 3, (255, 124, 0), 5)

        # Add predicted positions to the list
        predListX.append((kf.pos[0][0]))
        predListY.append((kf.pos[1][0]))

        # Show actual Position
        cv2.circle(img, (posX,posY), 5, (0, 255, 255), 1)
        errorListX.append(np.sqrt((kf.mean[0] - posX)**2))
        errorListY.append(np.sqrt((kf.mean[2] - posY)**2))
    

     
    
    # out.write(img)
    # Display the image
    # cv2.imshow('Image', imgContours)
    cv2.imshow('Predicted', img)
    cv2.waitKey(100)

# Release everything if job is finished
cap.release()
# out.release()
cv2.destroyAllWindows()

# time linspace 
time = np.linspace(0,duration,num=len(mus))


plt.subplot(4,1,1)
plt.title('Position X - axis wrt Time')
plt.plot(time,[mu[0] for mu in mus],'b',label='Predicted Position')
plt.plot(time,[mu[0] - 2*np.sqrt(cov[0,0]) for mu,cov in zip(mus,covs)],'--')
plt.plot(time,[mu[0] + 2*np.sqrt(cov[0,0]) for mu,cov in zip(mus,covs)],'--')
plt.plot(time,positionListX,'xg',label='Actual Positon')
plt.legend(loc="upper left")


plt.subplot(4,1,2)
plt.title('Position Y - axis wrt Time')
plt.plot(time,[mu[2] for mu in mus],'b',label='Predicted Position')
plt.plot(time,[mu[2] - 2*np.sqrt(cov[2,2]) for mu,cov in zip(mus,covs)],'--')
plt.plot(time,[mu[2] + 2*np.sqrt(cov[2,2]) for mu,cov in zip(mus,covs)],'--')
plt.plot(time,positionListY,'xg',label='Actual Position')
plt.legend(loc="upper left")




plt.subplot(4,1,3)
plt.title('Velocity')
plt.plot(time,[np.abs(mu[1]) for mu in mus],'--',label='Velocity X')
plt.plot(time,[np.abs(mu[3]) for mu in mus],'--',label='Velocity Y')
plt.legend(loc="upper left")

plt.subplot(4,1,4)
plt.title('Error')
plt.plot(time,[mu for mu in errorListX],'--',label='Error X')
plt.plot(time,[mu for mu in errorListY],'--',label='Error Y')
plt.legend(loc="upper left")




plt.show()
plt.ginput(1)


