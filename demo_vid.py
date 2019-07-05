import cv2
import numpy as np

import matplotlib.pyplot as plt


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('calle1.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

num_frames = 0
frames = []

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    num_frames += 1
	
    frames.append(frame)
	
    # Display the resulting frame
    #cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #  break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

print(num_frames)

frames = np.array(frames)
print(frames.size)
print(frames.shape)
print(type(frames))
#print(frames[0])

imgScale = 0.2
frame1 = cv2.resize(frames[0],(int(frames[0].shape[1]*imgScale),int(frames[0].shape[0]*imgScale)))
frame2 = cv2.resize(frames[20],(int(frames[20].shape[1]*imgScale),int(frames[20].shape[0]*imgScale)))

cv2.imshow('Frame1',frame1)
cv2.imshow('Frame2',frame2)

# Press Q on keyboard to  exit
cv2.waitKey(0)
cv2.destroyAllWindows()
