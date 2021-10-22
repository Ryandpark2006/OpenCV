import cv2 as cv
import numpy as np

frameWidth = 640 
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)   #brightness

def empty(a):
    pass

cv.namedWindow('HSV')
cv.resizeWindow('HSV', 640, 240)
cv.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv.createTrackbar("VALUE Max", "HSV", 255, 255, empty)


while True:
    success, img = cap.read()
    cv.imshow("Result", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h_min = cv.getTrackbarPos("HUE Min", "HSV")
    s_min = cv.getTrackbarPos("SAT Min", "HSV")
    v_min = cv.getTrackbarPos("VALUE Min", "HSV")
    h_max = cv.getTrackbarPos("HUE Max", "HSV")
    s_max = cv.getTrackbarPos("SAT Max", "HSV")
    v_max = cv.getTrackbarPos("VALUE Max", "HSV")    
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv.inRange(imgHSV, lower, upper)  # gives us filtered out image of that color

    cv.imshow("Mask", mask)  #keep colors that we don't want as black

cap.release()
cv.destroyAllWindows()