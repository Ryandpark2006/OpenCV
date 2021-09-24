import cv2 as cv
import numpy as np

# Access web cam
frameWidth = 640 
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)   #brightness

# Color Detection
# list of colors we have that we want to detect
myColors = [[0,99,144,83,255,255],
            [19,90,133,179,255,255],
            [29,90,0,161,255,255]]    # orange, blue, red, purple
myColorValues = [[51,153,255],    #BGR
                 [0,0,255],
                 [255,0,255]]
myPoints = []  #[x, y, colorID]

# def empty(a):
#     pass

# cv.namedWindow('HSV')
# cv.resizeWindow('HSV', 640, 240)
# cv.createTrackbar("HUE Min", "HSV", 0, 179, empty)
# cv.createTrackbar("SAT Min", "HSV", 0, 255, empty)
# cv.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
# cv.createTrackbar("HUE Max", "HSV", 179, 179, empty)
# cv.createTrackbar("SAT Max", "HSV", 255, 255, empty)
# cv.createTrackbar("VALUE Max", "HSV", 255, 255, empty)


# while True:
#     success, img = cap.read()
#     cv.imshow("Result", img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#     imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#     h_min = cv.getTrackbarPos("HUE Min", "HSV")
#     s_min = cv.getTrackbarPos("SAT Min", "HSV")
#     v_min = cv.getTrackbarPos("VALUE Min", "HSV")
#     h_max = cv.getTrackbarPos("HUE Max", "HSV")
#     s_max = cv.getTrackbarPos("SAT Max", "HSV")
#     v_max = cv.getTrackbarPos("VALUE Max", "HSV")    
#     print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask = cv.inRange(imgHSV, lower, upper)  # gives us filtered out image of that color

#     cv.imshow("Mask", mask)  #keep colors that we don't want as black


def findColor(img,myColors,myColorValues):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6]) 
        mask = cv.inRange(imgHSV, lower, upper)
        x,y = getContours(mask)
        cv.circle(imgResult,(x,y),10,myColorValues[count],cv.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count +=1
        # cv.imshow(str(color[0]), mask)
    return newPoints

def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>500:
            # cv.drawContours(imgResult, cnt, -1, (255,0,0), 3)
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv.boundingRect(approx)
    return x+w//2,y   #top point of tip point and center too

def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv.circle(imgResult,(point[0],point[1]),10, myColorValues[point[2]],cv.FILLED)
        

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)
    cv.imshow("Result", imgResult)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


