#go to mediapipe website to see which point corresponds to which number 

import cv2 as cv
import mediapipe as mp
import time

def rescaleFrame(frame, scale = 0.75):
    # images, videos, and live video
    width = int(frame.shape[1] * scale)    # frame.shape[1] is basically width
    height = int(frame.shape[0] * scale) # frame.shape[0] is basically height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


# cap = cv.VideoCapture('Notes/Advanced/PoseVideos/dancing.mp4')
cap = cv.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    # img = rescaleFrame(img, 0.2) #if you import videos 

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv.imshow("Image", img) 
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break