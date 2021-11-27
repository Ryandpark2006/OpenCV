import cv2 as cv
import numpy as np
import time

# HSV summary: 
# Hue is the color portion of the model (0 - 360 degrees)
# Saturation is the amount of grey in a particular color (0-100%) - as you get nearer to 0 and more grey is introduced, it produces a faded effect
# Value (brightness) works in conjunction with saturation and describes brightness or intensity of the color (0 - 100%) - where 0 is completely black and 100 is is the brightest and reveals the most color


# Pseudocode:
# Capture and store the background frame
# Detect the defined color using color detection and segmentation algorthm 
# Segment out the defined colored part by generating a mask
# Generate the final augmented output to create effect

# Prep for output video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Recording and caching the background for each frame 
cap = cv.VideoCapture(0)

time.sleep(3)
count = 0
background = 0

for i in range(60):
    success, background = cap.read()
# flip here
background = np.flip(background, axis = 1)

# Detecting the color portion in each frame
while(cap.isOpened()):
    success, img = cap.read()
    if not success:
        break
    count+=1
    #flip here
    img = np.flip(img, axis = 1)


    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Generate masks to detect red color 
    lower_red = np.array([0, 117, 100])
    upper_red = np.array([179, 255, 255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)

    lower_gray = np.array([77, 22, 16])
    upper_gray = np.array([149, 255, 148])
    mask2 = cv.inRange(hsv, lower_gray, upper_gray)

    mask1 = mask1 + mask2


# Replace red portion with mask
    mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv.morphologyEx(mask1, cv.MORPH_DILATE, np.ones((3,3), np.uint8))

    mask2 = cv.bitwise_not(mask1)

    res1 = cv.bitwise_and(img, img, mask=mask2)
    res2 = cv.bitwise_and(background, background, mask=mask1)

# Final Output
    finalOutput = cv.addWeighted(res1, 1, res2, 1, 0)
    out.write(finalOutput)
    cv.imshow("magic", finalOutput)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()