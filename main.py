import cv2
import math
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector
#mediapipe, cvzone

cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)

# hand detection
detector = HandDetector(detectionCon = 0.8, maxHands = 1)
# x - raw distance, y - value in cm
# the difference in neither linear nor consistent
# https://en.wikiversity.org/wiki/Algebra_II/Polynomial_Functions#:~:text=A%20polynomial%20function%20is%20a,3x2%20%2B%204x%20%2D%205

x = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff = np.polyfit(x, y, 2) #polynominal func 2 degrees / y = Ax^2 + Bx + C


while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)
    #hands,img = detector.findHands(img, draw=False)
    if hands:
        lmList = hands[0]['lmList']
        #print(lmlist)

        x,y,w,h = hands[0]['bbox']

        x1,y1 = lmList[5]
        x2,y2 = lmList[17]
        # https://google.github.io/mediapipe/solutions/hands.html
        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))  # rotation
        A,B,C = coff
        distanceCM = A*distance**2 + B*distance + C

        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 3)
        #print(distanceCM, distance)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5,y-10))


    cv2.imshow('Image', img)
    cv2.waitKey(1)