import time
import os
import HandTrackingModule as htm
import cv2

wCam = 720
hCam = 540

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (150,150))
    overlayList.append(image)

detector = htm.handDetector(maxHands=1)

pTime = 0
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []

        #Thumb fingers
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Remainning fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        totalFingers = fingers.count(1)

        img[:150,:150] = overlayList[totalFingers - 1]
        cv2.rectangle(img, (0, 200), (150, 410), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (30, 360), cv2.FONT_HERSHEY_PLAIN,10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (370,70), cv2.FONT_ITALIC, 2, (255,255,155), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()