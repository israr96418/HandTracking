import time

import cv2
from HandTracking import HandTrackingModule as HTM

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector = HTM.handDetector()
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring for empty frame")
        continue
    # if we pass draw = False it cannot draw landmark and connection b/w them
    img = detector.findHands(img, draw=True)
    # if we pass draw = False it cannot find the position of each landmark
    lmlist = detector.findPosition(img,draw=True)
    if len(lmlist) != 0:
        print(lmlist[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the camera
cap.release()
cv2.destroyAllWindows()