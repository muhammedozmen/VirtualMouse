import cv2
import numpy as np
import HandTracking as ht
import time
import autopy

w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0
detector = ht.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
#print(wScr,hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) !=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)

        # 3. Check that which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert Coordinates
            x3 = np.interp(x1, (0, w_cam), (0, wScr))
            y3 = np.interp(y1, (0, h_cam), (0, hScr))

            # 6. Smoothen Values
            # 7. Move Mouse
            autopy.mouse.move(x3, y3)
        # 8. Both Index and middle fingers are up : Clicking Mode
        # 9. Find distance between fingers
        # 10. Click mouse if distance is short

    # 11. Frame Rate
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)