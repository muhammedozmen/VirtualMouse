import cv2
import numpy as np
import HandTracking as ht
import time
import autopy

w_cam, h_cam = 640, 480
frame_r = 100  # Frame Reduction
smoothening = 6

p_time = 0
p_locX, p_locY = 0, 0  # Previous Locations
c_locX, c_locY = 0, 0  # Current Locations

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
detector = ht.handDetector(maxHands=2)
wScr, hScr = autopy.screen.size()
# print(wScr,hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check that which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frame_r, frame_r), (w_cam - frame_r, h_cam - frame_r),
                      (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert Coordinates
            x3 = np.interp(x1, (frame_r, w_cam - frame_r), (0, wScr))
            y3 = np.interp(y1, (frame_r, h_cam - frame_r), (0, hScr))

            # 6. Smoothen Values
            c_locX = p_locX + (x3 - p_locX) / smoothening
            c_locY = p_locY + (y3 - p_locY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - c_locX, c_locY)
            cv2.circle(img, (x1, y1), 9, (255, 0, 255),
                       cv2.FILLED)  # The circle on the finger when moving mode is activated
            p_locX, p_locY = c_locX, c_locY

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:

            # 9. Find distance between fingers
            length, img, line_info = detector.findDistance(8, 12,
                                                           img)  # Find distance between 2 fingers when clicking mode is activated
            print(length)

            # 10. Click mouse if distance is short
            if length < 21:
                cv2.circle(img, (line_info[4], line_info[5]),
                           9, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame Rate
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
