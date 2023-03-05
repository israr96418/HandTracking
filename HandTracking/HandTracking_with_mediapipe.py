import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        print(results.multi_hand_landmarks)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    # print(id, lm)
                    h,w, c = image.shape
                    #find image central position
                    cx, cy= int(lm.x * w) , int(lm.y*h)
                    print(id,cx,cy)
                    if id ==0:
                        cv2.circle(image, (cx, cy), 14,(255,0,255),cv2.FILLED)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('HandTracking', cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
