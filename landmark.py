import cv2 
import numpy as np
import mediapipe as mp

capture = cv2.VideoCapture(0)

mphands = mp.solutions.hands
Hands =mphands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,min_detection_confidence=0.7,min_tracking_confidence=0.6)
drawlms = mp.solutions.drawing_utils
 
while True:
    data, frame = capture.read()
    
    # converting from BGR to RGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    oframe = frame.copy()

    result = Hands.process(frame)
    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            drawlms.draw_landmarks(frame,hlm,mphands.HAND_CONNECTIONS)

    # concatenating two frames 
    LandMarkedVideo = cv2.flip( np.concatenate((oframe,frame), axis = 1),1)
    
    cv2.imshow('Marked',cv2.cvtColor(LandMarkedVideo,cv2.COLOR_RGB2BGR))
    

    
    if cv2.waitKey(1) == 27:
        break


capture.release()
cv2.destroyAllWindows()
