# import necessary packages
import time
import cv2
import numpy as np
import mediapipe as mp
from create_data import mediapipe_detection, draw_styled_landmarks
import tensorflow as tf
from tensorflow.keras.models import load_model
from DD_Net import build_DD_Net, convert_mediapipe_for_DDNet, C, LABELS


DD_Net = build_DD_Net(C.frame_l,C.joint_n,C.joint_d,C.feat_d,C.clc_coarse,C.filters)
# Load the gesture recognizer model
DD_Net.load_weights('models/DD_Net_large_14.h5')

colors = [(217,25,16), (117,245,16), (16,17,245)]
intervals = []
for i in range(len(LABELS)):
    intervals.append([])
def prob_viz(res, input_frame):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        color = (0, int(prob*255), 255 - int(255*(prob - 0.5)))
        cv2.rectangle(output_frame, (0, 60 + num* 40), (3*100, 90 + num * 40), (0,0,0), -1)
        cv2.rectangle(output_frame, (0, 60 + num* 40), (3*int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_frame, LABELS[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.rectangle(output_frame, (2*output_frame.shape[0]//3, 0), (2*output_frame.shape[0]//3 + 300, 30), (0, 0, 0), -1)
    cv2.putText(output_frame, LABELS[np.argmax(res)], (2*output_frame.shape[0]//3, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# Initialize the webcam
cap = cv2.VideoCapture('test_DDNet.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('test_DDNet_res.mp4',cv2.VideoWriter_fourcc(*'XVID'), frame_fps, (frame_width,frame_height))


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

preTime = 0
curTime = 0

X_1, X_0 = [], []
sentence = []
threshold = 0.6
past_frames = []
frame_num = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        preTime = time.time()
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        #
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        results = mediapipe_detection(framergb, holistic)

        # Draw landmarks
        draw_styled_landmarks(frame, results, True)

        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))

        keypoints_0, keypoints_1 = convert_mediapipe_for_DDNet(lh)

        X_0.append(keypoints_0)
        X_1.append(keypoints_1)
        X_0 = X_0[-C.frame_l:]
        X_1 = X_1[-C.frame_l:]

        if len(X_0) == C.frame_l:
            res = DD_Net.predict([np.expand_dims(np.array(X_0), axis=0),
                                  np.expand_dims(np.array(X_1), axis=0)])[0]
            print(LABELS[np.argmax(res)])
            #intervals[np.argmax(res)].append(frame_num)


            # Viz probabilities
            frame = prob_viz(res, frame)



        curTime = time.time()
        fps = 1 / (curTime - preTime)
        cv2.putText(frame,str(int(fps)),(x//2,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
        
        out.write(frame)

        if y > 720:
            frame = cv2.resize(frame, (1280, 720))
        # Show the final output
        cv2.imshow("Output", frame)
        frame_num += 1

        if cv2.waitKey(1) == ord('q'):
            break

# release the webcam and destroy all active windows
cap.release()
out.release()

cv2.destroyAllWindows()