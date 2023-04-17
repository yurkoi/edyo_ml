# import necessary packages
import time
import cv2
import numpy as np
import mediapipe as mp
from create_data import mediapipe_detection, draw_styled_landmarks, extract_keypoints, SEQUENCE_LENGTH, GESTURES
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# define model
model = Sequential()
model.add(LSTM(60*2, return_sequences=True, activation='relu', input_shape=(60,258)))
#model.add(Dropout(0.2))
model.add(LSTM(60, return_sequences=False, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dense(len(GESTURES), activation='softmax'))
# Load the gesture recognizer model
model.load_weights('models/gesture_wave_v2_without_dropout.h5')

colors = [(217,25,16), (117,245,16), (16,17,245)]
intervals = []
for i in range(len(GESTURES)):
    intervals.append([])
# for visualization
def prob_viz(res, input_frame, intervals=None):
    pairs = []
    if not intervals is None:
        for i, gesture in enumerate(intervals):
            pairs.append([])
            for frame in gesture:
                if pairs[i]:
                    if frame - pairs[i][-1][-1] <= 5:
                        pairs[i][-1][-1] = frame
                    else:
                        print(frame, pairs[i][-1][-1])
                        pairs[i].append([frame, frame])
                else:
                    pairs[i].append([frame, frame])
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, GESTURES[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        text = ''
        for j in pairs[num]:
            if j[1] - j[0] > 5:
                text += str(j[0]) + ' - ' + (str(j[1]) if j[1] != j[0] else 'curr') + '; '
        text = text[-50:]
        cv2.putText(output_frame, text, (120, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[num], 2,
                    cv2.LINE_AA)

    return output_frame


# Initialize the webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('gans_hands_processed.mp4',cv2.VideoWriter_fourcc(*'XVID'), 60, (frame_width,frame_height))


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

preTime = 0
curTime = 0

sequence = []
sentence = []
threshold = 0.6
past_frames = []
frame_num = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        preTime = time.time()
        # Read each frame from the webcam
        _, frame = cap.read()
        #
        x, y, c = frame.shape

        # Flip the frame vertically
        #frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        results = mediapipe_detection(framergb, holistic)

        # Draw landmarks
        draw_styled_landmarks(frame, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(GESTURES[np.argmax(res)])
            intervals[np.argmax(res)].append(frame_num)

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if GESTURES[np.argmax(res)] != sentence[-1]:
                        sentence.append(GESTURES[np.argmax(res)])
                else:
                    sentence.append(GESTURES[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            frame = prob_viz(res, frame, intervals)


        curTime = time.time()
        fps = 1 / (curTime - preTime)
        cv2.putText(frame,str(int(fps)),(x//2,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)

        #out.write(frame)

        if y > 720:
            frame = cv2.resize(frame, (1280, 720))
        # Show the final output
        cv2.imshow("Output", frame)
        frame_num += 1

        if cv2.waitKey(1) == ord('q'):
            break

# release the webcam and destroy all active windows
cap.release()
#out.release()

cv2.destroyAllWindows()