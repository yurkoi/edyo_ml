
import tflite_runtime.interpreter as tflite
from scipy.spatial.distance import cdist
import mediapipe as mp
import numpy as np
import logging

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GestureRecognizer(object):
    def __init__(self, model_weights_path, flip_image=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, max_num_hands=2):
        # mediapipe hands detector
        self.keypoint_detector = mp.solutions.hands.Hands(min_detection_confidence=min_detection_confidence,
                                                          min_tracking_confidence=min_tracking_confidence,
                                                          max_num_hands=max_num_hands)
        self.flip_image = flip_image

        # model for predicting gesture
        try:
            self.classifier = tflite.Interpreter(model_path=model_weights_path)
            self.classifier.allocate_tensors()
            self.input_details = self.classifier.get_input_details()
            self.output_details = self.classifier.get_output_details()
        except Exception as e:
            logging.error(e)

        # dummy gesture model input
        self.buffer_input1 = np.zeros((1, 32, 210))
        self.buffer_input2 = np.zeros((1, 32, 21, 3))
        # dummy gesture model output
        self.last_prediction = [0]*6
        self.last_prediction[2] = 1

        self.class_names = ['No gesture', 'Swipe Right', 'Swipe Left',
                            'Swipe Up', 'Swipe Down', 'Shake']

    def get_class_name(self, class_id):
        if class_id >= len(self.class_names):
            return 'incorrect id'
        return self.class_names[class_id]

    # find hands and predict gesture using image
    def update(self, image, last_input=None):
        if image is None:
            hands = []
        else:
            # find hand keypoints
            hands = self.find_hands(image, return_one_hand=True)
        # prepare gesture model input
        if len(hands):
            input_1, input_2 = self.create_JCD(hands[0]), hands[0]
        else:
            input_1, input_2 = np.zeros(210), np.zeros((21, 3))
        if last_input:
            buffer_input1 = last_input[0]
            buffer_input1[0, :-1] = buffer_input1[0, 1:]
            buffer_input1[0, -1] = input_1

            buffer_input2 = last_input[1]
            buffer_input2[0, :-1] = buffer_input2[0, 1:]
            buffer_input2[0, -1] = input_2
        else:
            buffer_input1 = self.buffer_input1
            buffer_input1[0, :-1] = buffer_input1[0, 1:]
            buffer_input1[0, -1] = input_1

            buffer_input2 = self.buffer_input2
            buffer_input2[0, :-1] = buffer_input2[0, 1:]
            buffer_input2[0, -1] = input_2

        self.buffer_input1[0, :-1] = buffer_input1[0, 1:]
        self.buffer_input1[0, -1] = input_1
        self.buffer_input2[0, :-1] = self.buffer_input2[0, 1:]
        self.buffer_input2[0, -1] = input_2

        # make gesture prediction
        prediction = self.predict_DDNet(buffer_input1, buffer_input2)[0]
        self.last_prediction = prediction

        return self.get_1_last_gesture(prediction), [buffer_input1, buffer_input2]

    def set_last_input(self, buffer_input1, buffer_input2):
        self.buffer_input1 = buffer_input1
        self.buffer_input2 = buffer_input2

    def get_last_input(self):
        return [self.buffer_input1, self.buffer_input2]

    def get_1_last_gesture(self, prediction=None):
        if prediction is None:
            prediction = self.last_prediction
        idx = np.argsort(prediction)[-1]
        return {'predicted_id': int(idx),
                'predicted_label': self.get_class_name(idx),
                'confidence': float(prediction[idx])}

    # return n prediction with highest confidence
    def get_last_gesture(self, top=3):
        result = []
        idxs = np.argsort(self.last_prediction)[-top:]
        for i in idxs:
            result.append({'predicted_id': i,
                           'predicted_label': self.get_class_name(i),
                           'confidence': self.last_prediction[i]})
        return result

    def predict_DDNet(self, model_input1, model_input2):
        self.classifier.set_tensor(self.input_details[0]['index'], model_input1.astype(np.float32))
        self.classifier.set_tensor(self.input_details[1]['index'], model_input2.astype(np.float32))
        self.classifier.invoke()

        prediction = self.classifier.get_tensor(self.output_details[0]['index'])
        return prediction

    # return array of found hands
    def find_hands(self, image, return_one_hand=True):
        input_image = np.fliplr(image) if self.flip_image else image
        # input_image.flags.writeable = False  # Image is no longer writeable
        result = self.keypoint_detector.process(input_image)  # Make prediction
        # input_image.flags.writeable = True
        if result.multi_hand_landmarks:
            # convert mediapipe output to array (num_hands X 21 X 3)
            if return_one_hand:
                scores = [i.classification[0].score for i in result.multi_handedness]
                hand = np.array([[res.x, res.y, res.z] for res in
                                 result.multi_hand_landmarks[np.argmax(scores)].landmark])
                hands = np.expand_dims(hand, axis=0)
            else:
                hands = np.array([[[lmrk.x, lmrk.y, lmrk.z] for lmrk in hand.landmark] for hand in
                                  result.multi_hand_landmarks])
        else:
            return []

        return hands

    @staticmethod
    def create_JCD(hand):
        # create second part of DDNet model input
        dist = cdist(hand, hand)
        JCD = []
        for i, d in enumerate(dist):
            JCD.extend(dist.T[i, i + 1:])
        return JCD
