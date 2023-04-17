import cv2
import numpy as np
import os
import shutil
import argparse
import mediapipe as mp

GESTURES = ['raise', 'wave', 'other']
SEQUENCE_LENGTH = 60
SELECTED_GESTURE = 'other'
DATASET_PATH = 'dataset_v2'

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image_rgb.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image_rgb)                 # Make prediction
    image_rgb.flags.writeable = True                   # Image is now writeable
    return results

def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results, draw_left_hand_only=False):
    if draw_left_hand_only:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=2)
                                  )

    else:
        # Draw face connections
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                         )
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)
                                 )
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2)
                                 )
        # Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2)
                                 )

def extract_keypoints(results, with_fase=False):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    if with_fase:
       return np.concatenate([pose, face, lh, rh])
    return np.concatenate([pose, lh, rh])


def data_creation():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    is_record = False
    current_sequence_length = 0
    exp_num = len(os.listdir(os.path.join(DATASET_PATH, SELECTED_GESTURE)))
    save_dir = os.path.join(DATASET_PATH, SELECTED_GESTURE, str(exp_num))

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Make detections
            results = mediapipe_detection(frame, holistic)
            # Draw landmarks
            draw_styled_landmarks(frame, results)

            if not is_record:
                text = ['Create data for gesture: ' + SELECTED_GESTURE.upper(),
                        'To start record gesture press \'s\',',
                        'Each gesture consists of ' + str(SEQUENCE_LENGTH) + ' frames',
                        'To stop record press \'e\'',
                        'Repeat if necessary',
                        'To exit press \'q\'']
                info_rect = np.zeros_like(frame)
                for i, line in enumerate(text):
                    cv2.putText(info_rect, line, (frame_width//4, frame_height//2 - 10*len(text) + i*20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)

                alpha = 0.2
                frame_blended = cv2.addWeighted(frame, alpha, info_rect, 1 - alpha, 0)
                frame[frame_height//2 - 15*len(text):frame_height//2 + 15*len(text), frame_width//5:4*frame_width//5] = \
                    frame_blended[frame_height//2 - 15*len(text):frame_height//2 + 15*len(text), frame_width//5:4*frame_width//5]
            else:
                keypoints = extract_keypoints(results)

                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                np.save(os.path.join(save_dir, str(current_sequence_length)), keypoints)

                current_sequence_length += 1
                if current_sequence_length == SEQUENCE_LENGTH:
                    print('data saved in ' + save_dir)
                    current_sequence_length = 0
                    exp_num += 1
                    save_dir = os.path.join(DATASET_PATH, SELECTED_GESTURE, str(exp_num))

            cv2.imshow('WebCam', frame)
            # Break gracefully
            key = cv2.waitKey(1)
            if key == ord('q'):
                if current_sequence_length != 0:
                    shutil.rmtree(save_dir)
                break
            elif key == ord('s'):
                is_record = True
            elif key == ord('e'):
                is_record = False
                if current_sequence_length != 0:
                    shutil.rmtree(save_dir)
                    current_sequence_length = 0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='crate_data.py')
    parser.add_argument('--gesture', type=str, default='other', help='name of gesture to create data for')
    parser.add_argument('--dataset_path', type=str, default='dataset_v2', help='path to dataset folder')
    opt = parser.parse_args()
    SELECTED_GESTURE = opt.gesture
    DATASET_PATH = opt.dataset_path

    if SELECTED_GESTURE in GESTURES and os.path.isdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, SELECTED_GESTURE)):
            os.mkdir(os.path.join(DATASET_PATH, SELECTED_GESTURE))
        data_creation()
    else:
        parser.print_help()
        if not os.path.isdir(DATASET_PATH):
            print('Dataset folder: ' + DATASET_PATH + ' not exists')
        else:
            print('Gesture ' + SELECTED_GESTURE.upper() + ' not exists')
