# @markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
import cv2
import random
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np


class GestureRecognition:
    base_options = python.BaseOptions(
        model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=1,
                                              )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    list_of_gesture = ['Closed_Fist', 'Open_Palm',
                       'Thumb_Down', 'Thumb_Up']

    def __init__(self):
        self.gesture_challenge_list = self.set_gesture_challenge()
        self.skip_gesture_list = []
        self.verified_image = None

    # STEP 1: Import the necessary modules.

    def set_gesture_challenge(self):
        return random.sample(GestureRecognition.list_of_gesture, 3)

    # remove gesture from list based on sequence of gesture
    def update_challenge_state(self, gesture):

        # make sure index not out of range
        if len(self.gesture_challenge_list) == 0:
            return

        if gesture == None:
            return

        elif gesture in self.skip_gesture_list:
            return

        elif gesture == self.gesture_challenge_list[0]:
            self.gesture_challenge_list.pop(0)
            self.skip_gesture_list.append(gesture)

    def check_challenge_complete(self):
        return len(self.gesture_challenge_list) == 0

    def recognize_hand_gesture(self, image):
        recognition_result = GestureRecognition.recognizer.recognize(
            image)

        # if there is no gesture detected
        if len(recognition_result.gestures) == 0:
            return None

        return recognition_result.gestures[0][0].category_name

    def load_image_from_file(self, frame):
        # Convert OpenCV frame to RGB format
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.asarray(frame))

    def reset_challenge(self):
        self.gesture_challenge_list = self.set_gesture_challenge()
        self.skip_gesture_list = []

    def access_verified_image(self):
        temp = self.verified_image
        self.verified_image = None
        return temp

# rc = GestureRecognition()


# # get video input from webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow('Frame', frame)
#         image = rc.load_image_from_file(frame)

#         gesture_detected = rc.recognize_hand_gesture(image)
#         print(str(rc.gesture_challenge_list) + " | " +
#               str(rc.skip_gesture_list) + " | " + str(gesture_detected))

#         rc.update_challenge_state(gesture_detected)

#         if rc.check_challenge_complete():
#             print("Challenge Complete")
#             break

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
