"""MediaPipe Qt6 GUI application"""
import os
import signal
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc.udp_client import SimpleUDPClient

from assets.mainwindow import Ui_MainWindow

signal.signal(signal.SIGINT, signal.SIG_DFL)


class CameraThread(QThread):

    changePixmap = Signal(QImage)
    handLandmarksResults = Signal(vision.HandLandmarkerResult)

    def __init__(self, camera_index=0, flip_horizontal=True):
        super().__init__()
        self.camera_index = camera_index
        self.flip_horizontal = flip_horizontal
        self.is_running = False
        self.calculate_fps_prev_time = 0.0
        self.width = 640
        self.height = 480
        # landmark constants
        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.handedness_text_color = (88, 205, 54)          # vibrant green

    def run(self):
        task_path = os.path.join(os.path.dirname(__file__), 'assets/hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=task_path,
                                          # delegate=python.BaseOptions.Delegate.GPU
                                          )
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)

        # open webcam
        self.is_running = True
        cap = cv2.VideoCapture(0)
        with vision.HandLandmarker.create_from_options(options) as detector:
            while self.is_running:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # Flip the image horizontally for a later selfie-view display
                if self.flip_horizontal:
                    image = cv2.flip(image, 1)
                # Convert the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to pass by
                # reference.
                image.flags.writeable = False
                # convert image to mp image for analysis
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                results = detector.detect(mp_image)
                # send signal with calculated mediapipe results for further processing outside this
                # thread
                self.handLandmarksResults.emit(results)
                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                annotated_image = self.draw_landmarks_on_image(image, results)
                # Calculate and draw fps
                cv2.putText(annotated_image,
                            f"fps: {self.calculate_fps():.2f}",
                            (10, 40),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.5,
                            (255, 255, 255),
                            2)
                # convert image from CV2 to Qt
                rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line,
                                              QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(h, w, Qt.AspectRatioMode.KeepAspectRatio)
                self.changePixmap.emit(p)
            cap.release()

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()            # type: ignore
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(                                    # type: ignore
                    x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(                                 # type: ignore
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,                                   # type: ignore
                solutions.drawing_styles.get_default_hand_landmarks_style(),        # type: ignore
                solutions.drawing_styles.get_default_hand_connections_style())      # type: ignore

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.margin

            # Draw handedness (left or right hand) on the image.
            hand_str = f"{handedness[0].category_name}"
            if self.flip_horizontal:
                if hand_str == 'Left':
                    hand_str = 'Right'
                else:
                    hand_str = 'Left'
            cv2.putText(annotated_image, hand_str,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size,
                        self.handedness_text_color,
                        self.font_thickness,
                        cv2.LINE_AA)
        return annotated_image

    def stop(self):
        self.is_running = False

    def calculate_fps(self):
        new_time = time.time()
        fps = 1 / (new_time - self.calculate_fps_prev_time)
        self.calculate_fps_prev_time = new_time
        return fps


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cam_id = 0
        self.flip_horizontal = True

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # start processing in a thread
        self.cam_thread = CameraThread(camera_index=self.cam_id,
                                       flip_horizontal=self.flip_horizontal)
        self.cam_thread.changePixmap.connect(self.setImage)
        self.cam_thread.finished.connect(self.clearCam)

        # OSC client
        self.osc_address = self._ui.oscAddress.text()
        self.osc_port = self._ui.oscPort.text()
        self.osc_client = SimpleUDPClient(self.osc_address, int(self.osc_port))
        self._ui.oscPort.editingFinished.connect(
            lambda: self.restart_osc_client(self._ui.oscAddress.text(), self._ui.oscPort.text()))

        self.cam_thread.handLandmarksResults.connect(self.sendHandLandmarkerOSC)

        # connect resize handlers
        # self._ui.camLabel.resizeEvent

        # connect buttons
        self._ui.startButton.clicked.connect(self.startCamera)
        self._ui.stopButton.clicked.connect(self.stopCamera)

        print(self._ui.camLabel.frameRect())

    @Slot(QImage)
    def setImage(self, image):
        # update image
        self._ui.camLabel.setPixmap(QPixmap.fromImage(image))

    @Slot(vision.HandLandmarkerResult)
    def sendHandLandmarkerOSC(self, results):
        # Send the landmarks through OSC
        # print(results)
        if self.osc_client:
            if (hasattr(results, 'hand_landmarks') and hasattr(results, 'handedness') and
                    len(results.handedness) > 0):
                hand = results.handedness[0][0].category_name
                if self.flip_horizontal:
                    if hand == 'Left':
                        hand = 'Right'
                    else:
                        hand = 'Left'
                for other_hand in ['Left', 'Right']:
                    if other_hand != hand:
                        self.osc_client.send_message(
                            f"/mediapipe_gui/hands/{other_hand.lower()}/tracking", False)
                if len(results.hand_landmarks) > 0:
                    self.osc_client.send_message(f"/mediapipe_gui/hands/{hand.lower()}/tracking",
                                                 True)
                    for idx, landmark in enumerate(results.hand_landmarks[0]):
                        self.osc_client.send_message(
                            f"/mediapipe_gui/hands/{hand.lower()}/{str(idx)}/xyz",
                            [landmark.x, landmark.y, landmark.z])
                        self.osc_client.send_message(
                            f"/mediapipe_gui/hands/{hand.lower()}/{str(idx)}/visibility",
                            landmark.visibility)
                        self.osc_client.send_message(
                            f"/mediapipe_gui/hands/{hand.lower()}/{str(idx)}/presence",
                            landmark.presence)
            else:
                for hand in ['Left', 'Right']:
                    self.osc_client.send_message(f"/mediapipe_gui/hands/{hand.lower()}/tracking",
                                                 False)

    def startCamera(self):
        self.cam_thread.start()

    def stopCamera(self):
        self.cam_thread.stop()
        self.cam_thread.quit()
        self.cam_thread.wait()

    def clearCam(self):
        # clear pixmap
        self._ui.camLabel.clear()
        self._ui.camLabel.setText('press START')

    def restart_osc_client(self, address, port):
        self.osc_address = address
        self.osc_port = int(port)
        self.osc_client = SimpleUDPClient(self.osc_address, self.osc_port)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.stopCamera)
    window.show()
    sys.exit(app.exec())
