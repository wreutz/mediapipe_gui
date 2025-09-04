"""MediaPipe Qt6 GUI application"""

import math
import os
import signal
import sys
import time

import cv2
import matplotlib.pyplot as plt
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

OSC_BASE_NAME = '/mediapipe_gui'


class CameraThread(QThread):

    changePixmap = Signal(QImage)
    mediapipeResults = Signal(tuple)

    def __init__(self, camera_index=0, flip_horizontal=True, ai_model='face_lm'):
        super().__init__()
        self.ai_model = ai_model
        self.camera_index = camera_index
        self.flip_horizontal = flip_horizontal
        self.is_running = False
        self.calculate_fps_prev_time = 0.0
        self.width = 640
        self.height = 480
        # landmark constants
        self.margin = 10    # pixels
        self.row_size = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.handedness_text_color = (88, 205, 54)          # vibrant green
        self.object_text_color = (255, 0, 0)                # red

    def run(self):
        if self.ai_model == 'hand_lm':
            task_path = os.path.join(os.path.dirname(__file__),
                                     'assets/mp_tasks/hand_landmarker.task')
            base_options = python.BaseOptions(model_asset_path=task_path,
                                              # delegate=python.BaseOptions.Delegate.GPU
                                              )
            options = vision.HandLandmarkerOptions(base_options=base_options,
                                                   num_hands=2,
                                                   min_hand_detection_confidence=0.6,
                                                   min_hand_presence_confidence=0.6,
                                                   min_tracking_confidence=0.6,
                                                   )
            # open webcam
            self.is_running = True
            cap = cv2.VideoCapture(0)
            with vision.HandLandmarker.create_from_options(options) as detector:
                while self.is_running:
                    self.capture_and_analyze(cap, detector)
                cap.release()
        elif self.ai_model == 'gesture':
            task_path = os.path.join(os.path.dirname(__file__),
                                     'assets/mp_tasks/gesture_recognizer.task')
            base_options = python.BaseOptions(model_asset_path=task_path,
                                              # delegate=python.BaseOptions.Delegate.GPU
                                              )
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                # running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                # canned_gestures_classifier_options=vision.GestureRecognizerOptions(
                #     max_results=-1,
                #     score_threshold=0.3,)
            )
            # open webcam
            self.is_running = True
            cap = cv2.VideoCapture(0)
            with vision.GestureRecognizer.create_from_options(options) as detector:
                while self.is_running:
                    self.capture_and_analyze(cap, detector)
                cap.release()
        elif self.ai_model == 'pose':
            task_path = os.path.join(os.path.dirname(__file__),
                                     'assets/mp_tasks/pose_landmarker_full.task')
            base_options = python.BaseOptions(model_asset_path=task_path,
                                              # delegate=python.BaseOptions.Delegate.GPU
                                              )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                # running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            # open webcam
            self.is_running = True
            cap = cv2.VideoCapture(0)
            with vision.PoseLandmarker.create_from_options(options) as detector:
                while self.is_running:
                    self.capture_and_analyze(cap, detector)
                cap.release()
        elif self.ai_model == 'face':
            task_path = os.path.join(os.path.dirname(__file__),
                                     'assets/mp_tasks/blaze_face_short_range.tflite')
            base_options = python.BaseOptions(model_asset_path=task_path,
                                              # delegate=python.BaseOptions.Delegate.GPU
                                              )
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                # running_mode=vision.RunningMode.LIVE_STREAM,
                min_detection_confidence=0.5,
                min_suppression_threshold=0.3,
            )
            # open webcam
            self.is_running = True
            cap = cv2.VideoCapture(0)
            with vision.FaceDetector.create_from_options(options) as detector:
                while self.is_running:
                    self.capture_and_analyze(cap, detector)
                cap.release()
        elif self.ai_model == 'face_lm':
            task_path = os.path.join(os.path.dirname(__file__),
                                     'assets/mp_tasks/face_landmarker.task')
            base_options = python.BaseOptions(model_asset_path=task_path,
                                              # delegate=python.BaseOptions.Delegate.GPU
                                              )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                # running_mode=vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            # open webcam
            self.is_running = True
            cap = cv2.VideoCapture(0)
            with vision.FaceLandmarker.create_from_options(options) as detector:
                while self.is_running:
                    self.capture_and_analyze(cap, detector)
                cap.release()
        elif self.ai_model == 'object':
            task_path = os.path.join(os.path.dirname(__file__),
                                     'assets/mp_tasks/efficientdet_lite0.tflite')
            base_options = python.BaseOptions(model_asset_path=task_path,
                                              # delegate=python.BaseOptions.Delegate.GPU
                                              )
            options = vision.ObjectDetectorOptions(base_options=base_options,
                                                   # running_mode=vision.RunningMode.LIVE_STREAM,
                                                   # max_results=-1,
                                                   score_threshold=0.3,
                                                   )
            # open webcam
            self.is_running = True
            cap = cv2.VideoCapture(0)
            with vision.ObjectDetector.create_from_options(options) as detector:
                while self.is_running:
                    self.capture_and_analyze(cap, detector)
                cap.release()
        else:
            print('not implemented task!')

    def capture_and_analyze(self, cap, detector):
        """Capture live video and run mediapipe sending signals."""
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            return

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
        if self.ai_model in ['gesture', ]:
            results = detector.recognize(mp_image)
        else:
            results = detector.detect(mp_image)
        # send signal with calculated mediapipe results for further processing outside
        # this thread
        self.mediapipeResults.emit((self.ai_model, results))
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        match self.ai_model:
            case 'hand_lm':
                annotated_image = self.draw_hand_landmarks_and_gestures_on_image(image, results)
            case 'gesture':
                annotated_image = self.draw_hand_landmarks_and_gestures_on_image(image, results)
            case 'pose':
                annotated_image = self.draw_poses_on_image(image, results)
            case 'face':
                annotated_image = self.draw_face_boxes_on_image(image, results)
            case 'face_lm':
                annotated_image = self.draw_face_landmarks_on_image(image, results)
            case 'object':
                annotated_image = self.draw_object_boxes_on_image(image, results)
            case _:
                annotated_image = image
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
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

    def draw_hand_landmarks_and_gestures_on_image(self, rgb_image, detection_result):
        """Draw hand landmarks on image and (if available) gesture recognitions."""
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        gesture_list = []
        gestures_str_list = []
        if hasattr(detection_result, 'gestures'):
            gesture_list = detection_result.gestures
            for gesture in gesture_list:
                gestures_str_list.append(f"{gesture[0].category_name} ({gesture[0].score:.2f})")
            gestures_str = ', '.join(gestures_str_list)
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
            cv2.putText(annotated_image,
                        hand_str,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size,
                        self.handedness_text_color,
                        self.font_thickness,
                        cv2.LINE_AA)
        # visualize gestures
        if len(gestures_str_list) > 0:
            height, width, _ = annotated_image.shape
            cv2.putText(annotated_image,
                        f"gestures: {gestures_str}",
                        (20, height - 40),
                        cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size,
                        (156, 156, 255),
                        self.font_thickness,
                        cv2.LINE_AA)
        return annotated_image

    def draw_object_boxes_on_image(self, rgb_image, detection_result):
        """Draw boxes and labels for recognized objects on the live image."""
        annotated_image = np.copy(rgb_image)
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.object_text_color, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.margin + bbox.origin_x,
                             self.margin + self.row_size * 2 + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.font_size * 2, self.object_text_color, self.font_thickness * 3)
        return annotated_image

    def draw_poses_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()            # type: ignore
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(                                    # type: ignore
                    x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(                                 # type: ignore
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,                                    # type: ignore
                solutions.drawing_styles.get_default_pose_landmarks_style())        # type: ignore
        return annotated_image

    def _normalized_to_pixel_coordinates(self, normalized_x, normalized_y, image_width,
                                         image_height):
        """Converts normalized value pair to pixel coordinates."""
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))
        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def draw_face_boxes_on_image(self, rgb_image, detection_result):
        """Draw boxes around detected faces on the image."""
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, (125, 255, 125), 3)
            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                                    width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)      # type: ignore
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.margin + bbox.origin_x,
                             self.margin + self.row_size + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.font_size, (125, 255, 125), self.font_thickness)
        return annotated_image

    def draw_face_landmarks_on_image(self, rgb_image, detection_result):
        """Draw face landmarks on image."""
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()            # type: ignore
            face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(  # type: ignore
                x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(                                 # type: ignore
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,            # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles                 # type: ignore
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(                                 # type: ignore
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,               # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles                 # type: ignore
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(                                 # type: ignore
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,                 # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles                 # type: ignore
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name
                                  for face_blendshapes_category in face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score
                                   for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores,
                      label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()

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
        self.ai_model = 'hand_lm'
        self.ai_model_running = False

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._ui.modelSelection.currentIndexChanged.connect(self.ai_model_changed)

        # start processing in a thread
        self.createCameraThread()

        # OSC client
        self.osc_address = self._ui.oscAddress.text()
        self.osc_port = self._ui.oscPort.text()
        self.osc_client = SimpleUDPClient(self.osc_address, int(self.osc_port))
        self._ui.oscAddress.editingFinished.connect(
            lambda: self.restart_osc_client(self._ui.oscAddress.text(), self._ui.oscPort.text()))
        self._ui.oscPort.editingFinished.connect(
            lambda: self.restart_osc_client(self._ui.oscAddress.text(), self._ui.oscPort.text()))

        # connect resize handlers
        # self._ui.camLabel.resizeEvent

        # connect buttons
        self._ui.startButton.clicked.connect(self.startCamera)
        self._ui.stopButton.clicked.connect(self.stopCamera)

    def ai_model_changed(self, index):
        """Combobox AI model selection changed."""
        match index:
            case 0:
                self.ai_model = 'hand_lm'
            case 1:
                self.ai_model = 'gesture'
            case 2:
                self.ai_model = 'pose'
            case 3:
                self.ai_model = 'face'
            case 4:
                self.ai_model = 'face_lm'
            case 5:
                self.ai_model = 'object'
            case _:
                return
        model_was_running = False
        if self.ai_model_running:
            model_was_running = True
        if model_was_running:
            self.stopCamera()
        self.createCameraThread()
        if model_was_running:
            self.startCamera()

    def createCameraThread(self):
        """Create new camera and ai analysis thread."""
        self.cam_thread = CameraThread(camera_index=self.cam_id,
                                       flip_horizontal=self.flip_horizontal,
                                       ai_model=self.ai_model)
        self.cam_thread.changePixmap.connect(self.setImage)
        self.cam_thread.finished.connect(self.clearCam)
        self.cam_thread.mediapipeResults.connect(self.sendOSC)

    @Slot(QImage)
    def setImage(self, image):
        # update image
        self._ui.camLabel.setPixmap(QPixmap.fromImage(image))

    @Slot(tuple)
    def sendOSC(self, data):
        # Send the landmarks through OSC
        print(data)
        ai_model, result = data
        if self.osc_client:
            if ai_model in ['hand_lm', 'gesture']:
                # hand landmark results
                if (hasattr(result, 'hand_landmarks') and hasattr(result, 'handedness') and
                        len(result.handedness) > 0):
                    hand = result.handedness[0][0].category_name
                    if self.flip_horizontal:
                        if hand == 'Left':
                            hand = 'Right'
                        else:
                            hand = 'Left'
                    for other_hand in ['Left', 'Right']:
                        if other_hand != hand:
                            self.osc_client.send_message(
                                f"{OSC_BASE_NAME}/hands/{other_hand.lower()}/tracking", False)
                    if len(result.hand_landmarks) > 0:
                        self.osc_client.send_message(
                            f"{OSC_BASE_NAME}/hands/{hand.lower()}/tracking", True)
                        for idx, landmark in enumerate(result.hand_landmarks[0]):
                            self.osc_client.send_message(
                                f"{OSC_BASE_NAME}/hands/{hand.lower()}/{str(idx)}/xyz",
                                [landmark.x, landmark.y, landmark.z])
                            self.osc_client.send_message(
                                f"{OSC_BASE_NAME}/hands/{hand.lower()}/{str(idx)}/visibility",
                                landmark.visibility)
                            self.osc_client.send_message(
                                f"{OSC_BASE_NAME}/hands/{hand.lower()}/{str(idx)}/presence",
                                landmark.presence)
                else:
                    for hand in ['Left', 'Right']:
                        self.osc_client.send_message(
                            f"{OSC_BASE_NAME}/hands/{hand.lower()}/tracking", False)
                if hasattr(result, 'gestures') and len(result.gestures) > 0:
                    for _idx, gesture in enumerate(result.gestures):
                        self.osc_client.send_message('{OSC_BASE_NAME}/gestures/index',
                                                     gesture[0].index)
                        self.osc_client.send_message('{OSC_BASE_NAME}/gestures/score',
                                                     gesture[0].score)
                        self.osc_client.send_message('{OSC_BASE_NAME}/gestures/displayname',
                                                     gesture[0].display_name)
                        self.osc_client.send_message('{OSC_BASE_NAME}/gestures/categoryname',
                                                     gesture[0].category_name)
            elif ai_model == 'pose':
                # TODO pose
                pass
            elif ai_model == 'face':
                # TODO face
                pass
            elif ai_model == 'face_lm':
                # TODO face landmarks
                pass
            elif ai_model == 'object':
                # send object recognition results
                if len(result.detections) > 0:
                    detection = result.detections[0]
                    bbox = detection.bounding_box
                    categories = detection.categories
                    # keypoints = detection.keypoints
                    self.osc_client.send_message('{OSC_BASE_NAME}/object/bbox/xy',
                                                 [bbox.origin_x, bbox.origin_y])
                    self.osc_client.send_message('{OSC_BASE_NAME}/object/bbox/wh',
                                                 [bbox.width, bbox.height])
                    if len(categories) > 1:
                        # TODO handle multiple categories for object
                        pass
                    elif len(categories) == 1:
                        category = categories[0]
                        self.osc_client.send_message('{OSC_BASE_NAME}/object/category/index',
                                                     category.index)
                        self.osc_client.send_message('{OSC_BASE_NAME}/object/category/score',
                                                     category.score)
                        self.osc_client.send_message('{OSC_BASE_NAME}/object/category/displayname',
                                                     category.display_name)
                        self.osc_client.send_message('{OSC_BASE_NAME}/object/category/categoryname',
                                                     category.category_name)

    def startCamera(self):
        self.cam_thread.start()
        self.ai_model_running = True

    def stopCamera(self):
        self.cam_thread.stop()
        self.cam_thread.quit()
        self.cam_thread.wait()
        self.ai_model_running = False

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
