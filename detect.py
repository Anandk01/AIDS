"""
Human Body Tracking Bot
Author: OpenAI GPT-4
Description: Detects a human using MediaPipe Pose, estimates torso center,
             and moves servos to keep the person centered.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import time


class ServoController:
    def __init__(self, port="COM7", pin_x=9, pin_y=10):
        self.board = pyfirmata.Arduino(port)
        self.servo_x = self.board.get_pin(f'd:{pin_x}:s')
        self.servo_y = self.board.get_pin(f'd:{pin_y}:s')
        self.position = [90, 90]  # initial center
        self.write_position()

    def update_position(self, x_deg, y_deg):
        self.position[0] = np.clip(x_deg, 0, 180)
        self.position[1] = np.clip(y_deg, 0, 180)
        self.write_position()

    def write_position(self):
        self.servo_x.write(self.position[0])
        self.servo_y.write(self.position[1])


class PoseTracker:
    def __init__(self, width=1280, height=720):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, width)
        self.cap.set(4, height)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible.")
        self.width = width
        self.height = height

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.drawer = mp.solutions.drawing_utils

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)

        torso_center = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # Get average position of left & right shoulder
            l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            cx = int((l_shoulder.x + r_shoulder.x) / 2 * self.width)
            cy = int((l_shoulder.y + r_shoulder.y) / 2 * self.height)
            torso_center = (cx, cy)

            self.drawer.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame, torso_center

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def map_position_to_servo(cx, cy, width, height):
    servo_x = np.interp(cx, [0, width], [0, 180])
    servo_y = np.interp(cy, [0, height], [0, 180])
    return servo_x, servo_y


def main():
    WIDTH, HEIGHT = 1280, 720
    pose_tracker = PoseTracker(WIDTH, HEIGHT)
    servo_controller = ServoController()

    try:
        while True:
            frame, center = pose_tracker.read_frame()
            if frame is None:
                break

            if center:
                cx, cy = center
                cv2.circle(frame, center, 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"Tracking ({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                servo_x, servo_y = map_position_to_servo(cx, cy, WIDTH, HEIGHT)
                servo_controller.update_position(servo_x, servo_y)

            else:
                cv2.putText(frame, "No Human Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Body Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pose_tracker.release()


if __name__ == "__main__":
    main()
"""
Human Body Tracking Bot
Author: OpenAI GPT-4
Description: Detects a human using MediaPipe Pose, estimates torso center,
             and moves servos to keep the person centered.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import time


class ServoController:
    def __init__(self, port="COM7", pin_x=9, pin_y=10):
        self.board = pyfirmata.Arduino(port)
        self.servo_x = self.board.get_pin(f'd:{pin_x}:s')
        self.servo_y = self.board.get_pin(f'd:{pin_y}:s')
        self.position = [90, 90]  # initial center
        self.write_position()

    def update_position(self, x_deg, y_deg):
        self.position[0] = np.clip(x_deg, 0, 180)
        self.position[1] = np.clip(y_deg, 0, 180)
        self.write_position()

    def write_position(self):
        self.servo_x.write(self.position[0])
        self.servo_y.write(self.position[1])


class PoseTracker:
    def __init__(self, width=1280, height=720):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, width)
        self.cap.set(4, height)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible.")
        self.width = width
        self.height = height

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.drawer = mp.solutions.drawing_utils

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)

        torso_center = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # Get average position of left & right shoulder
            l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            cx = int((l_shoulder.x + r_shoulder.x) / 2 * self.width)
            cy = int((l_shoulder.y + r_shoulder.y) / 2 * self.height)
            torso_center = (cx, cy)

            self.drawer.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame, torso_center

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def map_position_to_servo(cx, cy, width, height):
    servo_x = np.interp(cx, [0, width], [0, 180])
    servo_y = np.interp(cy, [0, height], [0, 180])
    return servo_x, servo_y


def main():
    WIDTH, HEIGHT = 1280, 720
    pose_tracker = PoseTracker(WIDTH, HEIGHT)
    servo_controller = ServoController()

    try:
        while True:
            frame, center = pose_tracker.read_frame()
            if frame is None:
                break

            if center:
                cx, cy = center
                cv2.circle(frame, center, 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"Tracking ({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                servo_x, servo_y = map_position_to_servo(cx, cy, WIDTH, HEIGHT)
                servo_controller.update_position(servo_x, servo_y)

            else:
                cv2.putText(frame, "No Human Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Body Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pose_tracker.release()


if __name__ == "__main__":
    main()

