import cv2
import mediapipe as mp
import numpy as np
import time


def calculate_angle3(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_angle2(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Squat counter variables
counter = 0
stage = None
issue = None

flat_shoulders_issues = 0
shoulders_message_printed = False

foot_wide_issues = 0
wide_message_printed = False
foot_narrow_issues = 0
narrow_message_printed = False

knee_outward_issues = 0
knee_outward_message_printed = False
knee_inward_issues = 0
knee_inward_message_printed = False

depth_counter = 0
depth_message_printed = False

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        cv2.namedWindow('Squat Form Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Squat Form Analysis', 1280, 920)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            shoulder_angle = calculate_angle2(left_shoulder, right_shoulder)
            depth_left_side = calculate_angle2(left_knee, left_hip)
            depth_right_side = calculate_angle2(right_knee, right_hip)
            foot_positioning_left_side = calculate_angle2(left_ankle, left_shoulder)
            foot_positioning_right_side = calculate_angle2(right_ankle, right_shoulder)
            knees_over_toes_right_side = calculate_angle2(right_ankle, right_knee)
            knees_over_toes_left_side = calculate_angle2(left_ankle, left_knee)

            issues = []
            # Check all angles for proper form and store any issues in a dictionary
            if depth_right_side > 190 and depth_left_side > 190:
                issues.append("good depth!")
                depth_counter += 1
            if shoulder_angle > 190 or shoulder_angle < 170:
                issues.append("make sure your shoulders are flat")
                flat_shoulders_issues += 1
            if foot_positioning_left_side > 90 and foot_positioning_right_side > 90:
                issues.append("your stance is too wide")
                foot_wide_issues += 1
            if foot_positioning_left_side < 80 and foot_positioning_right_side < 80:
                issues.append("your stance is too narrow")
                foot_narrow_issues += 1
            if knees_over_toes_right_side < 75 and knees_over_toes_left_side < 75:
                issues.append("push your knees outwards to align them with your feet")
                knee_outward_issues += 1
            if knees_over_toes_right_side > 115 and knees_over_toes_left_side > 115:
                issues.append("pull your knees inwards to align them with your feet")
                knee_inward_issues += 1
            # If there are issues, return them. Otherwise, return a "correct form" message.
            if issues:
                issue = " and ".join(issues)
            else:
                issue = "no issues with your form"

        except:
            pass

        # Render rep counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, str("test"),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Issue data
        cv2.putText(image, 'Issues', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, issue,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Squat Form Analysis', image)

        # if a detection is made in multiple frames, provide feedback for user to see afterwards
        # (multiple frames so that false positives are prevented)
        if depth_counter > 10 and not depth_message_printed:
            print("Your squat reached good depth!")
            shoulders_message_printed = True

        if flat_shoulders_issues > 30 and not shoulders_message_printed:
            print("focus on making sure your shoulders are flat")
            shoulders_message_printed = True

        if foot_wide_issues > 30 and not wide_message_printed:
            print("focus on ensuring your feet are shoulder width apart - at times they were too far from eachother")
            wide_message_printed = True

        if foot_narrow_issues > 30 and not narrow_message_printed:
            print("focus on ensuring your feet are shoulder width apart - at times they were too close to eachother")
            narrow_message_printed = True

        if knee_outward_issues > 30 and not knee_outward_message_printed:
            print("focus on pushing your knees outwards to align them with your feet")
            knee_outward_message_printed = True

        if knee_inward_issues > 30 and not knee_inward_message_printed:
            print("focus on pulling your knees inwards to align them with your feet")
            knee_inward_message_printed = True

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
