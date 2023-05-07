import cv2
import mediapipe as mp
import numpy as np


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
cv2.namedWindow('Squat Form Analysis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Squat Form Analysis', 1280, 720)

# Delay for 10 seconds before starting
for i in range(10, 0, -1):

    ret, frame = cap.read()
    # Display countdown timer on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(i), (50, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Squat Form Analysis', frame)
    cv2.waitKey(1000)

# Squat counter variables
counter = 0
stage = None
issue = None

ending_pose_frames = 0

flat_shoulders_issues = 0
shoulders_message_printed = False

foot_positioning_issues = 0
foot_message_printed = False

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
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
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
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            # Calculate angles
            shoulder_angle = calculate_angle2(left_shoulder, right_shoulder)
            depth_left_side = calculate_angle2(left_knee, left_hip)
            depth_right_side = calculate_angle2(right_knee, right_hip)
            foot_positioning_left_side = calculate_angle2(left_ankle, left_shoulder)
            foot_positioning_right_side = calculate_angle2(right_ankle, right_shoulder)
            knees_over_toes_right_side = calculate_angle2(right_ankle, right_knee)
            knees_over_toes_left_side = calculate_angle2(left_ankle, left_knee)
            heels_left = calculate_angle2(left_foot, left_heel)
            heels_right = calculate_angle2(right_foot, right_heel)
            # ending pose angle (left arm straight up in the air)
            ending_pose = calculate_angle2(left_shoulder, left_elbow)

            issues = []
            # Check all angles for proper form and store any issues in a dictionary
            if depth_right_side < 30 or depth_left_side < 30:
                issues.append("good depth!")
                depth_counter += 1
            if shoulder_angle > 190 or shoulder_angle < 170:
                issues.append("make sure your shoulders are flat")
                flat_shoulders_issues += 1
            if foot_positioning_left_side > 95.5 or foot_positioning_left_side < 85 or foot_positioning_right_side > 95.5 or foot_positioning_right_side < 85:
                issues.append("make sure your feet are shoulder width apart")
                foot_positioning_issues += 1
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

            # ending pose
            #if 0 < ending_pose < 30:
            #    ending_pose_frames += 1

        except:
            pass

        # Render rep counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        rounded1 = round(knees_over_toes_left_side, 2)
        rounded2 = round(knees_over_toes_right_side, 2)
        cv2.putText(image, str(0),
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
                                  mp_drawing.DrawingSpec(color=(52, 192, 235), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(235, 70, 235), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Squat Form Analysis', image)

        # if a detection is made in more than the specified frames, provide feedback for user to see afterwards
        # (multiple frames so that false positives are prevented)
        if depth_counter > 10 and not depth_message_printed:
            print("Your squat reached good depth!")
            depth_message_printed = True

        if flat_shoulders_issues > 60 and not shoulders_message_printed:
            print("focus on making sure your shoulders are flat")
            shoulders_message_printed = True

        if foot_positioning_issues > 60 and not foot_message_printed:
            print("focus on ensuring your feet are shoulder width apart")
            foot_message_printed = True

        if knee_outward_issues > 60 and not knee_outward_message_printed:
            print("focus on pushing your knees outwards to align them with your feet")
            knee_outward_message_printed = True

        if knee_inward_issues > 60 and not knee_inward_message_printed:
            print("focus on pulling your knees inwards to align them with your feet")
            knee_inward_message_printed = True

        # if user holds ending pose for more than 60 frames end application
        if ending_pose_frames > 60:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
