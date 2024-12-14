import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to calculate the Euclidean distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

# Function to check correctness of specific body parts and calculate their distances
def analyze_body_parts(landmarks):
    status = []    # list to store correctness
    distances = []  # list to store distance between landmarks

    # Analyze shoulders    Fetch left and right shoulder landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
    shoulder_correct = abs(left_shoulder.y - right_shoulder.y) < 0.1   #checks shoulders are horizontally aligned
    status.append(("Shoulders", "Correct" if shoulder_correct else "Incorrect", shoulder_distance))

    # Analyze elbows   
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_arm_distance = calculate_distance(left_elbow, left_wrist)
    right_arm_distance = calculate_distance(right_elbow, right_wrist)
    left_arm_correct = left_elbow.y > left_shoulder.y
    right_arm_correct = right_elbow.y > right_shoulder.y

    status.append(("Left Arm", "Correct" if left_arm_correct else "Incorrect", left_arm_distance))
    status.append(("Right Arm", "Correct" if right_arm_correct else "Incorrect", right_arm_distance))

    # Analyze hips
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_distance = calculate_distance(left_hip, right_hip)
    hip_correct = abs(left_hip.y - right_hip.y) < 0.1
    status.append(("Hips", "Correct" if hip_correct else "Incorrect", hip_distance))

    return status

# Start capturing video
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Define the dimensions to upscale the video
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

while cap.isOpened():            #checks webcam is working or not
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the video frame
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))       # ensure resolutions 

    # Convert the frame to RGB (MediaPipe expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose
    results = pose.process(frame_rgb)

    # If pose landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Analyze body parts
        body_status = analyze_body_parts(landmarks)

        # Draw landmarks and skeleton on the frame
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

        # Display status and distances on the video
        y_offset = 50
        for part, correctness, distance in body_status:  # display each part correctness and live distance
            color = (0, 255, 0) if correctness == "Correct" else (0, 0, 255)
            cv2.putText(
                frame, 
                f"{part}: {correctness}, Distance: {distance:.2f}", 
                (50, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                color, 
                2, 
                cv2.LINE_AA,
            )
            y_offset += 30

    # Add title above the frame
    title_frame = cv2.copyMakeBorder(frame, 80, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(
        title_frame,
        "Posture Detection using Mediapipe & OpenCV",
        (200, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # Show the final output with title
    cv2.imshow('Posture Detection', title_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
