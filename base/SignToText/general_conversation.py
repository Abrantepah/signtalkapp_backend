import cv2
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import mediapipe as mp


# Path to the xgboost folder (relative to this file)
base_path = os.path.join(os.path.dirname(__file__), "xgboost")

# Load label encoder
label_encoder_path = os.path.join(base_path, "label_encoder_g_2.pkl")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Load model
model_path = os.path.join(base_path, "xgboost_model2.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# === Constants ===
POSE_COUNT, HAND_COUNT = 33, 21
FRAME_COUNT = 10  # number of frames expected by the model

# === MediaPipe Setup (Pose + Hands only) ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Extract Pose + Hands Only ===
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_COUNT * 4)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(HAND_COUNT * 3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(HAND_COUNT * 3)
    return np.concatenate([pose, lh, rh])

# === Process a Sequence of Frames ===
def process_sequence(frames):
    sequence = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    # Sample or pad to match FRAME_COUNT
    if len(sequence) > FRAME_COUNT:
        indices = np.linspace(0, len(sequence)-1, FRAME_COUNT, dtype=int)
        sequence = [sequence[i] for i in indices]
    while len(sequence) < FRAME_COUNT:
        sequence.append(np.zeros_like(sequence[0]))

    full_vector = np.concatenate(sequence)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(full_vector.reshape(-1, 1)).flatten()

    return normalized.reshape(1, -1)

# === New Function: Predict from Video File ===
def predict_translation_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video.")

    # Process frames and predict
    vector = process_sequence(frames)
    pred = model.predict(vector)
    predicted_label = label_encoder.inverse_transform(pred)[0]

    return predicted_label
