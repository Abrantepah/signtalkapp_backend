# general_conversion.py
import cv2
import numpy as np
import pickle
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
import os

FRAME_COUNT = 10
POSE_COUNT = 33
HAND_COUNT = 21

# MediaPipe holistic setup (load once)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Base folder where category-specific models live
BASE_PATH = os.path.join(os.path.dirname(__file__), "xgboost")

# Map each category to its model and encoder filenames
CATEGORY_MODELS = {
    "laboratory":      ("xgboost_lab_model.pkl", "label_encoder_lab.pkl"),
    "child_welfare":   ("xgboost_cw_model.pkl", "label_encoder_cw.pkl"),
    "radiology":       ("xgboost_rad_model.pkl", "label_encoder_rad.pkl"),
    "pharmacy":        ("xgboost_pharm_model.pkl", "label_encoder_pharm.pkl"),
    "maternity":       ("xgboost_mat_model.pkl", "label_encoder_mat.pkl"),
    "opd":             ("xgboost_opd_model.pkl", "label_encoder_opd.pkl"),
}

# -----------------------
# Keypoint extraction
# -----------------------
def extract_keypoints(results):
    if results is None:
        return np.zeros(POSE_COUNT*4 + 2*HAND_COUNT*3, dtype=np.float32)

    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(POSE_COUNT*4, dtype=np.float32)

    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(HAND_COUNT*3, dtype=np.float32)

    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(HAND_COUNT*3, dtype=np.float32)

    return np.concatenate([pose, lh, rh]).astype(np.float32)

# -----------------------
# Convert frames to vector
# -----------------------
def process_frames_to_vector(frames):
    seq = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        seq.append(extract_keypoints(results))

    if len(seq) > FRAME_COUNT:
        indices = np.linspace(0, len(seq)-1, FRAME_COUNT, dtype=int)
        seq = [seq[i] for i in indices]
    while len(seq) < FRAME_COUNT:
        seq.append(np.zeros_like(seq[0]))

    full_vec = np.concatenate(seq)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(full_vec.reshape(-1, 1)).flatten()
    return normalized.reshape(1, -1)

# -----------------------
# Main pipeline
# -----------------------
def predict_translation_from_video(video_path, category: str):
    """
    Takes a video file and category, loads the appropriate model & label encoder,
    and returns the predicted sign label.
    """
    if category not in CATEGORY_MODELS:
        raise ValueError(f"Unknown category: {category}")

    model_file, encoder_file = CATEGORY_MODELS[category]

    # Load model & encoder
    model_path = os.path.join(BASE_PATH, model_file)
    encoder_path = os.path.join(BASE_PATH, encoder_file)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video.")

    # Convert frames to model-ready vector
    X = process_frames_to_vector(frames)

    # Predict
    pred_idx = model.predict(X)
    try:
        predicted_label = label_encoder.inverse_transform(pred_idx)[0]
    except Exception:
        predicted_label = str(int(pred_idx[0]))

    return predicted_label
