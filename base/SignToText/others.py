import os
import cv2
import numpy as np
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CONFIG
# ============================================================
FRAME_COUNT = 10
POSE_COUNT = 33
HAND_COUNT = 21
EXPECTED_FEATURE_SIZE = 2582  # keypoints + CNN features

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)
XGB_DIR = os.path.join(BASE_DIR, "xgboost")

# ============================================================
# MediaPipe Holistic
# ============================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# CNN Models
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet50
resnet = models.resnet50(weights=None)
resnet.fc = nn.Identity()
resnet_path = os.path.join(MODEL_DIR, "resnet50.pth")
resnet.load_state_dict(torch.load(resnet_path, map_location=device))
resnet.to(device).eval()

# Load EfficientNetB0
efficientnet = models.efficientnet_b0(weights=None)
efficientnet.classifier = nn.Identity()
efficientnet_path = os.path.join(MODEL_DIR, "efficientnet_b0.pth")
efficientnet.load_state_dict(torch.load(efficientnet_path, map_location=device))
efficientnet.to(device).eval()

# ============================================================
# CATEGORY → MODEL & ENCODER MAPPER
# ============================================================
CATEGORY_MODELS = {
    "laboratory":    ("xgboost_lab_model.pkl", "label_encoder_lab.pkl"),
    "child_welfare": ("xgboost_cw_model.pkl", "label_encoder_cw.pkl"),
    "radiology":     ("xgboost_rad_model.pkl", "label_encoder_rad.pkl"),
    "pharmacy":      ("xgboost_pharm_model.pkl", "label_encoder_pharm.pkl"),
    "maternity":     ("xgboost_mat_model.pkl", "label_encoder_mat.pkl"),
    "opd":           ("xgboost_opd_model.pkl", "label_encoder_opd.pkl"),
}

# ============================================================
# KEYPOINT EXTRACTION
# ============================================================
def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(POSE_COUNT * 4, dtype=np.float32)

    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(HAND_COUNT * 3, dtype=np.float32)

    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(HAND_COUNT * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh]).astype(np.float32)

# ============================================================
# CNN FEATURE EXTRACTION
# ============================================================
def extract_cnn_features(frame_files):
    imgs = []
    for f in frame_files:
        try:
            img = Image.open(f).convert("RGB")
            imgs.append(transform(img))
        except Exception as e:
            print(f"[Warning: Error loading {f}: {e}]")

    if len(imgs) == 0:
        return np.zeros(1), np.zeros(1)

    imgs = torch.stack(imgs).to(device)

    with torch.no_grad():
        res_feats = resnet(imgs).mean(dim=0).cpu().numpy()
        eff_feats = efficientnet(imgs).mean(dim=0).cpu().numpy()

    return np.array([res_feats.mean()]), np.array([eff_feats.mean()])

# ============================================================
# FRAMES → FEATURE VECTOR
# ============================================================
def process_frames_to_vector(frames):
    temp_dir = os.path.join(BASE_DIR, "temp_cnn_frames")
    os.makedirs(temp_dir, exist_ok=True)

    keypoints_seq = []
    frame_files = []

    indexes = np.linspace(0, len(frames) - 1, FRAME_COUNT, dtype=int)

    for i, idx in enumerate(indexes):
        frame = frames[idx]

        # Save CNN frame
        save_path = os.path.join(temp_dir, f"f{i}.png")
        cv2.imwrite(save_path, frame)
        frame_files.append(save_path)

        # Extract MediaPipe landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints_seq.append(extract_keypoints(results))

    # Normalize keypoints
    kp_vec = np.concatenate(keypoints_seq)
    scaler = MinMaxScaler()
    kp_norm = scaler.fit_transform(kp_vec.reshape(-1, 1)).flatten()

    # CNN features
    res_feat, eff_feat = extract_cnn_features(frame_files)

    # Combine: keypoints + CNN features → 2582
    final_vec = np.concatenate([kp_norm, res_feat, eff_feat]).reshape(1, -1)
    return final_vec

# ============================================================
# PREDICTION
# ============================================================
def predict_translation_from_video(video_path, category: str):
    if category not in CATEGORY_MODELS:
        raise ValueError(f"Unknown category: {category}")

    model_file, encoder_file = CATEGORY_MODELS[category]
    model_path = os.path.join(XGB_DIR, model_file)
    encoder_path = os.path.join(XGB_DIR, encoder_file)

    # Load model & encoder
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return "[Error: No frames extracted from video]"

    # Extract full feature vector
    X = process_frames_to_vector(frames)

    # Check shape
    if X.shape[1] != EXPECTED_FEATURE_SIZE:
        return f"[Shape mismatch: Expected {EXPECTED_FEATURE_SIZE}, got {X.shape[1]}]"

    # Predict label
    pred_idx = model.predict(X)
    label = label_encoder.inverse_transform(pred_idx)[0]
    return label
