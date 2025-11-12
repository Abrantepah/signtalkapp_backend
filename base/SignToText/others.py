import cv2
import os
import numpy as np
import pickle
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
FRAME_COUNT = 10
POSE_COUNT = 33
HAND_COUNT = 21
EXPECTED_FEATURE_SIZE = 2582

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MediaPipe Holistic (load once)
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
# CNN Models (load once)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

resnet = models.resnet50(weights=None)
resnet.load_state_dict(torch.load(r"C:\Users\Administrator\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth"))
resnet.fc = nn.Identity()
resnet.to(device).eval()

efficientnet = models.efficientnet_b0(weights=None)
efficientnet.load_state_dict(torch.load(r"C:\Users\Administrator\.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-7f5810bc.pth"))
efficientnet.classifier = nn.Identity()
efficientnet.to(device).eval()


# ============================================================
# CATEGORY → MODEL & ENCODER MAPPER
# ============================================================
BASE_PATH = os.path.join(os.path.dirname(__file__), "xgboost")

CATEGORY_MODELS = {
    "laboratory":      ("xgboost_lab_model.pkl", "label_encoder_lab.pkl"),
    "child_welfare":   ("xgboost_cw_model.pkl", "label_encoder_cw.pkl"),
    "radiology":       ("xgboost_rad_model.pkl", "label_encoder_rad.pkl"),
    "pharmacy":        ("xgboost_pharm_model.pkl", "label_encoder_pharm.pkl"),
    "maternity":       ("xgboost_mat_model.pkl", "label_encoder_mat.pkl"),
    "opd":             ("xgboost_opd_model.pkl", "label_encoder_opd.pkl"),
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
        except:
            pass

    if len(imgs) == 0:
        return np.zeros(1), np.zeros(1)

    imgs = torch.stack(imgs).to(device)

    with torch.no_grad():
        res_feats = resnet(imgs).mean(dim=0).cpu().numpy()
        eff_feats = efficientnet(imgs).mean(dim=0).cpu().numpy()

    # Reduce dimensionality like training
    return np.array([res_feats.mean()]), np.array([eff_feats.mean()])

# ============================================================
# FRAMES → FEATURE VECTOR
# ============================================================
def process_frames_to_vector(frames):
    temp_dir = "temp_cnn_frames"
    os.makedirs(temp_dir, exist_ok=True)

    keypoints_seq = []
    frame_files = []

    indexes = np.linspace(0, len(frames)-1, FRAME_COUNT, dtype=int)

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

    # CNN deep features
    res_feats, eff_feats = extract_cnn_features(frame_files)

    final_vec = np.concatenate([kp_norm, res_feats, eff_feats]).reshape(1, -1)

    return final_vec

# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================
def predict_translation_from_video(video_path, category: str):
    if category not in CATEGORY_MODELS:
        raise ValueError(f"Unknown category: {category}")

    model_file, encoder_file = CATEGORY_MODELS[category]

    # Load model + encoder
    with open(os.path.join(BASE_PATH, model_file), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_PATH, encoder_file), "rb") as f:
        label_encoder = pickle.load(f)

    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return "[Error: No frames extracted]"

    # Convert frames → feature vector
    X = process_frames_to_vector(frames)

    # Shape validation
    if X.shape[1] != EXPECTED_FEATURE_SIZE:
        return f"[Shape mismatch: Expected {EXPECTED_FEATURE_SIZE}, got {X.shape[1]}]"

    pred_idx = model.predict(X)
    label = label_encoder.inverse_transform(pred_idx)[0]

    return label
