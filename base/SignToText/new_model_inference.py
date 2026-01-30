# ================================================================
# MULTI-CATEGORY REAL-TIME SIGN LANGUAGE TRANSLATION (DJANGO READY)
# ================================================================

import os
import cv2
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import mediapipe as mp
import pickle

# ================================================================
# CONFIG
# ================================================================
FRAME_COUNT = 10
RECORD_DURATION = 6  # seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "new_models")


# ================================================================
# CATEGORY → MODEL REGISTRY
# ================================================================
CATEGORY_MODELS = {
    "general_conversion": {
        "model_path": "sign_transformer_best_gen_model.pt",
        "label_encoder": "sign_label_best_gen_encoder.pkl",
    },
    "patient_story": {
        "model_path": "sign_transformer_best_patstory04_model.pt",
        "label_encoder": "sign_label_best_patstory04_encoder.pkl",
    },
    "full": {  
        "model_path": "sign_transformer_best_full04_model.pt",
        "label_encoder": "sign_label_best_full04_encoder.pkl",
    },
}

# ================================================================
# MODEL CACHE (IMPORTANT FOR DJANGO)
# ================================================================
_MODEL_CACHE = {}

# ================================================================
# MediaPipe setup
# ================================================================
mp_holistic = mp.solutions.holistic

# ================================================================
# TRANSFORMER CLASSIFIER (UNCHANGED LOGIC)
# ================================================================
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        dropout=0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        enc = self.encoder(x)
        pooled = enc.mean(dim=1)
        return self.fc(self.dropout(pooled))

# ================================================================
# CATEGORY-AWARE MODEL LOADER (CACHED)
# ================================================================
def load_model_for_category(category: str):
    if category not in CATEGORY_MODELS:
        raise ValueError(f"Unknown category: {category}")

    if category in _MODEL_CACHE:
        return _MODEL_CACHE[category]

    cfg = CATEGORY_MODELS[category]

    model_path = os.path.join(MODELS_DIR, cfg["model_path"])
    encoder_path = os.path.join(MODELS_DIR, cfg["label_encoder"])

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = TransformerClassifier(
        input_dim=checkpoint["input_dim"],
        model_dim=checkpoint["model_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=3,
        ff_dim=1024,
        num_classes=len(label_encoder.classes_),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "frame_count": checkpoint["frame_count"],
    }

    _MODEL_CACHE[category] = bundle
    return bundle


# ================================================================
# FEATURE EXTRACTION (UNCHANGED LOGIC)
# ================================================================
def extract_pose_keypoints(img_pil):
    """Return pose + hands flattened vector (len = 258)."""
    if img_pil is None:
        return np.zeros(258, dtype=np.float32)

    img_rgb = np.array(img_pil)

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as h:
        results = h.process(img_rgb)

    pose = np.zeros(33 * 4)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
             for lm in results.pose_landmarks.landmark]
        ).flatten()

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark]
        ).flatten()

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark]
        ).flatten()

    return np.concatenate([pose, lh, rh]).astype(np.float32)

# ================================================================
# FRAME SAMPLING
# ================================================================
def sample_frames(frame_list, count=FRAME_COUNT):
    if len(frame_list) == 0:
        return [None] * count

    idxs = np.linspace(0, len(frame_list) - 1, count, dtype=int)
    sampled = []

    for i in idxs:
        img = frame_list[i]
        sampled.append(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if img is not None else None
        )

    return sampled

# ================================================================
# FRAMES → TENSOR
# ================================================================
def frames_to_tensor(frames):
    pose_vecs = [extract_pose_keypoints(f) for f in frames]
    pose_seq = np.stack(pose_vecs)             # (10, 258)
    pose_seq = torch.tensor(
        pose_seq, dtype=torch.float32
    ).unsqueeze(0)                              # (1, 10, 258)

    return pose_seq.to(DEVICE)

# ================================================================
# PREDICTION (CATEGORY AWARE)
# ================================================================
def predict_from_frames(frames, category: str):
    """
    frames   : list of OpenCV BGR frames
    category : one of CATEGORY_MODELS keys
    """
    bundle = load_model_for_category(category)

    sampled = sample_frames(frames, bundle["frame_count"])
    seq = frames_to_tensor(sampled)

    with torch.no_grad():
        logits = bundle["model"](seq)
        pred_idx = logits.argmax(dim=1).item()

    return bundle["label_encoder"].inverse_transform([pred_idx])[0]

# ================================================================
# VIDEO FILE → PREDICTION (DJANGO UPLOADS)
# ================================================================
def predict_from_video(video_path: str, category: str):
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

    return predict_from_frames(frames, category)

# ================================================================
# OPTIONAL: WEBCAM DEMO (LOCAL DEBUG ONLY)
# ================================================================
def run_webcam_demo(category="general"):
    cap = cv2.VideoCapture(0)
    predicted_text = ""

    print("[READY] Press 'R' to record | 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Prediction: {predicted_text}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Sign Translation", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        if key == ord("r"):
            frames = []
            start = time.time()

            while time.time() - start < RECORD_DURATION:
                ret2, f2 = cap.read()
                if ret2:
                    frames.append(f2)

                cv2.putText(f2, "Recording...",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                cv2.imshow("Sign Translation", f2)
                cv2.waitKey(1)

            predicted_text = predict_from_frames(frames, category)
            print("[✓] Prediction:", predicted_text)

    cap.release()
    cv2.destroyAllWindows()
