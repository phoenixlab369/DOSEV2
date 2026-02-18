#!/usr/bin/env python3
"""
Real-time Face Recognition using OAK-D Lite + InsightFace

Features:
- Loads saved embeddings from face_data/face_db
- Detects faces in real-time
- Matches faces using cosine similarity
- Draws bounding box with person's name
"""

import os
import cv2
import numpy as np
import depthai as dai
import insightface

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "face_data", "face_db")

# ----------------------------
# Recognition Settings
# ----------------------------
SIMILARITY_THRESHOLD = 0.5   # Lower = stricter, Higher = easier match
PREVIEW_SIZE = (640, 360)

# ----------------------------
# Load InsightFace
# ----------------------------
print("Loading InsightFace...")
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1)   # CPU mode
print("InsightFace ready")

# ----------------------------
# Load Face Database
# ----------------------------
print("\nLoading face database...")

known_embeddings = []
known_names = []

for file in os.listdir(DB_DIR):
    if file.endswith(".npy"):
        name = os.path.splitext(file)[0]
        embedding = np.load(os.path.join(DB_DIR, file))

        known_embeddings.append(embedding)
        known_names.append(name)

print(f"Loaded {len(known_names)} faces:", known_names)

# ----------------------------
# Cosine Similarity Function
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ----------------------------
# Find Best Match
# ----------------------------
def recognize_face(face_embedding):

    best_score = -1
    best_name = "Unknown"

    for i, known_embedding in enumerate(known_embeddings):

        score = cosine_similarity(face_embedding, known_embedding)

        if score > best_score:
            best_score = score
            best_name = known_names[i]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    else:
        return "Unknown", best_score


# ----------------------------
# DepthAI Pipeline
# ----------------------------
with dai.Pipeline() as pipeline:

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    previewOut = camRgb.requestOutput(PREVIEW_SIZE, dai.ImgFrame.Type.BGR888p)

    frameQueue = previewOut.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()

    print("\nStarting recognition... Press 'q' to exit")

    while pipeline.isRunning():

        imgFrame = frameQueue.get()

        if imgFrame is None:
            continue

        frame = imgFrame.getCvFrame()

        display_frame = frame.copy()

        # Detect faces
        faces = app.get(frame)

        for face in faces:

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            embedding = face.embedding

            # Recognize
            name, score = recognize_face(embedding)

            # Choose color
            if name == "Unknown":
                color = (0, 0, 255)   # Red
            else:
                color = (0, 255, 0)   # Green

            # Draw bounding box
            cv2.rectangle(display_frame,
                          (x1, y1),
                          (x2, y2),
                          color,
                          2)

            # Draw name
            label = f"{name} ({score:.2f})"

            cv2.putText(display_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        cv2.imshow("Face Recognition - OAK-D Lite", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
