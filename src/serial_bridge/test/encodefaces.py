#!/usr/bin/env python3
"""
Face Registration using OAK-D Lite Camera (DepthAI v3 API) + InsightFace
- Streams RGB frames from the OAK-D Lite via DepthAI pipeline
- Detects faces and extracts embeddings using InsightFace (CPU)
- Saves averaged embedding to face_db/<name>.npy for later recognition
"""

import os
import cv2
import numpy as np
import depthai as dai
import insightface

# ----------------------------
# Setup Paths (Dynamic)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "face_data", "known_faces")
DB_DIR = os.path.join(BASE_DIR, "face_data", "face_db")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------
# InsightFace Setup
# ----------------------------
print("Loading InsightFace model...")
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU mode
print("InsightFace ready.\n")

# ----------------------------
# Ask for Name
# ----------------------------
name = input("Enter person's name: ").strip().lower()
if not name:
    raise ValueError("Name cannot be empty.")

person_dir = os.path.join(KNOWN_FACES_DIR, name)
os.makedirs(person_dir, exist_ok=True)
print(f"\nCapturing images for: {name}")
print("Look directly at the camera.")
print("Press 'q' to stop early.\n")

MAX_IMAGES = 15       # Number of good face samples to collect
FRAME_SKIP = 3        # Process every Nth frame to avoid redundant captures
PREVIEW_SIZE = (640, 360)

# ----------------------------
# DepthAI Pipeline (OAK-D Lite)
# ----------------------------
with dai.Pipeline() as pipeline:

    # RGB Camera on CAM_A (color sensor on OAK-D Lite)
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # Request a preview output at our desired resolution
    previewOut = camRgb.requestOutput(PREVIEW_SIZE, dai.ImgFrame.Type.BGR888p)

    # Create an output queue to receive frames on the host
    frameQueue = previewOut.createOutputQueue(maxSize=4, blocking=False)

    embeddings = []
    image_count = 0
    frame_counter = 0

    print("Pipeline started. Waiting for camera frames...")
    pipeline.start()

    while pipeline.isRunning():
        imgFrame = frameQueue.get()
        if imgFrame is None:
            continue

        frame = imgFrame.getCvFrame()  # numpy BGR array
        frame_counter += 1

        # Only run face detection every FRAME_SKIP frames
        if frame_counter % FRAME_SKIP != 0:
            cv2.imshow("OAK-D Lite - Register Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped early by user.")
                break
            continue

        # ----------------------------
        # Face Detection + Embedding
        # ----------------------------
        faces = app.get(frame)

        display_frame = frame.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Clamp bbox to frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Draw bounding box and count
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Captured: {image_count}/{MAX_IMAGES}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Save embedding
            embeddings.append(face.embedding)

            # Save cropped face image
            face_img = frame[y1:y2, x1:x2]
            img_path = os.path.join(person_dir, f"{image_count:03d}.jpg")
            cv2.imwrite(img_path, face_img)
            image_count += 1
            print(f"  [✓] Captured image {image_count}/{MAX_IMAGES}")

            # Only capture one face per frame
            break

        # HUD overlay
        cv2.putText(
            display_frame,
            f"Person: {name}  |  Samples: {image_count}/{MAX_IMAGES}",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow("OAK-D Lite - Register Face", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early by user.")
            break

        if image_count >= MAX_IMAGES:
            print(f"\nReached target of {MAX_IMAGES} samples.")
            break

cv2.destroyAllWindows()

# ----------------------------
# Save Averaged Embedding
# ----------------------------
if len(embeddings) > 0:
    avg_embedding = np.mean(embeddings, axis=0)
    db_file = os.path.join(DB_DIR, f"{name}.npy")
    np.save(db_file, avg_embedding)
    print(f"\n[✓] Saved embedding for '{name}' → {db_file}")
    print(f"    Total samples used: {len(embeddings)}")
else:
    print("\n[✗] No face detected. Please try again with better lighting.")