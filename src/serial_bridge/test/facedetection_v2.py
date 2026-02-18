#!/usr/bin/env python3
"""
OAK-D Lite (DepthAI v3) + YOLOv6-nano SpatialDetectionNetwork + ObjectTracker (on-device)
Host: InsightFace recognition runs in a BACKGROUND THREAD (NO UI FREEZE)

What it does:
- Uses Luxonis "object_tracker" example-style pipeline (SpatialDetectionNetwork -> ObjectTracker)
- Displays bbox + tracking ID + status + X/Y/Z (mm)
- Runs face recognition ONCE per tracking ID (and optionally retries if "Unknown")
- Maintains cache map: tracking_id -> {"name","score","last_seen","pending"...}
- Clears cache when tracking is LOST/REMOVED or times out

Folder structure (same as your earlier registration code):
<project_root>/
  face_data/
    face_db/
      saravanan.npy
      tejesh.npy
  scripts/
    this_file.py

Notes:
- Person detector gives PERSON bbox, not FACE bbox. We do face detection inside the UPPER BODY crop.
- If you see "???" it means recognition not ready yet OR no face detected in crop.
"""

import os
import time
import cv2
import numpy as np
import depthai as dai
import insightface
import threading
import queue


# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "face_data", "face_db")


# ----------------------------
# TUNING
# ----------------------------
SIM_THRESHOLD = 0.50            # cosine similarity threshold (try 0.45 if too strict)
TRACK_TIMEOUT_S = 2.0           # remove cache entry if not seen for this many seconds
RECOG_RETRY_S = 1.5             # if Unknown, allow re-try after this many seconds
MAX_Z_MM = 4000                 # only run recognition if within this Z distance (mm). allow Z=0 too.
MIN_CROP = 80                   # min crop size for recognition
CROP_EXPAND = 0.20              # expand person bbox to include face better
UPPER_BODY_RATIO = 0.55         # use top 55% of person bbox for face detection

MAX_RECOG_QUEUE = 8             # recognition task queue size

FULL_FRAME_TRACKING = False     # keep False (faster, like example default)


# ----------------------------
# HELPERS
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_bbox(x1, y1, x2, y2, w, h, expand=0.2):
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * expand)
    dy = int(bh * expand)
    nx1 = clamp(x1 - dx, 0, w - 1)
    ny1 = clamp(y1 - dy, 0, h - 1)
    nx2 = clamp(x2 + dx, 0, w - 1)
    ny2 = clamp(y2 + dy, 0, h - 1)
    return nx1, ny1, nx2, ny2


def upper_body_crop(frame, x1, y1, x2, y2, w, h):
    """
    Take expanded person bbox, then keep only upper portion (where face likely is).
    This is the #1 reason your previous code showed ??? (face wasn't in crop).
    """
    ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, w, h, expand=CROP_EXPAND)
    bh = ey2 - ey1
    top_h = int(bh * UPPER_BODY_RATIO)
    cy2 = clamp(ey1 + top_h, ey1 + 2, h - 1)
    crop = frame[ey1:cy2, ex1:ex2]
    return crop


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return -1.0
    return float(np.dot(a, b) / denom)


def load_face_db(db_dir: str):
    known_names, known_embs = [], []
    if not os.path.isdir(db_dir):
        print(f"[WARN] Face DB folder not found: {db_dir}")
        return known_names, known_embs

    for f in sorted(os.listdir(db_dir)):
        if f.endswith(".npy"):
            name = os.path.splitext(f)[0]
            emb = np.load(os.path.join(db_dir, f)).astype(np.float32)
            known_names.append(name)
            known_embs.append(emb)

    return known_names, known_embs


def recognize_embedding(emb: np.ndarray, known_names, known_embs):
    best_name = "Unknown"
    best_score = -1.0
    for name, kemb in zip(known_names, known_embs):
        s = cosine_similarity(emb, kemb)
        if s > best_score:
            best_score = s
            best_name = name
    if best_score >= SIM_THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score


# ----------------------------
# INSIGHTFACE
# ----------------------------
print("Loading InsightFace (CPU)...")
app = insightface.app.FaceAnalysis()
# Make face detection reliable (can be (512,512) if you want faster)
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace ready.")

known_names, known_embs = load_face_db(DB_DIR)
print(f"Loaded face DB entries: {known_names if known_names else 'None'}")
if not known_names:
    print(f"[WARN] No .npy embeddings found in {DB_DIR}. Recognition will stay as '???'.")


# ----------------------------
# CACHE: tracking_id -> info
# ----------------------------
# cache[tid] = {
#   "name": str,
#   "score": float,
#   "last_seen": float,
#   "last_recog": float,
#   "pending": bool
# }
cache = {}


def should_enqueue_recognition(tid: int, now: float) -> bool:
    info = cache.get(tid)
    if info is None:
        return True

    # If a request is pending, don't enqueue again
    if info.get("pending", False):
        return False

    # If never recognized yet, do it
    if info.get("name", "???") in ("???", "…", "Recognizing..."):
        return True

    # If Unknown, allow retry after RECOG_RETRY_S
    if info.get("name") == "Unknown" and (now - info.get("last_recog", 0.0)) >= RECOG_RETRY_S:
        return True

    return False



def evict_stale(now: float):
    stale = [tid for tid, info in cache.items() if (now - info["last_seen"]) > TRACK_TIMEOUT_S]
    for tid in stale:
        del cache[tid]


# ----------------------------
# BACKGROUND RECOGNITION (NO FREEZE)
# ----------------------------
recog_queue = queue.Queue(maxsize=MAX_RECOG_QUEUE)
recog_results = queue.Queue(maxsize=64)
stop_event = threading.Event()


def recognition_worker():
    """
    Receives tasks: (tid, crop_bgr, ts)
    Sends results:  (tid, name, score, ts)
    """
    while not stop_event.is_set():
        try:
            tid, crop, ts = recog_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        name, score = "Unknown", -1.0
        print ("Face Recogtion trigered : ************************")
        try:
            faces = app.get(crop)
            if len(faces) > 0:
                # pick largest face in the crop
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )
                if known_names:
                    name, score = recognize_embedding(faces[0].embedding, known_names, known_embs)
                    print ("************************")
                    print (name)
                else:
                    name, score = "Unknown", -1.0
            else:
                name, score = "Unknown", -1.0
        except Exception:
            name, score = "Unknown", -1.0

        try:
            recog_results.put_nowait((tid, name, float(score), ts))
        except queue.Full:
            pass

        recog_queue.task_done()


worker = threading.Thread(target=recognition_worker, daemon=True)
worker.start()


# ----------------------------
# DEPTHAI PIPELINE (Luxonis example-style)
# ----------------------------
with dai.Pipeline() as pipeline:

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    leftOutput = monoLeft.requestOutput((1280, 720))
    rightOutput = monoRight.requestOutput((280, 720))
    leftOutput.link(stereo.left)
    rightOutput.link(stereo.right)

    spatialDetectionNetwork = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        camRgb, stereo, "yolov6-nano"
    )
    objectTracker = pipeline.create(dai.node.ObjectTracker)

    spatialDetectionNetwork.setConfidenceThreshold(0.6)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(6000)
    labelMap = spatialDetectionNetwork.getClasses()

    objectTracker.setDetectionLabelsToTrack([0])  # person
    objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # Output queues
    preview = objectTracker.passthroughTrackerFrame.createOutputQueue()
    tracklets = objectTracker.out.createOutputQueue()

    if FULL_FRAME_TRACKING:
        camRgb.requestFullResolutionOutput().link(objectTracker.inputTrackerFrame)
        objectTracker.inputTrackerFrame.setBlocking(False)
        objectTracker.inputTrackerFrame.setMaxSize(1)
    else:
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)

    # FPS calc
    startTime = time.monotonic()
    counter = 0
    fps = 0.0

    print("\nStarting pipeline... Press 'q' to quit.")
    pipeline.start()

    try:
        while pipeline.isRunning():
            imgFrame = preview.get()
            track = tracklets.get()

            frame = imgFrame.getCvFrame()
            h, w = frame.shape[:2]
            trackletsData = track.tracklets

            # FPS update
            counter += 1
            now_mono = time.monotonic()
            if (now_mono - startTime) > 1:
                fps = counter / (now_mono - startTime)
                counter = 0
                startTime = now_mono

            now = time.time()

            # Apply recognition results (non-blocking)
            # --- Apply completed recognition results (MUST be early in loop) ---
            while True:
                try:
                    tid_r, name_r, score_r, ts_r = recog_results.get_nowait()
                except queue.Empty:
                    break

                print(f"[ui-update] tid={tid_r} -> {name_r} ({score_r:.2f})")

                # Create entry if missing (track might have been created after enqueue)
                if tid_r not in cache:
                    cache[tid_r] = {
                        "name": name_r,
                        "score": score_r,
                        "last_seen": time.time(),
                        "last_recog": time.time(),
                        "pending": False,
                    }
                else:
                    cache[tid_r]["name"] = name_r
                    cache[tid_r]["score"] = score_r
                    cache[tid_r]["pending"] = False
                    cache[tid_r]["last_seen"] = time.time()

                recog_results.task_done()


            # Draw tracked objects
            for t in trackletsData:
                tid = t.id
                status_name = t.status.name

                roi = t.roi.denormalize(w, h)
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                x1, y1 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1)
                x2, y2 = clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Spatial coordinates in mm (from SpatialDetectionNetwork -> ObjectTracker)
                Xmm = int(t.spatialCoordinates.x)
                Ymm = int(t.spatialCoordinates.y)
                Zmm = int(t.spatialCoordinates.z)

                try:
                    label = labelMap[t.label]
                except Exception:
                    label = str(t.label)

                # If lost/removed: clear cache entry immediately
                if status_name in ("LOST", "REMOVED"):
                    if tid in cache:
                        del cache[tid]
                    # Still draw it this frame if you want; otherwise skip
                    continue
                else:
                    # ensure cache entry exists
                    if tid not in cache:
                        cache[tid] = {
                            "name": "???",
                            "score": 0.0,
                            "last_seen": now,
                            "last_recog": 0.0,
                            "pending": False,
                        }
                    else:
                        cache[tid]["last_seen"] = now

                # Enqueue recognition ONCE per ID (non-blocking)
                # - Allow if Z is 0 (depth can be invalid sometimes)
                # - Or if within MAX_Z_MM
                print(
                    f"DEBUG -> ID:{tid} "
                    f"status:{status_name} "
                    f"Z:{Zmm} "
                    f"known_names:{len(known_names)} "
                    f"should_enqueue:{should_enqueue_recognition(tid, now)}"
                )

                if status_name in ("NEW", "TRACKED") and known_names and should_enqueue_recognition(tid, now):
                    crop = upper_body_crop(frame, x1, y1, x2, y2, w, h)

                    if crop.shape[0] >= MIN_CROP and crop.shape[1] >= MIN_CROP:
                        try:
                            # enqueue first
                            recog_queue.put_nowait((tid, crop.copy(), now))

                            # mark pending only if enqueue succeeded
                            cache[tid]["pending"] = True
                            cache[tid]["last_recog"] = now
                            cache[tid]["name"] = "Recognizing..."

                            print(f"[enqueue] tid={tid} queued recognition")
                        except queue.Full:
                            # don't set pending if queue is full
                            print("[enqueue] queue full, will retry")
                    else:
                        # crop too small: do NOT set pending
                        pass


                name = cache.get(tid, {}).get("name", "???")
                score = cache.get(tid, {}).get("score", 0.0)

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                cv2.putText(frame, f"{label}", (x1 + 8, y1 + 18),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, f"ID:{tid} {status_name}", (x1 + 8, y1 + 36),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, f"Name: {name} ({score:.2f})", (x1 + 8, y1 + 54),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, f"X:{Xmm}mm Y:{Ymm}mm Z:{Zmm}mm", (x1 + 8, y1 + 72),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

            # Evict old track IDs
            evict_stale(now)

            cv2.putText(frame, f"NN fps: {fps:.2f}", (2, frame.shape[0] - 6),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("OAK-D Tracker + Face Recognition (no freeze)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cv2.destroyAllWindows()
