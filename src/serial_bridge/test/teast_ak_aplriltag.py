#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import threading
from flask import Flask, Response

# ── Flask web stream setup ────────────────────────────────────────
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

def generate():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            _, jpeg = cv2.imencode('.jpg', latest_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <body style="background:#111; display:flex; justify-content:center">
        <img src="/video" style="max-width:100%; border:2px solid #444">
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ── Main depthai pipeline ─────────────────────────────────────────
fullFrameTracking = False

with dai.Pipeline() as pipeline:
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    leftOutput = monoLeft.requestOutput((640, 400))
    rightOutput = monoRight.requestOutput((640, 400))
    leftOutput.link(stereo.left)
    rightOutput.link(stereo.right)

    spatialDetectionNetwork = pipeline.create(
        dai.node.SpatialDetectionNetwork).build(camRgb, stereo, "yolov6-nano")
    objectTracker = pipeline.create(dai.node.ObjectTracker)

    spatialDetectionNetwork.setConfidenceThreshold(0.6)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)
    labelMap = spatialDetectionNetwork.getClasses()

    objectTracker.setDetectionLabelsToTrack([0])  # track only person
    objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    objectTracker.setTrackerIdAssignmentPolicy(
        dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    preview   = objectTracker.passthroughTrackerFrame.createOutputQueue()
    tracklets = objectTracker.out.createOutputQueue()

    if fullFrameTracking:
        camRgb.requestFullResolutionOutput().link(objectTracker.inputTrackerFrame)
        objectTracker.inputTrackerFrame.setBlocking(False)
        objectTracker.inputTrackerFrame.setMaxSize(1)
    else:
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    # Start Flask stream BEFORE pipeline loop
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True),
        daemon=True
    )
    flask_thread.start()
    print("Stream running → open http://192.168.1.126:5000 on Windows browser")

    pipeline.start()
    while pipeline.isRunning():
        imgFrame = preview.get()
        track    = tracklets.get()

        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets

        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frame, str(label),
                        (x1+10, y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"ID: {[t.id]}",
                        (x1+10, y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, t.status.name,
                        (x1+10, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm",
                        (x1+10, y1+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm",
                        (x1+10, y1+80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm",
                        (x1+10, y1+95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.putText(frame, f"NN fps: {fps:.2f}",
                    (2, frame.shape[0]-4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        # ← replaces cv2.imshow
        with frame_lock:
            latest_frame = frame.copy()