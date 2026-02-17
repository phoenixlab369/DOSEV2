#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)

# ORIG – explicit XLink removed in v3
# xoutVideo = pipeline.create(dai.node.XLinkOut)
# xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setVideoSize(1080, 720)

# Linking
# ORIG
# camRgb.video.link(xoutVideo.input)
# NEW – output queue straight from the node
videoQueue = camRgb.video.createOutputQueue()

# ORIG – entire `with dai.Device` block removed
# with dai.Device(pipeline) as device:
#   video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
#   while True:
# NEW – start the pipeline
pipeline.start()
while pipeline.isRunning():
    videoIn = videoQueue.get()  # blocking
    cv2.imshow("video", videoIn.getCvFrame())
    if cv2.waitKey(1) == ord('q'):
        break
