#!/usr/bin/python

import cv2;
import time
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
import argparse
import threading

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaAllocMapped, cudaMemcpy
from flask import Flask, jsonify, Response

geometryStr = "iiii"

totalParkingSlot = 0
parkingSlotInfoLock = threading.Lock()

# In-memory data store
parkingSlots = []

# Define the Flask app
app = Flask(__name__)
subscribers = []

@app.route('/parkingslots', methods=['GET'])
def getParkingLots():
    with parkingSlotInfoLock:
        return jsonify(parkingSlots)

@app.route('/parkingslots/<int:parkingId>', methods=['GET'])
def getParkingInformation(parkingId):
    parkingSlot = next((lot for lot in parkingSlots if lot['parking-id'] == parkingId), None)
    with parkingSlotInfoLock:
        if parkingLot:
           return jsonify(parkingSlot)
    return jsonify({"error": "Not parking exists for this id"}), 404

@app.route('/publish', methods=['POST'])
def publish(message):
    print("New message to publish: ", message)
    for sub in subscribers:
        sub.put(message)
    return {"status": "published"}, 201

@app.route('/stream')
def stream():
    def event_stream():
        from queue import Queue
        q = Queue()
        subscribers.append(q)
        print("Subscribing to the service")
        try:
            while True:
                msg = q.get()
                print("Message: ", msg)
                yield f"data: {msg}\n\n"
        finally:
            subscribers.remove(q)
    return Response(event_stream(), mimetype="text/event-stream")

# Function to run the server
def run_flask():
    print("Starting the flask server...")
    app.run(port=5000)

# For our use case, this will be useful to identify the parking lot in the area
def identifyObjectBoundaryAuto(frame):
    # Step 1: Convert the color image to the grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Step 2: Blur the image using GaussianBlur algorithm
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 50, minLineLength = 50, maxLineGap = 20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Step 4: Find the contour of the object
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourList = []
    packedBoundingRectData = []
    with parkingSlotInfoLock:
        parkingSlots.clear()
        index = 0
        parkingId = 0;
        # Pack the coordinates of the bounding rect of the contour area
        for cnt in contours:
            x, y, width, height = cv2.boundingRect(cnt)
            approxSides = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approxSides) == 4 and x != 0 and y != 0 and width > 70 and height > 150:
                contourList.append(index)
                packedBoundingRectData.append(struct.pack(geometryStr, x, y, width, height))
                parkingId += 1
                parkingData = {}
                parkingData['parking-id'] = parkingId
                parkingData['free'] = True
                parkingData['x'] = x
                parkingData['y'] = y
                parkingData['width'] = width
                parkingData['height'] = height
                parkingSlots.append(parkingData)
            index += 1
    # print("Contour information: ", len(contourList))
    # Step 5: Draw the contour in the original image
    for cIndex in contourList:
        cv2.drawContours(frame, contours[cIndex], -1, (0, 128, 0), 3)
    cv2.imshow("Live video feed", frame)
    return packedBoundingRectData

# For our use case, this will be useful to identify the parking lot in the area
def identifyObjectBoundary(frame):
    # Step 1: Convert the color image to the grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Step 2: Blur the image using GaussianBlur algorithm
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Step 3: Apply thresholding to convert the grayscale image to binary image (i.e. the background will be turned into dark in contrast with the foreground)
    ret, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY_INV)
    # Step 4: Find the contour of the object
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contourList = []
    packedBoundingRectData = []
    with parkingSlotInfoLock:
        parkingSlots.clear()
        index = 0
        parkingId = 1;
        # Pack the coordinates of the bounding rect of the contour area
        for cnt in contours:
            x, y, width, height = cv2.boundingRect(cnt)
            approxSides = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approxSides) == 4 and x != 0 and y != 0 and width > 70 and height > 150:
                contourList.append(index)
                packedBoundingRectData.append(struct.pack(geometryStr, x, y, width, height))
                parkingData = {}
                parkingData['parking-id'] = parkingId
                parkingData['free'] = True
                parkingData['x'] = x
                parkingData['y'] = y
                parkingData['width'] = width
                parkingData['height'] = height
                parkingId += 1
                parkingSlots.append(parkingData)
            index += 1
    global totalParkingSlot
    if totalParkingSlot != len(contourList):
        totalParkingSlot = len(contourList)
        parkingInfo = {}
        parkingInfo['available'] = len(contourList)
        publish(parkingInfo)
    print("Identified parking information: ", len(contourList))
    # Step 5: Draw the contour in the original image
    # for cIndex in contourList:
    #    cv2.drawContours(frame, contours[cIndex], -1, (0, 128, 0), 3)
    # cv2.imshow("Live video feed", frame)
    return packedBoundingRectData

# Space - Take the snapshot of the live feed and save it locally as an image
def waitAndHandleKeyInput(frame):
    k = cv2.waitKey(1)
    if k%256 == 27:
       # ESC pressed
       print("Escape hit, closing...")
       return True 
    elif k%256 == 32:
       # SPACE pressed
       img_name = "opencv_frame_{}.png".format(img_counter)
       cv2.imwrite(img_name, frame)
    return False

def convertCudaFrameToCv2Frame(cudaFrame):
    # print("Cuda image information: ", cudaFrame.width, cudaFrame.height, cudaFrame.format)
    # Allocate CUDA mapped memory for efficient transfer
    cuda_mem = cudaAllocMapped(width=cudaFrame.width,
                               height=cudaFrame.height,
                               format=cudaFrame.format)
    # Copy the cudaImage to the cudaMappedMemory
    cudaMemcpy(cuda_mem, cudaFrame)

    # Convert cudaMappedMemory to a NumPy array
    # Determine the appropriate dtype based on the image format
    if cudaFrame.format == "rgb8" or cudaFrame.format == "rgba8":
        dtype = np.uint8
        channels = 4 if cudaFrame.format == "rgba8" else 3
    elif cudaFrame.format == "rgb32f" or cudaFrame.format == "rgba32f":
        dtype = np.float32
        channels = 4 if cudaFrame.format == "rgba32f" else 3
    else:
        print(f"Unsupported format: {cudaFrame.format}")
        sys.exit(1)

    return np.array(cuda_mem, copy=False).reshape(cudaFrame.height, cudaFrame.width, channels)

def checkIfTheAreaOccupied(boxArea, objArea, threshold):
    boxX1, boxY1, boxX2, boxY2 = boxArea
    objX1, objY1, objX2, objY2 = objArea
    xA = max(boxX1, objX1)
    yA = max(boxY1, objY1)
    xB = min(boxX2, objX2)
    yB = min(boxY2, objY2)
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxArea = (boxX2 - boxX1) * (boxY2 - boxY1)
    return ((interArea / boxArea) >= threshold)

if __name__ == '__main__':
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # dies with main thread
    flask_thread.start()

    # parse the command line
    parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
    try:
        print("Parsing command line argument...")
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    # net = detectNet(args.network, sys.argv, args.threshold)
    net = detectNet(model="ssd-mobilenet.onnx", labels="labels.txt",
                    input_blob="input_0", output_cvg="scores", output_bbox="boxes",
                    threshold=args.threshold)

    while True:
        # capture the next image
        cudaFrame = input.Capture()
        if cudaFrame is None: # timeout
            continue
        inputFrame = convertCudaFrameToCv2Frame(cudaFrame)
        packedBoundRectData = identifyObjectBoundary(inputFrame)
        # packedBoundRectData = identifyObjectBoundaryAuto(inputFrame)
        # detect objects in the image (with overlay)
        detections = net.Detect(cudaFrame, overlay=args.overlay)
        # print the detections
        # print("detected {:d} objects in image".format(len(detections)))
        totalAreaOccupied = 0
        for pSlotData in parkingSlots:
            oldState = pSlotData['free']
            if len(detections) > 0:
                occupied = False
                for detection in detections:
                    boxArea = [detection.Left, detection.Top, detection.Right, detection.Bottom]
                    objArea = [pSlotData['x'], pSlotData['y'], pSlotData['x'] + pSlotData['width'], pSlotData['y'] + pSlotData['height']]
                    occupied = checkIfTheAreaOccupied(boxArea, objArea, 0.75)
                    pSlotData['free'] = (occupied == False)
                    if occupied == True:
                        totalAreaOccupied += 1
                    if oldState != pSlotData['free']:
                        publish(pSlotData)
            else:
                pSlotData['free'] = True
                #if oldState != pSlotData['free']:
                publish(pSlotData)
        print("Total parking slot: ", len(packedBoundRectData))
        print("Occupied parking slot: ", totalAreaOccupied)
        # render the image
        output.Render(cudaFrame)
        # update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
        if waitAndHandleKeyInput(inputFrame) == True:
            break;
