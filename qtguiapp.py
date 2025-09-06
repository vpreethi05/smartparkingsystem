#!/usr/bin/python

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QRect

import json
import requests
import sseclient
import threading
import sys
import math
import ast

APP_URL = 'http://localhost:5000/'

parkingAvailability = {}
parkingSlotJson = None

freeLabel = None
window = None
freeList = []

class ColorRect(QWidget):
    def __init__(self, color, idx):
        super().__init__()
        self.color = color
        self.setMinimumSize(100, 100)  # Set minimum size for each rectangle
        self.parkingIdx = idx + 1
        self.text = f"#{self.parkingIdx}"
        self.textColor = Qt.black
    def setColor(self, color):
        self.color = color
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Fill rectangle with the specified color
        painter.fillRect(self.rect(), QColor(self.color))
        # Draw border
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(self.rect())
        # Draw text centered in the rectangle
        painter.setPen(QPen(self.textColor))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        # Calculate text position (centered)
        text_rect = QRect(self.rect())
        painter.drawText(text_rect, Qt.AlignCenter, self.text)

def getParkingSlots():
    PARKING_SLOT_URL = APP_URL + 'parkingslots'
    response = requests.get(PARKING_SLOT_URL)
    if response.status_code == 200:
        return response.json()
    return None

# Function to listen for server messages
def listen_to_server():
    SUBSCRIBE_URL = APP_URL + 'stream'
    response = requests.get(SUBSCRIBE_URL, stream=True)
    global parkingSlotJson
    parkingSlotCount = len(parkingSlotJson)
    for line in response.iter_lines():
        if line:
            lineStr = line.decode('utf-8')
            start = lineStr.find('{')
            end = lineStr.find('}')
            if start != -1 and end != -1:
                dataStr = lineStr[start : end + 1]
                dictObj = ast.literal_eval(dataStr)
                parkingIdKey = 'parking-id'
                if parkingIdKey in dictObj:
                    parkingIdx = dictObj['parking-id']
                    idx = int(parkingIdx)
                    if idx < len(freeList):
                        freeList[idx] = dictObj['free']
                    # Update the free label
                    # global freeLabel
                    free = 'Free: ' + str(freeList.count(True))
                    freeLabel.setText(free)
                    # Update the color to the parking area
                    rect = parkingAvailability[parkingIdx]
                    color = 'red' if dictObj['free'] == False else 'green'
                    rect.setColor(color)
                else:
                    parkingCount = dictObj['available']
                    # if parkingCount != parkingSlotCount:
                    #    relayout()

# Relayout all the items
def relayout():
    verticalLayout = QVBoxLayout()

    # Create a horizontal layout for showing occupancy
    infoLayout = QHBoxLayout()
    # Get the free parking slots
    free = 'Free: ' + str(10)
    global freeLabel
    freeLabel = QLabel(free)
    freeLabel.setFixedWidth(100)
    infoLayout.addWidget(freeLabel, alignment=Qt.AlignTop)
    # Get the total parking slots
    total = 'Total: ' + str(20)
    totalLabel = QLabel(total)
    infoLayout.addWidget(totalLabel, alignment=Qt.AlignTop)

    # Create a container widget
    infoContainer = QWidget()
    infoContainer.setLayout(infoLayout)
    verticalLayout.addWidget(infoContainer, alignment=Qt.AlignTop)

    gridLayout = QGridLayout()
    gridLayout.setSpacing(10)  # Space between rectangles
    gridLayout.setContentsMargins(20, 20, 20, 20)  # Margins around the grid

    global parkingSlotJson
    parkingSlotJson = getParkingSlots()
    for data in parkingSlotJson:
        print("Data: ", data)

    totalLabel.setText('Total: ' + str(len(parkingSlotJson)))
    if len(parkingSlotJson) > 0:
        for index in enumerate(parkingSlotJson):
            freeList.append(True)
    freeLabel.setText('Free: ' + str(freeList.count(True)))

    MAX_COLUMNS = 4
    cols = len(parkingSlotJson)
    if len(parkingSlotJson) > MAX_COLUMNS:
        cols = MAX_COLUMNS
    rows = math.ceil(len(parkingSlotJson) / MAX_COLUMNS)

    for row in range(rows):
        for col in range(cols):
            # Alternate between red and green
            parkingData = parkingSlotJson[row + col]
            color = 'red' if parkingData['free'] == False else 'green'
            rect = ColorRect(color, row + col)
            parkingAvailability[parkingData['parking-id']] = rect
            gridLayout.addWidget(rect, row, col, alignment=Qt.AlignTop)

    parkingLotContainer = QWidget()
    parkingLotContainer.setLayout(gridLayout)
    verticalLayout.addWidget(parkingLotContainer, alignment=Qt.AlignTop)

    global window
    # Get the current layout
    old_layout = window.layout()
    if old_layout is not None:
        # Remove all items from the layout
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Remove the layout from the widget
        old_layout.setParent(None)

    window.setLayout(verticalLayout)

# Main function to start the listener in a new thread
def start_listening():
    listening_thread = threading.Thread(target=listen_to_server)
    listening_thread.daemon = True  # This makes the thread exit when the main program exits
    listening_thread.start()

def show_label():
    app = QApplication(sys.argv)

    # Create main window
    global window
    window = QWidget()
    window.setWindowTitle('Parking Lot')
    window.setGeometry(100, 100, 800, 300)  # x, y, width, height

    relayout()

    # Subscribe to the server
    start_listening()

    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    show_label()
