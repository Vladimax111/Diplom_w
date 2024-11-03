from django.apps import AppConfig


class ObjectDetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'object_detection'


import os
import cv2
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Загрузка модели
net = cv2.dnn.readNetFromCaffe(
    'object_detection/MobileNetSSD_deploy.prototxt',
    'object_detection/mobilenet_iter_73000 (4).caffemodel'

)

# Классы, которые может распознавать модель
classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
              "dog", "horse", "motorbike", "person", "plant", "sheep",
              "sofa", "train", "tvmonitor"]

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Чтение изображения
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Подготовка к детекции
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Обработка детекций
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Порог уверенности
            idx = int(detections[0, 0, i, 1])
            results.append((classNames[idx], confidence))

    return {'results': results}

if __name__ == '__main__':
    app.run(debug=True)


import cv2
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Загрузка модели YOLO
net = cv2.dnn.readNet("object_detection/yolov3.weights", "object_detection/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Классы, которые может распознавать модель
with open("object_detection/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Чтение изображения
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Подготовка к детекции
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Обработка детекций
    results = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Порог уверенности
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                results.append((class_names[class_id], confidence, (x, y, w, h)))

    return {'results': results}

if __name__ == '__main__':
    app.run(debug=True)


