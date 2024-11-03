import cv2
import numpy as np
from django.core.files.base import ContentFile
from .models import ImageFeed, DetectedObject
import cv2
import numpy as np
from django.core.files.base import ContentFile
from .models import ImageFeed, DetectedObject

# Список классов, определяемых моделью YOLO
YOLO_LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Загрузка модели YOLO
def load_yolo_model():
    def load_yolo_model() :
        try :
            net = cv2.dnn.readNet (
                r"C:\Users\Zver\Desktop\detection_site\object_detection\yolov3.weights" ,
                r"C:\Users\Zver\Desktop\detection_site\object_detection\yolov3.cfg" )
            print ( "YOLO model loaded successfully." )
            return net
        except Exception as e :
            print ( f"Error loading YOLO model: {e}" )
            return None


# Обработка изображения
def process_image(image_feed_id):
    try:
        image_feed = ImageFeed.objects.get(id=image_feed_id)
        image_path = image_feed.image.path
        net = load_yolo_model()
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        img = cv2.imread(image_path)
        if img is None:
            print("Failed to load image")
            return False

        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Порог уверенности
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i[0]
            box = boxes[i]
            (x, y, w, h) = box
            label = str(YOLO_LABELS[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            DetectedObject.objects.create(
                image_feed=image_feed,
                object_type=label,
                location=f"{x},{y},{x + w},{y + h}",
                confidence=float(confidence)
            )

        result, encoded_img = cv2.imencode('.jpg', img)
        if result:
            content = ContentFile(encoded_img.tobytes(), f'processed_{image_feed.image.name}')
            image_feed.processed_image.save(content.name, content, save=True)

        return True

    except ImageFeed.DoesNotExist:
        print("ImageFeed not found.")
        return False

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

@login_required
def upload_image(request):
    if request.method == 'POST':
        form = ImageFeed(request.POST, request.FILES)
        if form.is_valid():
            image_feed = form.save(commit=False)
            image_feed.user = request.user
            image_feed.save()

            # Обработка изображения
            process_image(image_feed.id)

            return redirect('dashboard')
    else:
        form = ImageFeed()
    return render(request, 'upload.html', {'form': form})
