from django.shortcuts import render

# Create your views here.
# object_detection/views.py

import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .forms import RegistrationForm, ImageUploadForm
from .models import ImageFeed


class_names = {0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorbike",
               5: "aeroplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
               10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter",
               14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
               20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
               25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase",
               30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite",
               35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard",
               39: "tennis racket", 40: "bottle", 41: "wine glass", 42: "cup", 43: "fork",
               44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple",
               49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog",
               54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch",
               59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "TV",
               64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone",
               69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator",
               74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear",
               79: "hair drier", 80: "toothbrush"}




def home(request):
    return render(request, 'object_detection/home.html')

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('dashboard')
    else:
        form = RegistrationForm()
    return render(request, 'object_detection/register.html', {'form': form})


@login_required
def dashboard(request):
    images = ImageFeed.objects.filter(user=request.user)
    return render(request, 'object_detection/dashboard.html', {'images': images})

@login_required
def add_image_feed(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_feed = form.save(commit=False)
            image_feed.user = request.user
            # Обработка изображения и применение модели здесь
            image_feed.save()
            return redirect('dashboard')
    else:
        form = ImageUploadForm()
    return render(request, 'object_detection/add_image_feed.html', {'form': form})



from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'object_detection/login.html', {'form': form})


from django.contrib.auth import logout
from django.shortcuts import redirect

def user_logout(request):
    logout(request)
    return redirect('login')






import os
import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from .forms import RegistrationForm
from .models import ImageFeed


prototxt_path = r"C:\Users\Zver\Desktop\detection_site\object_detection\MobileNetSSD_deploy.prototxt"
caffemodel_path = r"C:\Users\Zver\Desktop\detection_site\object_detection\mobilenet_iter_73000 (4).caffemodel"



net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def preprocess_image(image_path):
    return cv2.imread(image_path)

def detect_objects(image_path):
    image = preprocess_image(image_path)
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    return detections

def process_detections(detections, threshold=0.5):
    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= threshold:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            (startX, startY, endX, endY) = box.astype("int")
            detected_objects.append({
                'box': (startX, startY, endX, endY),
                'score': confidence,
                'class': int(detections[0, 0, i, 1])
            })
    return detected_objects

def draw_boxes(image_path, detected_objects):
    image = cv2.imread(image_path)
    for obj in detected_objects:
        (startX, startY, endX, endY) = obj['box']
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    result_path = r'C:\Users\Zver\Desktop\detection_site\object_detection\templates\object_detection/result.jpg'
    cv2.imwrite(result_path, image)
    return result_path



def upload_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        img_upload = ImageUpload(user=request.user, image=image)
        img_upload.save()

        input_tensor = preprocess_image(img_upload.image.path)
        detections = detect_objects(input_tensor)
        results = process_detections(detections)
        result_image_path = draw_boxes(img_upload.image.path, results)

        return render(request, 'results.html', {'results': results, 'result_image': result_image_path})

    return render(request, 'upload.html')


# object_detection/views.py
from django.shortcuts import render , redirect
from .forms import ImageUploadForm
from .models import ImageFeed
from django.contrib.auth.decorators import login_required
import cv2
import numpy as np


@login_required
def dashboard(request) :
    if request.method == 'POST' :
        form = ImageUploadForm ( request.POST , request.FILES )
        if form.is_valid ( ) :
            image_feed = form.save ( commit = False )
            image_feed.user = request.user
            image_feed.save ( )
            process_image ( image_feed )  # Ваша функция для обработки изображений
            return redirect ( 'dashboard' )
    else :
        form = ImageUploadForm ( )

    images = ImageFeed.objects.filter ( user = request.user )
    return render ( request , 'object_detection/dashboard.html' , {'form' : form , 'images' : images} )


def process_image(image_feed) :
    net = cv2.dnn.readNetFromCaffe ( 'mobilenet_ssd_deploy.prototxt' , 'mobilenet_iter_73000.caffemodel' )
    image = cv2.imread ( image_feed.image.path )

    # Проверьте, загружено ли изображение
    if image is None :
        print ( f"Error: Image at {image_feed.image.path} could not be loaded." )
        return

    blob = cv2.dnn.blobFromImage ( image , 0.007843 , (300 , 300) , 127.5 )
    net.setInput ( blob )
    detections = net.forward ( )

    found_object = False  # Флаг для проверки, был ли найден объект

    for i in range ( detections.shape [ 2 ] ) :
        confidence = detections [ 0 , 0 , i , 2 ]
        if confidence > 0.5 :  # Уровень уверенности
            class_id = int ( detections [ 0 , 0 , i , 1 ] )
            image_feed.result = str ( class_id )  # Здесь может быть сопоставление ID с классами
            image_feed.confidence = confidence
            found_object = True
            break

    if not found_object :
        image_feed.result = "No object detected"
        image_feed.confidence = 0.0

    image_feed.save ( )


import cv2
import os
from django.conf import settings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prototxt_path = os.path.join(BASE_DIR, 'object_detection', 'MobileNetSSD_deploy.prototxt')
caffemodel_path = os.path.join(BASE_DIR, 'object_detection', 'mobilenet_iter_73000 (4).caffemodel')


net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.template.loader import render_to_string
from django.core.mail import send_mail

def send_password_reset_email(user, request):
    token = default_token_generator.make_token(user)
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    domain = get_current_site(request).domain
    link = f'http://{domain}/reset/{uid}/{token}/'
    subject = 'Восстановление пароля'
    message = render_to_string('password_reset_email.html', {'link': link})
    send_mail(subject, message, 'from@example.com', [user.email])


from django.http import HttpResponse

@login_required
def delete_image(request, image_id):
    image_feed = ImageFeed.objects.get(id=image_id)
    if image_feed.user == request.user:
        image_feed.delete()
        # Удалите файл из файловой системы
        if image_feed.image:
            image_feed.image.delete(save=False)
        if image_feed.processed_image:
            image_feed.processed_image.delete(save=False)
    return redirect('dashboard')


def load_yolo_model():
    net = cv2.dnn.readNet(r"C:\Users\Zver\Desktop\detection_site\object_detection\yolov3.weights", r"C:\Users\Zver\Desktop\detection_site\object_detection\yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Инициализация модели
yolo_net, yolo_output_layers = load_yolo_model()


import matplotlib.pyplot as plt
from django.http import HttpResponse
import io
import base64

@login_required
def detection_chart(request):
    images = ImageFeed.objects.filter(user=request.user)
    object_counts = {}

    for image in images:
        detected_objects = image.detected_objects.all()
        for obj in detected_objects:
            object_counts[obj.object_type] = object_counts.get(obj.object_type, 0) + 1

    labels = list(object_counts.keys())
    sizes = list(object_counts.values())

    # Создание графика
    plt.figure(figsize=(10, 5))
    plt.bar(labels, sizes)
    plt.xlabel('Объекты')
    plt.ylabel('Количество')
    plt.title('Распознанные объекты')
    plt.xticks(rotation=45)

    # Сохранение графика в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_png = buf.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'object_detection/detection_chart.html', {'graph': graph})


@login_required
def delete_image(request, image_id):
    try:
        image_feed = ImageFeed.objects.get(id=image_id, user=request.user)
        image_feed.delete()
        return redirect('dashboard')
    except ImageFeed.DoesNotExist:
        return redirect('dashboard')


def detect_yolo_objects(image_path) :
    image = cv2.imread ( image_path )
    if image is None :
        print ( f"Error: Image at {image_path} could not be loaded." )
        return [ ] , [ ] , [ ]

    height , width = image.shape [ :2 ]
    blob = cv2.dnn.blobFromImage ( image , 0.00392 , (416 , 416) , (0 , 0 , 0) , True , crop = False )
    yolo_net.setInput ( blob )
    outputs = yolo_net.forward ( yolo_output_layers )

    boxes , confidences , class_ids = [ ] , [ ] , [ ]
    for output in outputs :
        for detection in output :
            scores = detection [ 5 : ]
            class_id = np.argmax ( scores )
            confidence = scores [ class_id ]
            if confidence > 0.5 :  # Порог уверенности
                center_x = int ( detection [ 0 ] * width )
                center_y = int ( detection [ 1 ] * height )
                w = int ( detection [ 2 ] * width )
                h = int ( detection [ 3 ] * height )
                x = int ( center_x - w / 2 )
                y = int ( center_y - h / 2 )
                boxes.append ( [ x , y , w , h ] )
                confidences.append ( float ( confidence ) )
                class_ids.append ( class_id )

    return boxes , confidences , class_ids


@login_required
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_feed = form.save(commit=False)
            image_feed.user = request.user
            image_feed.save()

            # Используем YOLO для распознавания объектов
            boxes, confidences, class_ids = detect_yolo_objects(image_feed.image.path)

            # Обработка результатов и сохранение их в базе данных
            for i in range(len(boxes)):
                (x, y, w, h) = boxes[i]
                label = str(class_ids[i])  # Здесь можно добавить сопоставление с именами классов
                confidence = confidences[i]

                DetectedObject.objects.create(
                    image_feed=image_feed,
                    object_type=label,
                    confidence=confidence,
                    location=f"{x},{y},{x + w},{y + h}"
                )

            return redirect('dashboard')
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})


