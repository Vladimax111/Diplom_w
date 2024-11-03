# object_detection/urls.py
from .views import delete_image
from . import views  # Эта строка импортирует модуль views из текущего пакета
from django.urls import path
from .views import home, register, dashboard, add_image_feed, login_view

urlpatterns = [
    path('detection_chart/', views.detection_chart, name='detection_chart'),
    path ( 'delete/<int:image_id>/' , delete_image , name = 'delete_image'),
    path('', home, name='home'),
    path('register/', register, name='register'),
    path('dashboard/', dashboard, name='dashboard'),
    path('add/', add_image_feed, name='add_image_feed'),
    path('login/', login_view, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('upload_image/', views.upload_image, name='upload_image'),
]

