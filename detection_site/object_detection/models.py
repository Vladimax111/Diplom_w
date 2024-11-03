from django.db import models

# Create your models here.
import os
from django.db import models
from django.contrib.auth.models import User

class ImageFeed(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')
    processed_image = models.ImageField(upload_to='processed_images/', blank=True)
    result = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name



    def delete(self, *args, **kwargs):  # Удаляем файл из файловой системы
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        super().delete(*args, **kwargs)


class DetectedObject(models.Model):
    image_feed = models.ForeignKey(ImageFeed, related_name='detected_objects', on_delete=models.CASCADE)
    object_type = models.CharField(max_length=100)
    confidence = models.FloatField()
    location = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.object_type} ({self.confidence * 100}%) on {self.image_feed.image.name}"

