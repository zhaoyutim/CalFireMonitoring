from django.db import models


class Videos(models.Model):
    description = models.CharField(max_length=200)
    timestamp = models.DateTimeField('date published')
    bucket_link = models.CharField(max_length=200)
