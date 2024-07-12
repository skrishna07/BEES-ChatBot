from django.contrib.auth.models import AbstractUser,Group
from django.db import models

class User(AbstractUser):
    email = models.EmailField(unique=True)
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('user', 'User'),
    )

    role = models.CharField(max_length=15, choices=ROLE_CHOICES)
    def __str__(self):
        return self.username

class Request(models.Model):
    id = models.AutoField(primary_key=True)  # Auto-incrementing primary key
    datetime = models.DateTimeField()  # Datetime field
    user = models.CharField(max_length=100)  # Assuming user is a string field
    status = models.CharField(max_length=50)  # Assuming status is a string field