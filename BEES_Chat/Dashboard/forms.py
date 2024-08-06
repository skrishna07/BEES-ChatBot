from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import User

class UserRegistrationForm(UserCreationForm):
    email=forms.EmailField(required=True)
    
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('user', 'User'),
    ]
    
    role=forms.ChoiceField(choices=ROLE_CHOICES)
    # role = forms.CharField(max_length=5, choices=ROLE_CHOICES, default='User')
    class Meta:
        model=User
        fields=('username','email','role','password1','password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already exists")
        return email