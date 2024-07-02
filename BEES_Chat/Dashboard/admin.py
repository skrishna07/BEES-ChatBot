# Register your models here.

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User
from .forms import UserRegistrationForm

class CustomUserAdmin(UserAdmin):
    add_form = UserRegistrationForm
    model = User
    

admin.site.register(User, CustomUserAdmin)
