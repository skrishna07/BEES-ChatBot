from django.urls import re_path

from . import views

urlpatterns = [
    re_path('signup', views.signup),
    re_path('login', views.login),
    re_path('chat', views.AIResponse),
    re_path('new_token', views.regenerate_token),
]
