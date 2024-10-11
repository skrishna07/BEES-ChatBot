from django.urls import re_path

from . import views

urlpatterns = [
    re_path('ai_search_chat', views.AISearchResponse),
    re_path('chat', views.AIResponse),
    re_path('ip-details', views.IPDetails),
    re_path('session-details', views.SessionDetails),
]
