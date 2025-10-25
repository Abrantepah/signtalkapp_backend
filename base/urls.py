from django.urls import path
from . import views

urlpatterns = [
    # path('', views.home, name='home'),
    path('api/text-audio-to-sign/', views.text_and_audio_to_sign_api, name='text_and_audio_to_sign_api'),
    path('api/sign-to-text/', views.sign_to_text_api, name='sign_to_text_api'),
]
