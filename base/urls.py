from django.urls import path, re_path
from . import views

urlpatterns = [
    # JWT token endpoints
    # path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Session Connection API
    path('api/generate-code/', views.generate_code),
    path('api/join-code/', views.join_with_code),

    #Translation API
    path('api/text-audio-to-sign/', views.text_and_audio_to_sign_api),
    path('api/text-audio-to-sign-channel/', views.text_and_audio_to_sign_api_channel), # New endpoint using Channels
    path('api/sign-to-text/', views.sign_to_text_api),
]
