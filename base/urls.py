from django.urls import path
from . import views

urlpatterns = [
    # JWT token endpoints
    # path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Your existing APIs

    path('api/text-audio-to-sign/', views.text_and_audio_to_sign_api, name='text_and_audio_to_sign_api'),
    path('api/sign-to-text/', views.sign_to_text_api, name='sign_to_text_api'),
]
