from django.urls import re_path
from . import consumers # This looks for the consumers.py file you just made

websocket_urlpatterns = [
    # This captures the session_id from your Flutter URL
    re_path(r'ws/consultation/(?P<session_id>[^/]+)/$', consumers.ConsultationConsumer.as_asgi()),
]