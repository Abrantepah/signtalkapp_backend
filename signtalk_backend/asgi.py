import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import base.signaling  # This refers to your base/signaling.py file

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'signtalk_backend.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            base.signaling.websocket_urlpatterns
        )
    ),
})