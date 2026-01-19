import os
from django.core.wsgi import get_wsgi_application

# This line MUST match your folder structure
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'signtalk_backend.settings')

application = get_wsgi_application()
