import random
import string
from django.utils import timezone
from datetime import timedelta
from .models import ConsultationSession


def generate_pairing_code(length=6):
    return ''.join(random.choices(string.digits, k=length))


def create_session():
    while True:
        code = generate_pairing_code()
        if not ConsultationSession.objects.filter(pairing_code=code).exists():
            break

    return ConsultationSession.objects.create(
        pairing_code=code,
        expires_at=timezone.now() + timedelta(minutes=10)
    )
