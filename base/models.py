import uuid
from django.db import models
from django.utils import timezone


class ConsultationSession(models.Model):
    """
    Represents one live consultation between two devices
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # pairing
    pairing_code = models.CharField(max_length=6, unique=True)

    doctor_connected = models.BooleanField(default=False)
    patient_connected = models.BooleanField(default=False)

    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    def is_expired(self):
        return timezone.now() > self.expires_at

    def __str__(self):
        return f"Session {self.pairing_code}"
