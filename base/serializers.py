from rest_framework import serializers
from .models import ConsultationSession


class ConsultationSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConsultationSession
        fields = [
            'id',
            'pairing_code',
            'doctor_connected',
            'patient_connected',
            'is_active',
        ]
