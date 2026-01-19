from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework import status
import os, tempfile
from pathlib import Path
from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone

from .models import ConsultationSession
from .serializers import ConsultationSessionSerializer
from .utils import create_session
import asyncio
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# import your audio/text processing modules
# import audio, text_2_sign, search_video



#import the necessary modules from base
from base.TextAndAudio2Sign import text_2_sign, audio, search_video
from base.SignToText import general_conversation, others, new_model_inference

@api_view(['POST'])
def generate_code(request):
    session = create_session()

    return Response({
        "pairing_code": session.pairing_code,
        "session_id": str(session.id),
        "expires_in_minutes": 10,
        "websocket_url": f"wss://signtalkgh.com/ws/consultation/{session.id}/"
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
def join_with_code(request):
    code = request.data.get('pairing_code')

    session = get_object_or_404(
        ConsultationSession,
        pairing_code=code,
        is_active=True
    )

    if session.is_expired():
        return Response({"error": "Code expired"}, status=400)

    if session.patient_connected:
        return Response({"error": "Patient already connected"}, status=400)

    session.patient_connected = True
    session.save()

    return Response({
        "session_id": str(session.id),
        "websocket_url": f"wss://signtalkgh.com/ws/consultation/{session.id}/"
    })


@api_view(["POST"])
def text_and_audio_to_sign_api_channel(request):
    """
    Receives text or audio from doctor, returns sign video paths,
    and broadcasts response to the session WebSocket group (doctor + patient).
    """
    transcribed_text = ""
    video_urls = []
    avatar_video_url = []
    mode = None

    try:
        session_id = request.data.get("session_id")
        if not session_id:
            return Response({"error": "session_id is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        # 1 Handle audio input
        audio_file = request.FILES.get("audio")
        if audio_file:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                for chunk in audio_file.chunks():
                    tmpfile.write(chunk)
                tmpfile_path = tmpfile.name

            # process audio → get transcription
            transcribed_text = audio.process_audio_file(tmpfile_path)

            # cleanup
            os.remove(tmpfile_path)

        # 2 Handle text input (overrides audio transcription if provided)
        input_text = request.data.get("text")
        if input_text:
            transcribed_text = input_text.strip()

        if not transcribed_text:
            return Response({"error": "No valid input (text or audio) provided."},
                            status=status.HTTP_400_BAD_REQUEST)

        # 3 Dataset paths
        media_root_path = Path(settings.MEDIA_ROOT)
        sentence_dataset_path = media_root_path / "sentence_avatar"
        word_dataset_path = media_root_path / "word_avatar"
        output_video_path = media_root_path / "created_avatar"

        # 4 Process text through pipeline
        response = text_2_sign.retrieve_video(transcribed_text)
        if response is None:
            payload = {
                "type": "broadcast_message",
                "message": {
                    "mode": None,
                    "transcribed_text": transcribed_text,
                    "avatar_paths": [],
                    "video_paths": [],
                }
            }
            # send to WebSocket group
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(f"consult_{session_id}", payload)

            return Response(payload["message"], status=status.HTTP_200_OK)

        mode = response.get("mode")
        videos = response.get("videos", [])

        # 5 Sentence mode
        if mode == "sentence":
            label = videos[0]
            video_abs_path = search_video.search_video_by_label(sentence_dataset_path, label)
            avatar_video_abs_path = search_video.search_avatar_video_by_label(sentence_dataset_path, label)

            video_urls.append(
                settings.MEDIA_URL + "sentence_avatar/" + os.path.basename(video_abs_path)
                if video_abs_path else None
            )
            avatar_video_url.append(
                settings.MEDIA_URL + "sentence_avatar/" + os.path.basename(video_abs_path)
                if video_abs_path else None
            )

        # 6 Word-by-word mode
        elif mode == "word-by-word":
            labels = videos
            video_abs_path = search_video.search_word_videos_by_labels(word_dataset_path, labels, output_video_path)
            if video_abs_path:
                avatar_video_url.append(settings.MEDIA_URL + "created_avatar/" + os.path.basename(video_abs_path))
            else:
                avatar_video_url.append(None)

        # 7 Final API response
        payload = {
            "type": "broadcast_message",
            "message": {
                "mode": mode,
                "transcribed_text": transcribed_text,
                "avatar_paths": avatar_video_url,
                "video_paths": video_urls,
            }
        }

        # Send response to WebSocket group so both doctor and patient receive it
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(f"consult_{session_id}", payload)

        return Response(payload["message"], status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




#Text and Audio to Sign API
@api_view(["POST"])
# @parser_classes([MultiPartParser, FormParser, JSONParser])
# @permission_classes([IsAuthenticated])
# @parser_classes([MultiPartParser, FormParser, JSONParser])
def text_and_audio_to_sign_api(request):
    """
    API endpoint: Receives text or audio and returns sign video paths.
    """
    transcribed_text = ""
    video_urls = []
    avatar_video_url = []
    mode = None

    try:
        # 1️⃣ Handle audio input
        audio_file = request.FILES.get("audio")
        if audio_file:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                for chunk in audio_file.chunks():
                    tmpfile.write(chunk)
                tmpfile_path = tmpfile.name

            # process audio → get transcription + sentence ids
            transcribed_text = audio.process_audio_file(tmpfile_path)

            # cleanup
            os.remove(tmpfile_path)

        # 2️⃣ Handle text input (overrides transcription if provided directly)
        input_text = request.data.get("text")
        if input_text:
            transcribed_text = input_text.strip()

        if not transcribed_text:
            return Response({"error": "No valid input (text or audio) provided."},
                            status=status.HTTP_400_BAD_REQUEST)

        # 3️⃣ Dataset paths
        media_root_path = Path(settings.MEDIA_ROOT)
        sentence_dataset_path = media_root_path / "sentence_avatar"
        word_dataset_path = media_root_path / "word_avatar"
        # avatar_dataset_paths = media_root_path / "sentence_avatar"
        output_video_path = media_root_path / "created_avatar"
        # output_video_path = r"C:\Users\Administrator\Desktop\datasets\merged_sentence.mp4"

        # 4️⃣ Process text through pipeline
        response = text_2_sign.retrieve_video(transcribed_text)
        if response is None:
            return Response({
                "mode": None,
                "transcribed_text": transcribed_text,
                "avatar_paths": [],
                "video_paths": []
            }, status=status.HTTP_200_OK)

        mode = response.get("mode")
        videos = response.get("videos", [])

        # 5️⃣ Sentence mode
        if mode == "sentence":
            label = videos[0]
            video_abs_path = search_video.search_video_by_label(sentence_dataset_path, label)
            avatar_video_abs_path = search_video.search_avatar_video_by_label(sentence_dataset_path, label)

            video_urls.append(
                settings.MEDIA_URL + "sentence_avatar/" + os.path.basename(video_abs_path)
                if video_abs_path else None
            )
            avatar_video_url.append(
                settings.MEDIA_URL + "sentence_avatar/" + os.path.basename(video_abs_path)
                if video_abs_path else None
            ) 

        # 6️⃣ Word-by-word mode
        elif mode == "word-by-word":

            labels = videos
            video_abs_path = search_video.search_word_videos_by_labels(word_dataset_path, labels, output_video_path)

            # print(f"video abs path for views: {video_abs_path}")

            if video_abs_path:
                avatar_video_url.append(settings.MEDIA_URL + "created_avatar/" + os.path.basename(video_abs_path))
            else:
                avatar_video_url.append(None)

        # 7️⃣ Final API response
        return Response({
            "mode": mode,
            "transcribed_text": transcribed_text,
            "avatar_paths": avatar_video_url,
            "video_paths": video_urls,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Sign to text API
@api_view(["POST"])
# @parser_classes([MultiPartParser, FormParser])
# @permission_classes([IsAuthenticated])
# @parser_classes([MultiPartParser, FormParser])
def sign_to_text_api(request):
    """
    API endpoint: Receives a sign language video and returns the translated text.
    """
    try:
        video_file = request.FILES.get("video")
        category = request.data.get("category", "full")  # default to 'general_conversion'

        if not video_file:
            return Response(
                {"error": "No video uploaded"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Save uploaded video to temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            for chunk in video_file.chunks():
                tmpfile.write(chunk)
            tmpfile_path = tmpfile.name

        # Run prediction pipeline with category
        # if category == "general":
        #     # Use general model
        #     translation_text = general_conversation.predict_translation_from_video(tmpfile_path)
        # else:
        #     # Use category-specific model
        translation_text = new_model_inference.predict_from_video(tmpfile_path, category)

        # Cleanup
        os.remove(tmpfile_path)

        return Response(
            {"translation": translation_text},
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
