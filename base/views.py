from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework import status
import os, tempfile
from pathlib import Path
from django.conf import settings
<<<<<<< HEAD
=======
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes

>>>>>>> 67cea46 (with text to sign)

#import the necessary modules from base
from base.TextAndAudio2Sign import text_2_sign, audio, search_video
from base.SignToText import general_conversation, others



<<<<<<< HEAD
#Text and Audio to Sign API
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
=======

#Text and Audio to Sign API
@api_view(["POST"])
# @parser_classes([MultiPartParser, FormParser, JSONParser])
# @permission_classes([IsAuthenticated])
# @parser_classes([MultiPartParser, FormParser, JSONParser])
>>>>>>> 67cea46 (with text to sign)
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
            transcribed_text, video_ids = audio.process_audio_file(tmpfile_path)

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
                settings.MEDIA_URL + "sentence_avatar/" + os.path.basename(avatar_video_abs_path)
                if avatar_video_abs_path else None
            )

        # 6️⃣ Word-by-word mode
        elif mode == "word-by-word":

            labels = videos
            video_abs_path = search_video.search_word_videos_by_labels(word_dataset_path, labels, output_video_path)

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
<<<<<<< HEAD
@parser_classes([MultiPartParser, FormParser])
=======
# @parser_classes([MultiPartParser, FormParser])
# @permission_classes([IsAuthenticated])
# @parser_classes([MultiPartParser, FormParser])
>>>>>>> 67cea46 (with text to sign)
def sign_to_text_api(request):
    """
    API endpoint: Receives a sign language video and returns the translated text.
    """
    try:
        video_file = request.FILES.get("video")
        category = request.data.get("category", "general")  # default to 'general'

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
        if category == "general":
            # Use general model
            translation_text = general_conversation.predict_translation_from_video(tmpfile_path)
        else:
            # Use category-specific model
            translation_text = others.predict_translation_from_video(tmpfile_path, category)

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
