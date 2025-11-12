import os
import random
from moviepy import VideoFileClip, concatenate_videoclips
import csv
import subprocess
import tempfile



#search videos on one label and sentence level
def search_video_by_label(directory, label):
    label = str(label)
    """
    Searches for video files in the directory that match a given label.
    Matches files named:
        - '1.mp4'
        - '1A.mp4', '1B.avi', etc.
    Randomly returns one of the matching video file paths, or None if not found.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    valid_suffixes = ('A', 'B', 'C', 'D', 'E', 'F')
    matches = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                name_no_ext = os.path.splitext(file)[0]

                # Match exact label (e.g., '1.mp4')
                if name_no_ext == label:
                    matches.append(os.path.join(root, file))
                
                # Match label with suffix (e.g., '1A.mp4')
                elif (name_no_ext.startswith(label) and
                      len(name_no_ext) > len(label) and
                      name_no_ext[len(label)] in valid_suffixes):
                    matches.append(os.path.join(root, file))

    if matches:
        return random.choice(matches)
    
    return None



# Search avatar video for a single label (random if multiple files match)
def search_avatar_video_by_label(directory, label):
    label = str(label)
    """
    Searches for avatar video files in the directory with names starting with the given label.
    Randomly returns one of the matching video file paths, or None if none found.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    matches = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                file_name_without_ext = os.path.splitext(file)[0]
                if file_name_without_ext.startswith(label):
                    matches.append(os.path.join(root, file))

    if matches:
        return random.choice(matches)
    
    return None



def get_video_properties(video_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,codec_name,r_frame_rate',
            '-of', 'csv=p=0',
            video_path
        ]
        output = subprocess.check_output(cmd).decode().strip().split(',')
        width, height, codec, fps = output[0], output[1], output[2], output[3]
        fps_val = eval(fps) if '/' in fps else float(fps)
        return int(width), int(height), codec.strip(), fps_val
    except Exception as e:
        print(f"Could not read video properties for {video_path}: {e}")
        return None



def videos_have_same_properties(video_paths):
    """Check if all videos share the same codec, resolution, and fps."""
    props = [get_video_properties(p) for p in video_paths if p and os.path.exists(p)]
    props = [p for p in props if p]
    return len(set(props)) == 1 if props else False


def standardize_video(input_path, output_path, width=640, height=480, fps=30):
    """Convert video to uniform format for concatenation."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'scale={width}:{height},fps={fps}',
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'aac', '-strict', 'experimental',
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)  # 20 sec limit
    except subprocess.TimeoutExpired:
        print(f"Timeout: FFmpeg took too long on {input_path}")


def fast_concatenate_videos(video_paths, output_path):
    """Super-fast concatenation using FFmpeg (no re-encoding)."""
    valid_videos = [p for p in video_paths if p and os.path.exists(p)]
    if not valid_videos:
        print("No valid videos found to merge.")
        return None

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as list_file:
        for video in valid_videos:
            list_file.write(f"file '{os.path.abspath(video)}'\n")
        list_path = list_file.name

    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(list_path)
    print(f"Fast merged video saved to: {output_path}")
    return output_path


def concatenate_videos(video_paths, output_path):
    """Smart concatenation: try fast path; fallback to standardization."""
    valid_videos = [p for p in video_paths if p and os.path.exists(p)]
    if not valid_videos:
        print("No valid videos to merge.")
        return None

    if videos_have_same_properties(valid_videos):
        # Fast path: direct concat
        return fast_concatenate_videos(valid_videos, output_path)
    else:
        print("Videos not standardized — normalizing...")
        temp_dir = tempfile.mkdtemp()
        standardized_paths = []
        for i, path in enumerate(valid_videos):
            temp_out = os.path.join(temp_dir, f"std_{i}.mp4")
            standardize_video(path, temp_out)
            standardized_paths.append(temp_out)
        return fast_concatenate_videos(standardized_paths, output_path)


def save_labels_to_csv(labels, video_paths, output_csv_path):
    """Save mapping of labels and video paths to a CSV."""
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Label', 'VideoPath'])
        for label, path in zip(labels, video_paths):
            writer.writerow([label, path or "Not Found"])
    print(f"Labels CSV saved to: {output_csv_path}")

# #add concatenated avatar videos path to the function argument
# def search_word_videos_by_labels(directory, labels, concat_video_output_path):
#     """
#     Search for avatar videos for concatenated labels first.
#     If not found, search word-by-word and merge found clips efficiently.
#     """
#     labels_concat = " ".join([str(label) for label in labels])
#     avatar_video = search_avatar_video_by_label(directory, labels_concat)
#     if avatar_video:
#         return avatar_video

#     video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
#     label_to_files = {}

#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(video_extensions):
#                 file_name_without_ext = os.path.splitext(file)[0]
#                 label_number = ''.join(filter(str.isdigit, file_name_without_ext))
#                 label_to_files.setdefault(label_number, []).append(os.path.join(root, file))

#     matches = []
#     for label in labels:
#         label_str = str(label)
#         selected_video = random.choice(label_to_files[label_str]) if label_str in label_to_files else None
#         matches.append(selected_video)

#     merged_videos_path = concatenate_videos(matches, concat_video_output_path)

#     # Save label mapping CSV next to the output
#     csv_path = os.path.splitext(concat_video_output_path)[0] + "_labels.csv"
#     save_labels_to_csv(labels, matches, csv_path)

#     return merged_videos_path


def search_word_videos_by_labels(directory, labels, output_dir):
    """
    Search for avatar videos for concatenated labels first.
    If not found, search word-by-word and merge found clips efficiently.
    Save each merged output as numbered files (1.mp4, 2.mp4, etc.)
    and record mappings (number, sentence) in a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_output_path = os.path.join(output_dir, "output_mapping.csv")
    file_exists = os.path.exists(csv_output_path)

    # Initialize CSV (create header if file doesn't exist)
    with open(csv_output_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["ID", "Sentence", "VideoPath"])

        # generate next ID based on number of rows
        next_id = sum(1 for _ in open(csv_output_path, encoding='utf-8'))  # count lines
        sentence_text = " ".join(map(str, labels))
        output_video_path = os.path.join(output_dir, f"{next_id}.mp4")

        # Step 1: Check if full sentence avatar exists
        labels_concat = " ".join([str(label) for label in labels])
        avatar_video = search_avatar_video_by_label(directory, labels_concat)
        if avatar_video:
            writer.writerow([next_id, sentence_text, avatar_video])
            print(f"✅ Found sentence avatar for '{sentence_text}', saved as {next_id}.mp4")
            return avatar_video

        # Step 2: Word-by-word fallback
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        label_to_files = {}

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(video_extensions):
                    file_name_without_ext = os.path.splitext(file)[0]
                    label_number = ''.join(filter(str.isdigit, file_name_without_ext))
                    label_to_files.setdefault(label_number, []).append(os.path.join(root, file))

        matches = []
        for label in labels:
            label_str = str(label)
            selected_video = random.choice(label_to_files[label_str]) if label_str in label_to_files else None
            matches.append(selected_video)

        merged_videos_path = concatenate_videos(matches, output_video_path)

        # Step 3: Save mapping (ID, sentence, video)
        writer.writerow([next_id, sentence_text, merged_videos_path or "❌ Failed"])
        print(f"📄 Mapping saved → ID:{next_id}, Sentence:'{sentence_text}'")

        return merged_videos_path






# def search_word_videos_by_labels(directory, labels, concat_video_output_path):
#     """ 
#     Search for avatar videos for concatenated labels first. 
#     If not found, search word-by-word and randomly select a variation if multiple exist (like 1A, 1B).
#     Returns:
#         path to concatenated video
#     """
#     # 1️⃣ Check for a full sentence avatar video first
#     labels_concat = " ".join([str(label) for label in labels])
#     avatar_video = search_avatar_video_by_label(directory, labels_concat)
#     if avatar_video:
#         return avatar_video  # Single avatar video exists

#     # 2️⃣ Search word-by-word
#     video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
#     label_to_files = {}

#     # Index all available video files
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(video_extensions):
#                 file_name_without_ext = os.path.splitext(file)[0]
#                 # Extract numeric part as label number (e.g., 1A -> 1)
#                 label_number = ''.join(filter(str.isdigit, file_name_without_ext))
#                 if label_number not in label_to_files:
#                     label_to_files[label_number] = []
#                 label_to_files[label_number].append(os.path.join(root, file))

#     # Retrieve videos in order, randomly selecting one if multiple variations
#     matches = []
#     for label in labels:
#         label_str = str(label)
#         if label_str in label_to_files:
#             selected_video = random.choice(label_to_files[label_str])
#             matches.append(selected_video)
#         else:
#             matches.append(None)  # Video not found for this label

#     # Concatenate the found videos
#     merged_videos_path = concatenate_videos(matches, concat_video_output_path)
#     return merged_videos_path




# ######## To do: create a sheet with the merged videos labels and Ids ###############
# def concatenate_videos(video_paths, output_path):
#     clips = []
#     for path in video_paths:
#         if path and os.path.exists(path):
#             clip = VideoFileClip(path)
#             clips.append(clip)
#         else:
#             print(f"⚠️ Video not found: {path}")

#     if clips:
#         final_clip = concatenate_videoclips(clips, method="compose")  # 'compose' handles size/fps differences
#         final_clip.write_videofile(output_path, codec='libx264')
#         print(f"✅ Merged video saved to: {output_path}")
#         return output_path
#     else:
#         print("❌ No videos to merge.")
#         return None



# # Search video on words (try concatenated first, then fall back to individual words)
# # videos saved on this directory must be avatar videos for words
# def search_word_videos_by_labels(directory, labels, concat_video_output_path):
#     """ 
#     Search for avatar videos for concatenated labels before searching for word videos.
#     Returns:
#         list of matching video file paths (can include None if not found).
#     """
#     # Put the labels together as one sentence (e.g., ["thank", "you"] -> "thank you")
#     labels_concat = " ".join([str(label) for label in labels])
#     avatar_video = search_avatar_video_by_label(directory, labels_concat)
    
    
#     if avatar_video:
#         return avatar_video  # Return as a single-item list if found
    
#     # Otherwise, search word-by-word
#     video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
#     label_to_file = {}

#     # Index all available video files
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(video_extensions):
#                 file_name_without_ext = os.path.splitext(file)[0]
#                 label_to_file[file_name_without_ext] = os.path.join(root, file)

#     # Retrieve in labels order
#     matches = []
#     for label in labels:
#         file_path = label_to_file.get(str(label))
#         matches.append(file_path)  # Can be None if not found
#         #concatenate the found videos
#         merged_videos_path = concatenate_videos(matches, concat_video_output_path)

#     return merged_videos_path




# def search_word_gifs_by_labels(directory, labels):
#     """
#     For each label, tries to find a corresponding GIF file (exact match without extension).
#     Returns a list of matching GIF file paths, in the same order as labels.
#     """
#     if not isinstance(labels, (list, tuple)):
#         labels = [labels]

#     gif_extension = ('.gif',)
#     label_to_file = {}

#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(gif_extension):
#                 file_name_without_ext = os.path.splitext(file)[0]
#                 label_to_file[file_name_without_ext] = os.path.join(root, file)

#     matches = []
#     for label in labels:
#         file_path = label_to_file.get(str(label))
#         matches.append(file_path)

#     return matches





# def get_or_create_merged_video(labels, video_paths, media_root_path):
#     sentence_csv_path = media_root_path / "word_to_sentence_map.csv"
#     os.makedirs(sentence_csv_path.parent, exist_ok=True)

#     # Construct sentence string
#     sentence = " ".join(labels)

#     # Check if sentence exists in CSV
#     existing_id = None
#     if os.path.exists(sentence_csv_path):
#         with open(sentence_csv_path, mode='r', newline='', encoding='utf-8') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if len(row) >= 2 and row[1] == sentence:
#                     existing_id = row[0]
#                     break

#     if existing_id:
#         print(f"✅ Existing merged video found for sentence: '{sentence}' with ID: {existing_id}")
#         existing_video_path = media_root_path / "merged_sentences" / f"{existing_id}.mp4"
#         return str(existing_video_path), existing_id

#     # If not, merge videos
#     print(f"🔄 Merging new video for sentence: '{sentence}'")

#     # Determine next ID
#     if os.path.exists(sentence_csv_path):
#         with open(sentence_csv_path, mode='r', newline='', encoding='utf-8') as file:
#             reader = csv.reader(file)
#             ids = [int(row[0]) for row in reader if row and row[0].isdigit()]
#             next_id = max(ids, default=0) + 1
#     else:
#         next_id = 1

#     output_video_path = media_root_path / "merged_words_videos" / f"{next_id}.mp4"

#     # Concatenate videos
#     clips = []
#     for path in video_paths:
#         if path and os.path.exists(path):
#             clip = VideoFileClip(path)
#             clips.append(clip)
#         else:
#             print(f"⚠️ Video not found: {path}")

#     if clips:
#         final_clip = concatenate_videoclips(clips, method="compose")
#         final_clip.write_videofile(str(output_video_path), codec='libx264')
#         print(f"✅ Merged video saved to: {output_video_path}")

#         # Append mapping to CSV
#         with open(sentence_csv_path, mode='a', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow([next_id, sentence])

#         return str(output_video_path), next_id
#     else:
#         print("❌ No videos to merge.")
#         return None, None

