# helper.py

import cv2
import subprocess
import os

def create_video_writer(video_cap, output_filename):
    """
    Initializes the OpenCV VideoWriter with the 'mp4v' codec.
    """
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0  # Default FPS if unable to get from the video

    # Initialize the VideoWriter with 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    return writer

def reencode_video(input_path, output_path):
    ffmpeg_binary = os.path.join(os.getcwd(), 'bin', 'ffmpeg')  # Adjust the path as needed
    command = [
        ffmpeg_binary,
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
