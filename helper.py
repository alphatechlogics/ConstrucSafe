# helper.py

import subprocess

def reencode_video(input_path, output_path):
    ffmpeg_binary = 'ffmpeg'  # Use the system-installed ffmpeg
    command = [
        ffmpeg_binary,
        '-y',  # Overwrite output files without asking
        '-i', input_path,  # Input file
        '-c:v', 'libx264',  # Video codec
        '-preset', 'fast',  # Encoding speed
        '-crf', '22',  # Quality parameter (lower is better)
        '-c:a', 'aac',  # Audio codec
        '-b:a', '128k',  # Audio bitrate
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("FFmpeg executable not found.")
        return False
