import cv2
import numpy as np
import pyaudio
import wave
from pydub import AudioSegment
import subprocess
import shutil
import os

# Load video file
cap = cv2.VideoCapture('test_video.mp4')
video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load audio file
audio = AudioSegment.from_file('test_audio.mp3', format='mp3')
audio_frame_rate = audio.frame_rate
audio_frame_count = audio.frame_count()
audio_channels = audio.channels

# Set output file parameters
output_filename = 'output.mp4'
output_frame_rate = video_frame_rate
output_frame_count = video_frame_count
output_channels = audio_channels

# Check if FFmpeg is available
try:
    subprocess.check_call(['ffmpeg', '-version'])
    use_ffmpeg = True
except:
    use_ffmpeg = False

if use_ffmpeg:
    # Use FFmpeg to synchronize video and audio
    temp_video_filename = 'temp_video.mp4'
    temp_audio_filename = 'temp_audio.mp3'
    shutil.copyfile('test_video.mp4', temp_video_filename)
    audio.export(temp_audio_filename, format='mp3')
    process = subprocess.Popen(['ffmpeg', '-y', '-i', temp_video_filename, '-i', temp_audio_filename, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-filter:v', f'fps={output_frame_rate}', '-filter:a', f'aresample=async=1:ocl=1:first_pts=0', '-c:a', 'aac', '-b:a', '256k', '-strict', '-2', '-f', 'segment', '-segment_time', '0.1', '-segment_format', 'mpegts', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)

    # while True:
    #     # Read frame from output stream
    #     output_data = process.stdout.read(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 3))

    #     if len(output_data) != 0:
    #         # Convert output data to numpy array and reshape to frame size
    #         output_frame = np.frombuffer(output_data, dtype=np.uint8).reshape((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))

    #         # Display synchronized video feed
    #         cv2.imshow('Synchronized Video Feed', output_frame)
    #         if cv2.waitKey(1) == ord('q'):
    #             break
    #     else:
    #         # Break if end of stream is reached
    #         break

    # # Release resources
    # process.terminate()
    # cv2.destroyAllWindows()
    # os.remove(temp_video_filename)
    # os.remove(temp_audio_filename)

else:
    # If FFmpeg is not available, exit with an error message
    print("FFmpeg is not available. Please install FFmpeg to use this script.")
    exit()
