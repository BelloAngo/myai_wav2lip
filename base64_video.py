import base64
import io
import os
import cv2
import tempfile
import numpy as np
import librosa
from scipy.io import wavfile
from moviepy.editor import ImageSequenceClip, AudioFileClip

def create_video_from_frame_and_audio(frames, audio_chunk, sample_rate, output_file):
    # Create a video clip from the single image frame, repeating it
    
    video_clip = ImageSequenceClip(frames, fps=25)
    
    # Save the audio chunk to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        wavfile.write(temp_audio_path, sample_rate, (audio_chunk * 32767).astype(np.int16))  # Convert float to int16
    
    # Load audio from the temporary WAV file
    audio_clip = AudioFileClip(temp_audio_path)
    
    # Set audio to video clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write video to the temporary file
    video_clip.write_videofile(output_file, codec='libx264', fps=25, threads=4, audio_codec='aac', verbose=False)
    
    # Clean up the temporary audio file
    os.remove(temp_audio_path)

def video_to_base64(video_file):
    with open(video_file, "rb") as video:
        video_binary = video.read()
        video_base64 = base64.b64encode(video_binary).decode('utf-8')
    return video_base64

def save_base64_to_file(base64_string, output_base64_file):
    with open(output_base64_file, "w") as file:
        file.write(base64_string)

def main(frame_path, audio_stream, sample_rate, base64_output_file):
    
    frames = cv2.imread(frame_path)[:,:,::-1]
    
    #Extend frames to 100 
    frames = [frames] * 50
        
    # Load the audio chunk using librosa
    audio_chunk, _ = librosa.load(audio_stream, sr=sample_rate)
    
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_path = temp_video_file.name
        
        # Create the video with the in-memory audio chunk
        create_video_from_frame_and_audio(frames, audio_chunk, sample_rate, temp_video_path)
        
    # Convert video to base64
    base64_string = video_to_base64(temp_video_path)
    
    # Save base64 string to file
    save_base64_to_file(base64_string, base64_output_file)
    print(f"Base64 video saved to {base64_output_file}")
    
def convert(frames, audio_chunk, sample_rate):
        
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_path = temp_video_file.name
        
        # Create the video with the in-memory audio chunk
        create_video_from_frame_and_audio(frames, audio_chunk, sample_rate, temp_video_path)
        
    # Convert video to base64
    base64_string = video_to_base64(temp_video_path)
    
    return base64_string

if __name__ == "__main__":
    frame_path = 'head_4.jpeg'  # Path to the image frame
    audio_file = 'speech_20240314153225102.mp3'  # Path to the audio file
    base64_output_file = 'video_base64.txt'  # Output file for base64 string
    
    main(frame_path, audio_file, 16000,base64_output_file)
