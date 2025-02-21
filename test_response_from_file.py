"""
Testing the response from a response file

"""

import json
import base64
import io
import soundfile as sf
import numpy as np
import cv2
import pygame
import time

def decode_base64_audio(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)
    audio_io = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_io)
    # Write the audio data to a WAV file
    sf.write('output_temp.wav', audio_data, sample_rate)
    return audio_data, sample_rate
    
def decode_base64_images(base64_list):
    images = []
    for b64_str in base64_list:
        image_data = base64.b64decode(b64_str)
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        images.append(img)
    return images

def play_audio_video(images, audio_data, sample_rate, fps):
    num_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
    
    # Initialize Pygame mixer for audio
    pygame.mixer.init(frequency=sample_rate, channels=num_channels)
    # sound = pygame.sndarray.make_sound(audio_data)
    sound = pygame.mixer.Sound('output_temp.wav')
    sound.play()
    
    # Initialize Pygame display for video
    pygame.display.init()
    screen = pygame.display.set_mode((images[0].shape[1], images[0].shape[0]))

    # Calculate time per frame
    frame_time = 1.0 / fps

    for img in images:
        # Convert image to surface
        img_surface = pygame.surfarray.make_surface(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((1, 0, 2)))
        screen.blit(img_surface, (0, 0))
        pygame.display.flip()

        # Sleep to maintain frame rate
        time.sleep(frame_time)
    
    # Wait for the audio to finish
    while pygame.mixer.get_busy():
        pygame.time.delay(100)
    
    # pygame.quit()

if __name__ == "__main__":
    #Read JSON file
    with open('response.json', 'r') as file:
        response_data = json.load(file)
        
    print(type(response_data))
    print(response_data.keys())
    print(type(response_data['output']))
    print(type(response_data['output'][0]))
    lists = response_data['output']
    frame_counter = 0
    for list_item in lists:
        list_value = json.loads(list_item)
        print(type(list_value))
        print(list_value.keys())

        base64_frames = list_value['frames']
        base64_audio = list_value['audio']

        frames = decode_base64_images(base64_frames)
        frame_counter+=len(frames)
        audio_data, sample_rate = decode_base64_audio(base64_audio)

        # Set the frame rate (fps)
        fps = 25

        # Play the synchronized audio and video
        play_audio_video(frames, audio_data, sample_rate, fps)

    print("Total Frames: ",frame_counter)