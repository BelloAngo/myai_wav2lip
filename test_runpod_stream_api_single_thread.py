import runpod
import os
import base64
import cv2
import numpy as np
from io import BytesIO
import time
import json
import queue

from test_response_from_file import decode_base64_audio,decode_base64_images,play_audio_video

from dotenv import load_dotenv
load_dotenv(override=True)

# Set the API key and endpoint
runpod.api_key = os.getenv('runpod_api_key')
endpoint = runpod.Endpoint("za2iqcbvgiufvk")

# Queue to store the frames
frame_queue = queue.Queue()

def save_frames_to_buffer():
    start_time = time.time()
    # Read the test_input.json as dictionary
    with open("test_input.json", "r") as f:
        input_data = json.load(f)
    
    run_request = endpoint.run(input_data)
    first = True
    for output in run_request.stream():
        if first:
            first = False
            print("First Frame Received")
            end_time = time.time()
            print("Time Taken", end_time - start_time)  # Show the time taken to receive the first frame
        frame_data = output
        frame_queue.put(frame_data)  # Save the batch of frames to the queue
        print("Frame Queue Size", frame_queue.qsize())

def display_frames_from_buffer():
    initial_frame_threshold = 1  # The frames start getting displayed from this threshold
    attempts = 0
    max_attempts = 5  # Number of max attempts to read the queue
    frames_ready = False

    while attempts < max_attempts:
        if frame_queue.qsize() < initial_frame_threshold and not frames_ready:
            continue
        else:
            frames_ready = True

        try:
            frame_data = frame_queue.get(timeout=2)  # Wait for up to 2 seconds for a new frame
            frame_data_list = json.loads(frame_data)  # Loads the frame data as a list
            base64_frames = frame_data_list['frames']
            base64_audio = frame_data_list['audio']
            # for data in frame_data_list['frames']:
            #     base64_data = base64.b64decode(data)  # Decode each frame in the batch
            #     frame_np = np.frombuffer(base64_data, np.uint8)  # Convert the base64 data to a numpy array
            #     frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

            #     if frame is not None:
            #         cv2.imshow('Received Frame', frame)
            #         cv2.waitKey(5)
            frames = decode_base64_images(base64_frames)
            audio_data, sample_rate = decode_base64_audio(base64_audio)
            # Set the frame rate (fps)
            fps = 25

            # Play the synchronized audio and video
            play_audio_video(frames, audio_data, sample_rate, fps)
            attempts = 0  # Reset attempts after successful frame display
        except queue.Empty:
            attempts += 1
            print(f"No new frames received. Attempt {attempts} of {max_attempts}.")

    print("No new frames received. Exiting.")

# Main execution flow
save_frames_to_buffer()
display_frames_from_buffer()

# Release resources
cv2.destroyAllWindows()
