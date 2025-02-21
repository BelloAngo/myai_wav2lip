import runpod
import os
import base64
import cv2
import numpy as np
from io import BytesIO
import threading
import queue
import time
import json

from dotenv import load_dotenv
load_dotenv(override=True)

# Set the API key and endpoint
runpod.api_key = os.getenv('runpod_api_key')
endpoint = runpod.Endpoint("za2iqcbvgiufvk")

# Queue to store the frames
frame_queue = queue.Queue()

"""

This function is called to read the input data from the json file and then share the data to the endpoint
The endpoint shares frames in a batch of 25 which is then saved in queue which is then used in display_frames_from_buffer for displaying.


"""
def save_frames_to_buffer():
    
    start_time = time.time()
    #Read the test_input.json as dictionary
    with open("test_input.json", "r") as f:
        input_data = json.load(f)
    
    
    
    run_request = endpoint.run(
        input_data
    )
    first = True
    for output in run_request.stream():
        if first == True:
            first = False
            print("First Frame Received")
            end_time = time.time()
            print("TIme Take", end_time - start_time) #Show the time taken to receive the first frame
        frame_data = output
        frame_queue.put(frame_data) #save the batch of frames to the queue
        print("Frame Queue Size", frame_queue.qsize())
        
        
def display_frames_from_buffer():
    initial_frame_threshold = 1 #The frames starts getting displayed from this threshold
    attempts = 0
    max_attempts = 5  #Number of max attempts to read the queue
    frames_ready = False
    
    while attempts < max_attempts:
        if frame_queue.qsize() < initial_frame_threshold and not frames_ready:
            continue
        else:
            frames_ready = True

        try:
            frame_data = frame_queue.get(timeout=2)  # Wait for up to 2 seconds for a new frame
            frame_data_list = json.loads(frame_data) # Loads the frame data as a list, as discussed earlier the every element in a frame_queue is a batch(list) of 25 frames
            for data in frame_data_list:
                #The login in here will be according to you on how you want to show the frames in your project in JS.
                base64_data = base64.b64decode(data) #Each frame in the batch is encoded in base64 show we decode it
                frame_np = np.frombuffer(base64_data, np.uint8) #Converting the base64 data to a numpy array 
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR) 
                
                if frame is not None:
                    cv2.imshow('Received Frame', frame)
                    cv2.waitKey(1)
            attempts = 0  # Reset attempts after successful frame display
        except queue.Empty:
            attempts += 1
            print(f"No new frames received. Attempt {attempts} of {max_attempts}.")
            
    print("No new frames received. Exiting.")

# Create and start the threads
save_thread = threading.Thread(target=save_frames_to_buffer)
display_thread = threading.Thread(target=display_frames_from_buffer)

save_thread.start()
display_thread.start()

# Join the threads
save_thread.join()
display_thread.join()

# Release resources
cv2.destroyAllWindows()
