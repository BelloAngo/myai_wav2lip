from time import sleep
import runpod
import cv2
import base64

def handler(job):
    job_input = job["input"]["prompt"]
    video_path = 'input.mp4'  # Change this to your video file path
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return "Error"
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        _, buffer = cv2.imencode('.jpeg', frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        yield base64_frame
        
    cap.release()


runpod.serverless.start(
    {
        "handler": handler,
        "return_aggregate_stream": True,  # Ensures aggregated results are streamed back
    }
) 
