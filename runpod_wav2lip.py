from time import sleep
import runpod
import cv2
import base64
import numpy as np
from runpod_wav2lip_util import *
import json
import io
import librosa
import soundfile as sf
from base64_video import convert

print("Loading Model")
model = load_model(checkpoint_path)
print("Model loaded")




def handler(job):
    face_data = job["input"]["face"]
    face_data = base64.b64decode(face_data)
    face_np = np.frombuffer(face_data, np.uint8)
    face = cv2.imdecode(face_np, cv2.IMREAD_COLOR)
    full_frames = [face]
    
    sample_rate = 16000
    
    audio_base64 = job["input"]["audio"]
    audio_bytes = base64.b64decode(audio_base64)
    audio_stream = io.BytesIO(audio_bytes)
    wav, _ = librosa.load(audio_stream, sr=sample_rate)
    mel = audio.melspectrogram(wav)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        return ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0

    # Breaking mel into chunks
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]
    
    gen = datagen(full_frames.copy(), mel_chunks, face)
    print("Generating")
    
    frame_h, frame_w = full_frames[0].shape[:-1]

    #TODO Dynamic base64 list
    frame_list = []
    frame_counter = 0
    total_frames = 0
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            gen):
        
        img_batch = torch.FloatTensor(
            np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(
            np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            
            #Resize the frame
            f = cv2.resize(f,(200,200))
            total_frames+=1
            frame_list.append(f[:,:,::-1])
            
            if len(frame_list) == 25:  # Yield when batch size is reached
                
                start_duration = frame_counter/fps
                frame_counter += len(frame_list)
                end_duration = frame_counter/fps
                
                # Number of samples per second (sample rate)
                start_sample = int(start_duration * sample_rate)
                end_sample = int(end_duration * sample_rate)
                # Extract the relevant portion of the audio
                audio_segment = wav[start_sample:end_sample]
                # # Save the audio segment to a BytesIO object
                # buffer = io.BytesIO()
                # sf.write(buffer, audio_segment, sample_rate, format='wav')
                # buffer.seek(0)
                # # Encode to base64
                # audio_sample_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                # data = {"frames":frame_list,"audio":audio_sample_base64}
                
                base64_list = convert(frame_list, audio_segment, sample_rate)
                data = {"video":base64_list}
                
                #Encode frame_list to json
                json_string = json.dumps(data)
                yield json_string
                frame_list = []
        
        if len(frame_list) == 25:  # Yield when batch size is reached
            start_duration = frame_counter/fps
            frame_counter += len(frame_list)
            end_duration = frame_counter/fps
            
            # Number of samples per second (sample rate)
            start_sample = int(start_duration * sample_rate)
            end_sample = int(end_duration * sample_rate)
            # Extract the relevant portion of the audio
            audio_segment = wav[start_sample:end_sample]
            # # Save the audio segment to a BytesIO object
            # buffer = io.BytesIO()
            # sf.write(buffer, audio_segment, sample_rate, format='wav')
            # buffer.seek(0)
            # # Encode to base64
            # audio_sample_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # data = {"frames":frame_list,"audio":audio_sample_base64}
            
            base64_list = convert(frame_list, audio_segment, sample_rate)
            data = {"video":base64_list}
            
            #Encode frame_list to json
            json_string = json.dumps(data)
            yield json_string
            frame_list = []

    if len(frame_list) > 0:
        start_duration = frame_counter/fps
        frame_counter += len(frame_list)
        end_duration = frame_counter/fps
        
        # Number of samples per second (sample rate)
        start_sample = int(start_duration * sample_rate)
        end_sample = int(end_duration * sample_rate)
        # Extract the relevant portion of the audio
        audio_segment = wav[start_sample:end_sample]
        # # Save the audio segment to a BytesIO object
        # buffer = io.BytesIO()
        # sf.write(buffer, audio_segment, sample_rate, format='wav')
        # buffer.seek(0)
        # # Encode to base64
        # audio_sample_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # data = {"frames":frame_list,"audio":audio_sample_base64}
        
        base64_list = convert(frame_list, audio_segment, sample_rate)
        data = {"video":base64_list}
        
        #Encode frame_list to json
        json_string = json.dumps(data)
        yield json_string
        frame_list = []    
    print("Total Frames",total_frames)       

runpod.serverless.start(
    {
        "handler": handler,
        "return_aggregate_stream": True,  # Ensures aggregated results are streamed back
    }
) 
