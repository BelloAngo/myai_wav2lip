from os import listdir, path
import numpy as np
import scipy
import cv2
import os
import sys
import argparse
import audio
import json
import subprocess
import random
import string
from tqdm import tqdm
from glob import glob
import torch
import face_detection
from models import Wav2Lip
import platform
import pickle
import asyncio
import websockets
import base64

checkpoint_path = './checkpoints/wav2lip_gan.pth'
face = 'head_4.jpeg'
audio_path = './speech_20240314153225102.mp3'
outfile = 'results/result_voice.mp4'
static = False
fps = 25.0
pads = [0, 10, 0, 0]
face_det_batch_size = 1
wav2lip_batch_size = 1
resize_factor = 1
crop = [0, -1, 0, -1]
box = [-1, -1, -1, -1]
rotate = False
nosmooth = False
img_size = 96

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

if os.path.isfile(face) and face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    static = True


def save_list_to_pickle(lst, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(lst, file)


def load_list_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, filename):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size
    print(filename)
    loaded_list = load_list_from_pickle(
        filename.split('.')[0] +
        '.pickle') if os.path.exists(
        filename.split('.')[0] +
        '.pickle') else None
    if loaded_list is not None:
        return loaded_list
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(
                    np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError(
                'Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)]
               for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    # Save the list to the pickle file

    save_list_to_pickle(results, filename.split('.')[0] + '.pickle')
    return results


def datagen(frames, mels, filename):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if  static:
            # //TODO: Save the detection results so we do not have to compute and reduce latency
            face_det_results = face_detect([frames[0]], filename)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results = [
            [f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [
                    len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(
            mel_batch, [
                len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


async def main(websocket):
    if not os.path.isfile(face):
        raise ValueError(
            '--face argument must be a valid path to video/image file')

    # Here we taking the image
    elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        print("Image mode")
        full_frames = [cv2.imread(face)]

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
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

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, face)

    print("Generating")
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=1)):
        if i == 0:
            model = load_model(checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[i].shape[:-1]
            
        print("Size of the batch: {}".format(img_batch.shape))

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
            _, buffer = cv2.imencode('.jpg', f)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            
            # Send the base64-encoded frame to the client
            await websocket.send(base64_frame)
            
            # cv2.imshow('Client Side', f)
            # cv2.waitKey(1)
            
            break
            
            

    #command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio, 'temp/result.avi', outfile)
    #subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
    start_server = websockets.serve(main, "localhost", 8765,max_size=None)  # Adjust the host and port
    print("WebSocket server started.")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
