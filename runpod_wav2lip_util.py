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
static = False
fps = 25.0
pads = [0, 10, 0, 0]
face_det_batch_size = 1
wav2lip_batch_size = 50
resize_factor = 1
crop = [0, -1, 0, -1]
box = [-1, -1, -1, -1]
rotate = False
nosmooth = False
img_size = 96

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

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
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
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
    return results


def datagen(frames, mels, filename):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        face_det_results = face_detect([frames[0]], filename)

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
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path,
                                weights_only=False,
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


