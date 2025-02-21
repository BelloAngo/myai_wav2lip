import base64
import io
import json
import logging
from typing import Generator

import cv2
import librosa
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from base64_video import convert
from runpod_wav2lip_util import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants (move these to a config file or environment variables in production)
SAMPLE_RATE = 16000
FPS = 25
MEL_STEP_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (move this to a separate function or module)
print("Loading Model")
model = load_model(checkpoint_path)
model.to(DEVICE)
print("Model loaded")


def decode_face_data(face_data: str) -> np.ndarray:
    """Decode base64 face data into a numpy array."""
    try:
        face_bytes = base64.b64decode(face_data)
        face_np = np.frombuffer(face_bytes, np.uint8)
        face = cv2.imdecode(face_np, cv2.IMREAD_COLOR)
        return face
    except Exception as e:
        logger.error(f"Error decoding face data: {e}")
        raise ValueError("Invalid face data")


def decode_audio_data(audio_base64: str) -> np.ndarray:
    """Decode base64 audio data into a numpy array."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_stream = io.BytesIO(audio_bytes)
        wav, _ = librosa.load(audio_stream, sr=SAMPLE_RATE)
        return wav
    except Exception as e:
        logger.error(f"Error decoding audio data: {e}")
        raise ValueError("Invalid audio data")


def generate_mel_chunks(wav: np.ndarray) -> list:
    """Generate mel-spectrogram chunks from audio data."""
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains NaN values. Add noise to the audio and try again."
        )

    mel_chunks = []
    mel_idx_multiplier = 80.0 / FPS
    i = 0

    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1

    logger.info(f"Generated {len(mel_chunks)} mel chunks")
    return mel_chunks


def process_frames(
    full_frames: list, mel_chunks: list, face: np.ndarray, wav
) -> Generator[str, None, None]:
    """Process frames and yield base64-encoded video chunks."""
    gen = datagen(full_frames.copy(), mel_chunks, face)
    frame_list = []
    frame_counter = 0

    for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(DEVICE)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(DEVICE)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            f = cv2.resize(f, (200, 200))
            frame_list.append(f[:, :, ::-1])

            if len(frame_list) == 25:  # Yield a batch of 25 frames
                start_duration = frame_counter / FPS
                frame_counter += len(frame_list)
                end_duration = frame_counter / FPS

                start_sample = int(start_duration * SAMPLE_RATE)
                end_sample = int(end_duration * SAMPLE_RATE)
                audio_segment = wav[start_sample:end_sample]

                base64_list = convert(frame_list, audio_segment, SAMPLE_RATE)
                yield json.dumps({"video": base64_list})
                frame_list = []

    # Yield remaining frames
    if frame_list:
        start_duration = frame_counter / FPS
        frame_counter += len(frame_list)
        end_duration = frame_counter / FPS

        start_sample = int(start_duration * SAMPLE_RATE)
        end_sample = int(end_duration * SAMPLE_RATE)
        audio_segment = wav[start_sample:end_sample]

        base64_list = convert(frame_list, audio_segment, SAMPLE_RATE)
        yield json.dumps({"video": base64_list})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            # Input validation
            if "face" not in data or "audio" not in data:
                await websocket.send_json(
                    {"error": "Missing 'face' or 'audio' in input"}
                )
                continue

            try:
                face = decode_face_data(data["face"])
                wav = decode_audio_data(data["audio"])
                mel_chunks = generate_mel_chunks(wav)
                full_frames = [face]

                for result in process_frames(full_frames, mel_chunks, face, wav):
                    await websocket.send_text(result)

            except ValueError as e:
                logger.error(f"Input validation error: {e}")
                await websocket.send_json({"error": str(e)})
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await websocket.send_json({"error": "An internal error occurred"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await websocket.close()


@app.get("/")
async def get():
    return HTMLResponse(content="<html><body><h1>WebSocket Server</h1></body></html>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
