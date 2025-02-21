# server.py
import asyncio
import websockets
import json
import cv2
import base64

async def send_frames(websocket, path):
    if path == "/first":
        while True:
            frame = cv2.imread('head_4.jpeg')
            #f = cv2.resize(f,(200,200))
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            
            # Send the base64-encoded frame to the client
            await websocket.send(base64_frame)
    else:
        await websocket.send("Invalid path")
        print("Invalid path")
        
        
async def start_server():
    # off_server = "localhost"
    on_server = "0.0.0.0"
    async with websockets.serve(send_frames, on_server, 8765):
        print("WebSocket server started at " + on_server)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(start_server())
