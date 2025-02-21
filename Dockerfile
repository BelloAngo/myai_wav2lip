# Use the official Python image from the Docker Hub
FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install the runpod package
RUN pip install -r requirements.txt

#Install Torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Command to run the application
CMD ["python", "runpod_wav2lip.py"]
