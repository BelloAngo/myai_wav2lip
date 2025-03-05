# Command to start the application
1. `sudo docker build -t myaiwav2lip .`
2. `sudo docker run --port 8000:8000 --gpus all -e SERVER_API_KEY=<api-key> --name server myaiwav2lip`