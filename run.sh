#!/bin/bash

# Check if the image name and container name are passed as arguments
IMAGE_NAME="knu-ts-app"
CONTAINER_NAME="knu-ts-container"

# Optional: Allow overriding the default image/container names via args
if [ -n "$1" ]; then
  IMAGE_NAME=$1
fi

if [ -n "$2" ]; then
  CONTAINER_NAME=$2
fi

echo "Running container..."

# Run the Docker container
docker run --rm -it --name $CONTAINER_NAME -v $(pwd):/knu-ts-hackathon -w /knu-ts-hackathon $IMAGE_NAME