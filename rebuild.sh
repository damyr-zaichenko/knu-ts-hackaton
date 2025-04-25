#!/bin/bash

# Script to rebuild the Docker image

cd '$(dirname "$0")'

# Rebuild the Docker image
echo "Rebuilding the Docker image..."
docker build --no-cache -t knu-ts-app .
