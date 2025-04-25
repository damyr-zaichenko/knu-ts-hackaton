# Base image
FROM python:3.10-slim

# Set working directory inside the container to match your repo structure
WORKDIR /knu-ts-hackathon

# Copy the whole repo into the container
COPY . /knu-ts-hackathon

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables for TimesFM cache paths
ENV HF_HOME=/knu-ts-hackathon/models/timesfm \
    TRANSFORMERS_CACHE=/knu-ts-hackathon/models/timesfm \
    TORCH_HOME=/knu-ts-hackathon/models/timesfm

# Set the default command to execute the entrypoint script
ENTRYPOINT ["sleep", "infinity"]