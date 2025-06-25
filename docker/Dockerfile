# PaliGemma2 Medical Inference Docker Image (Slim Version)
# FLARE 2025 Medical Multimodal VQA Challenge
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for optimal inference
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create application directory and models directory
WORKDIR /app
RUN mkdir -p /app/models

# Pre-download PaliGemma2 model during build
COPY download_models.py /app/download_models.py
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
RUN python /app/download_models.py && rm /app/download_models.py

# Copy inference script and related files
COPY inference.py /app/
COPY predict.sh /app/

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Make scripts executable
RUN chmod +x /app/predict.sh

# Set working directory
WORKDIR /app

CMD ["python", "inference.py", "--help"] 