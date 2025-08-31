# Multi-stage Dockerfile for taxi benchmark experiments
FROM python:3.9-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY run_experiment.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/configs /app/output /tmp/taxi_data_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Development stage
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest \
    black \
    flake8 \
    mypy


# Expose Jupyter port
EXPOSE 8888

# Run as root for development
USER root

CMD ["/bin/bash"] 