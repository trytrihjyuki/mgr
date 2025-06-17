# Simple Dockerfile for AWS Bipartite Matching Experiments
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_aws.txt .
RUN pip install --no-cache-dir -r requirements_aws.txt

# Copy application code
COPY aws_config.py .
COPY aws_s3_manager.py .
COPY aws_experiment_runner.py .

# Copy experiment code
COPY Rideshare_experiment/bin/ ./Rideshare_experiment/bin/
COPY Crowd_sourcing_experiment/bin/ ./Crowd_sourcing_experiment/bin/

# Create directories for temporary data
RUN mkdir -p /tmp/experiments

# Set environment variables
ENV PYTHONPATH=/app
ENV AWS_DEFAULT_REGION=us-east-1

# Default command
CMD ["python", "aws_experiment_runner.py", "--help"] 