# Use the official AWS Lambda Python runtime
FROM public.ecr.aws/lambda/python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies first (they change less frequently)
RUN yum update -y && \
    yum install -y gcc g++ && \
    yum clean all && \
    rm -rf /var/cache/yum

# Create requirements file with pinned versions for reproducible builds
COPY <<EOF /tmp/requirements.txt
boto3==1.34.0
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
networkx==3.1
PuLP==2.7.0
scikit-learn==1.3.0
EOF

# Install Python dependencies (do this before copying code for better caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy the function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"] 