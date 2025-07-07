# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies with retry logic and skip update
# Note: Lambda base images are kept current, so yum update is not needed
RUN yum install -y gcc gcc-c++ make || \
    (sleep 5 && yum install -y gcc gcc-c++ make) || \
    (sleep 10 && yum install -y gcc gcc-c++ make) && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"] 