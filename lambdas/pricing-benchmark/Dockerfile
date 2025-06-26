FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies for scientific packages
RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"] 