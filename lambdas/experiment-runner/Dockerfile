# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies with retry logic and skip update
# Note: Lambda base images are kept current, so yum update is not needed
RUN yum install -y gcc gcc-c++ make gfortran openblas-devel lapack-devel || \
    (sleep 5 && yum install -y gcc gcc-c++ make gfortran openblas-devel lapack-devel) || \
    (sleep 10 && yum install -y gcc gcc-c++ make gfortran openblas-devel lapack-devel) && \
    yum clean all && \
    rm -rf /var/cache/yum

# Set environment variables for numpy/scipy compilation
ENV OPENBLAS=/usr/lib64
ENV LAPACK=/usr/lib64
ENV ATLAS=None
ENV BLAS=/usr/lib64

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt --no-cache-dir

# Copy function code and source modules
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY ../../src ${LAMBDA_TASK_ROOT}/src

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
