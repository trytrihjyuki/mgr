# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies for scientific packages
RUN yum update -y && \
    yum install -y gcc gcc-c++ make gfortran && \
    yum install -y openblas-devel lapack-devel && \
    yum clean all

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
