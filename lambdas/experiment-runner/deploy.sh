#!/bin/bash

# AWS Lambda Container Image Deployment
# Using container images up to 10GB instead of layers (250MB limit)
# Reference: https://aws.amazon.com/fr/blogs/aws/new-for-aws-lambda-container-image-support/

set -e

FUNCTION_NAME="rideshare-experiment-runner"
REGION="${REGION:-eu-north-1}"
BUCKET_NAME="${BUCKET_NAME:-magisterka}"
ECR_REPOSITORY="rideshare-experiment-runner"
export REGION BUCKET_NAME

echo "🚀 AWS Lambda Container Image Deployment"
echo "📋 Function: $FUNCTION_NAME"
echo "📋 Region: $REGION"
echo "📋 ECR Repository: $ECR_REPOSITORY"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not running. Please install and start Docker to continue."
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if we can build for linux/amd64 platform
echo "🔍 Checking Docker platform support..."
if ! docker buildx version &> /dev/null; then
    echo "⚠️ Docker buildx not available - using standard build with platform flag"
fi

# Step 1: Create Dockerfile for Lambda container image
echo ""
echo "===== STEP 1: Creating Dockerfile for Lambda Container Image ====="

cat > Dockerfile << 'EOF'
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
EOF

echo "✅ Dockerfile created"

# Step 2: Create requirements.txt with all scientific packages
echo ""
echo "===== STEP 2: Creating requirements.txt with all packages ====="

cat > requirements.txt << 'EOF'
# Scientific computing packages
numpy==1.21.6
pandas==1.5.3
scipy==1.9.3
networkx==2.8.8

# Geospatial packages
pyproj==3.4.1
geopy==2.3.0

# Optimization packages
pulp==2.7.0

# AWS packages
boto3==1.26.137
botocore==1.29.137

# Additional utilities
python-dateutil==2.8.2
pytz==2023.3
EOF

echo "✅ Requirements file created"

# Step 3: Get AWS account ID and create ECR repository
echo ""
echo "===== STEP 3: Setting up ECR Repository ====="

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo "📋 Account ID: $ACCOUNT_ID"
echo "📋 ECR URI: $ECR_URI"

# Create ECR repository if it doesn't exist
if ! aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $REGION 2>/dev/null; then
    echo "🆕 Creating ECR repository..."
    aws ecr create-repository \
        --repository-name $ECR_REPOSITORY \
        --image-scanning-configuration scanOnPush=true \
        --region $REGION
    echo "✅ ECR repository created"
else
    echo "📝 ECR repository already exists"
fi

# Step 4: Build container image
echo ""
echo "===== STEP 4: Building Container Image ====="

echo "📦 Building Docker image for AWS Lambda (x86_64 platform)..."

# Method 1: Try with buildx for explicit platform and manifest control
if docker buildx version &> /dev/null; then
    echo "   Using docker buildx for platform-specific build..."
    # Disable attestations and use simple Docker manifest format
    export DOCKER_BUILDKIT=1
    export BUILDX_NO_DEFAULT_ATTESTATIONS=1
    
    # Create or use existing builder that supports linux/amd64
    docker buildx create --name lambda-builder --driver docker-container --use 2>/dev/null || \
    docker buildx use lambda-builder 2>/dev/null || true
    
    docker buildx build \
        --platform linux/amd64 \
        --provenance=false \
        --sbom=false \
        --output type=docker \
        -t $ECR_REPOSITORY:latest .
else
    echo "   Using standard docker build with platform flag..."
    # Fallback to standard docker build
    export DOCKER_BUILDKIT=1
    export BUILDX_NO_DEFAULT_ATTESTATIONS=1
    docker build --platform linux/amd64 --provenance=false --sbom=false -t $ECR_REPOSITORY:latest .
fi

echo "📦 Tagging image for ECR..."
docker tag $ECR_REPOSITORY:latest $ECR_URI:latest

# Verify image architecture
echo "🔍 Verifying image architecture..."
DOCKER_ARCH=$(docker inspect $ECR_REPOSITORY:latest --format='{{.Architecture}}')
DOCKER_OS=$(docker inspect $ECR_REPOSITORY:latest --format='{{.Os}}')
echo "📋 Image Architecture: $DOCKER_OS/$DOCKER_ARCH"

if [ "$DOCKER_ARCH" != "amd64" ] || [ "$DOCKER_OS" != "linux" ]; then
    echo "❌ Error: Image must be linux/amd64 for AWS Lambda compatibility"
    echo "   Current: $DOCKER_OS/$DOCKER_ARCH"
    exit 1
fi

DOCKER_IMAGE_SIZE=$(docker images $ECR_REPOSITORY:latest --format "table {{.Size}}" | tail -n +2)
echo "📏 Container image size: $DOCKER_IMAGE_SIZE"

# Step 5: Test container locally (optional)
echo ""
echo "===== STEP 5: Testing Container Locally ====="

echo "🧪 Testing container locally..."
# Run container in background for testing (ensure platform compatibility)
CONTAINER_ID=$(docker run -d --platform linux/amd64 -p 9000:8080 $ECR_REPOSITORY:latest)

# Wait a moment for container to start
sleep 3

# Test the container
echo "📄 Testing numpy import..."
curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
    -d '{"test_mode": "numpy_only"}' > local-test-output.json

echo "📄 Local test output:"
cat local-test-output.json && echo

# Stop the test container
docker stop $CONTAINER_ID >/dev/null
docker rm $CONTAINER_ID >/dev/null

# Check if test was successful
if grep -q '"statusCode": 200' local-test-output.json && grep -q 'imports_successful.*true' local-test-output.json; then
    echo "✅ Local test successful!"
else
    echo "❌ Local test failed - check output above"
    rm -f local-test-output.json
    exit 1
fi

rm -f local-test-output.json

# Step 6: Push image to ECR
echo ""
echo "===== STEP 6: Pushing Image to ECR ====="

echo "🔐 Logging in to ECR..."
if ! aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI; then
    echo "❌ Failed to login to ECR"
    exit 1
fi

# Clean up any existing latest tag in ECR to avoid manifest conflicts
echo "🧹 Cleaning up existing ECR images..."
aws ecr batch-delete-image \
    --repository-name $ECR_REPOSITORY \
    --image-ids imageTag=latest \
    --region $REGION 2>/dev/null || echo "   No existing images to clean"

echo "📤 Pushing image to ECR..."
if ! docker push $ECR_URI:latest; then
    echo "❌ Failed to push image to ECR"
    exit 1
fi

echo "✅ Image pushed successfully"

# Verify the image was pushed correctly
echo "🔍 Verifying image in ECR..."
ECR_IMAGE_DETAILS=$(aws ecr describe-images \
    --repository-name $ECR_REPOSITORY \
    --image-ids imageTag=latest \
    --region $REGION \
    --query 'imageDetails[0].[imageSizeInBytes,imageManifestMediaType,imagePushedAt]' \
    --output text 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "📋 ECR Image Details: $ECR_IMAGE_DETAILS"
    
    # Check if manifest type is compatible with Lambda
    MANIFEST_TYPE=$(echo "$ECR_IMAGE_DETAILS" | cut -f2)
    if [[ "$MANIFEST_TYPE" == *"docker"* ]] || [[ "$MANIFEST_TYPE" == *"application/vnd.docker"* ]]; then
        echo "✅ Manifest type is Lambda-compatible: $MANIFEST_TYPE"
    else
        echo "⚠️ Manifest type may not be Lambda-compatible: $MANIFEST_TYPE"
        echo "   Expected: application/vnd.docker.distribution.manifest.v2+json or similar"
    fi
else
    echo "⚠️ Could not verify image details in ECR"
fi

# Step 7: Create IAM role for Lambda (if needed)
echo ""
echo "===== STEP 7: Creating IAM Role ====="

# Create trust policy for Lambda
cat > lambda-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create execution policy for Lambda
cat > lambda-execution-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF

# Check if role exists
if aws iam get-role --role-name lambda-execution-role --region $REGION 2>/dev/null; then
    echo "📝 IAM role already exists"
else
    echo "🆕 Creating new IAM role..."
    aws iam create-role \
        --role-name lambda-execution-role \
        --assume-role-policy-document file://lambda-trust-policy.json \
        --region $REGION
    
    aws iam put-role-policy \
        --role-name lambda-execution-role \
        --policy-name lambda-execution-policy \
        --policy-document file://lambda-execution-policy.json \
        --region $REGION
    
    echo "⏳ Waiting for role to be available..."
    sleep 10
fi

# Cleanup policy files
rm lambda-trust-policy.json lambda-execution-policy.json
echo "✅ IAM role ready"

# Step 8: Create or update Lambda function
echo ""
echo "===== STEP 8: Deploying Lambda Function ====="

# Get the image digest from ECR
IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name $ECR_REPOSITORY \
    --image-ids imageTag=latest \
    --region $REGION \
    --query 'imageDetails[0].imageDigest' \
    --output text)

IMAGE_URI="${ECR_URI}@${IMAGE_DIGEST}"

echo "📋 Using image: $IMAGE_URI"

if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "📝 Updating existing Lambda function..."
    
    if ! aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $IMAGE_URI \
        --region $REGION; then
        echo "❌ Failed to update Lambda function code"
        exit 1
    fi
    
    echo "⏳ Waiting for function update to complete..."
    if ! aws lambda wait function-updated --function-name $FUNCTION_NAME --region $REGION; then
        echo "❌ Function update timed out or failed"
        exit 1
    fi
    
    echo "🔧 Updating function configuration..."
    if ! aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --timeout 900 \
        --memory-size 3008 \
        --region $REGION; then
        echo "❌ Failed to update Lambda function configuration"
        exit 1
    fi

else
    echo "🆕 Creating new Lambda function..."
    echo "📋 Image URI: $IMAGE_URI"
    echo "📋 Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role"
    
    if ! aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$IMAGE_URI \
        --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
        --timeout 900 \
        --memory-size 3008 \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --description "Pricing methods benchmark experiment runner (container image)" \
        --region $REGION; then
        echo "❌ Failed to create Lambda function"
        echo "💡 Common issues:"
        echo "   • Image architecture must be linux/amd64"
        echo "   • ECR image must be in same region as Lambda"
        echo "   • IAM role must have correct permissions"
        echo "   • Image manifest must be Docker v2 compatible"
        exit 1
    fi
fi

echo "✅ Lambda function deployed"

# Step 9: Test the deployed function
echo ""
echo "===== STEP 9: Testing Deployed Function ====="

echo "🧪 Testing numpy import on deployed function..."
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"test_mode": "numpy_only"}' \
    --region $REGION \
    --cli-binary-format raw-in-base64-out \
    test-output.json

echo "📄 Test output:"
cat test-output.json && echo

# Check if successful
if grep -q '"statusCode": 200' test-output.json && grep -q 'imports_successful.*true' test-output.json; then
    echo "✅ SUCCESS: Numpy and all imports working!"
    
    echo ""
    echo "🧪 Testing full pricing benchmark system..."
    
    aws lambda invoke \
        --function-name $FUNCTION_NAME \
        --payload '{
            "vehicle_type": "green",
            "year": 2019,
            "month": 10,
            "day": 1,
            "borough": "Manhattan",
            "scenario": "comprehensive_benchmark"
        }' \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        benchmark-test-output.json

    echo "📄 Benchmark test output:"
    cat benchmark-test-output.json && echo
    
    if grep -q '"statusCode": 200' benchmark-test-output.json; then
        echo "✅ SUCCESS: Full pricing benchmark system working!"
    else
        echo "⚠️ Benchmark test had issues - check output above"
    fi
    
    rm benchmark-test-output.json
else
    echo "❌ Import test failed - check output above"
fi

rm test-output.json

# Cleanup
echo ""
echo "===== CLEANUP ====="
echo "🧹 Cleaning up local Docker images and files..."
docker rmi $ECR_REPOSITORY:latest >/dev/null 2>&1 || true
# Clean up any dangling images from the build
docker image prune -f >/dev/null 2>&1 || true
# Clean up buildx builder if created
docker buildx rm lambda-builder >/dev/null 2>&1 || true
rm -f requirements.txt Dockerfile

echo ""
echo "🎉 AWS Lambda Container Image Deployment Complete!"
echo "✅ Function Name: $FUNCTION_NAME"
echo "✅ Image URI: $IMAGE_URI"
echo "✅ Container Size: $DOCKER_IMAGE_SIZE"
echo "✅ Memory: 3008MB (maximum for scientific computing)"
echo "✅ Timeout: 15 minutes"
echo ""
echo "📋 Architecture:"
echo "   📦 Container Image: All scientific packages included"
echo "   🧠 Lambda Function: Full 4 pricing methods benchmark system"
echo "   🔗 No size limitations (up to 10GB container image)"
echo ""
echo "🎯 Ready for full-scale pricing method benchmarks with all algorithms!"
echo "✅ Container Image Lambda deployed successfully" 