#!/bin/bash

set -e  # Exit on any error

# Configuration
REGION=${AWS_REGION:-us-east-1}
FUNCTION_NAME="pricing-benchmark"
ROLE_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role"
IMAGE_TAG="pricing-benchmark:latest"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_NAME}"

echo "🚀 Starting Lambda deployment for ${FUNCTION_NAME}"
echo "   Region: ${REGION}"
echo "   Platform: $(uname -m)"

# Check if we're on ARM64 and warn about build time
if [[ $(uname -m) == "arm64" ]]; then
    echo "⚠️  Detected ARM64 (Apple Silicon) - cross-platform build will take longer"
    echo "   Building for linux/amd64 platform required by AWS Lambda"
fi

# Create ECR repository if it doesn't exist
echo "📦 Setting up ECR repository..."
if ! aws ecr describe-repositories --repository-names ${FUNCTION_NAME} --region ${REGION} >/dev/null 2>&1; then
    echo "   Creating ECR repository: ${FUNCTION_NAME}"
    aws ecr create-repository --repository-name ${FUNCTION_NAME} --region ${REGION}
else
    echo "   ECR repository already exists: ${FUNCTION_NAME}"
fi

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

# Build the Docker image with platform specification and no cache for clean build
echo "🔨 Building Docker image for linux/amd64..."
echo "   This may take 10-15 minutes on Apple Silicon due to cross-platform compilation"
echo "   Progress will be shown below..."

# Use buildx for better cross-platform support
docker buildx build \
    --platform linux/amd64 \
    --tag ${IMAGE_TAG} \
    --tag ${ECR_REPO}:latest \
    --load \
    --progress=plain \
    --no-cache \
    .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker build completed successfully"

# Push the image to ECR
echo "📤 Pushing image to ECR..."
docker push ${ECR_REPO}:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi

echo "✅ Image pushed successfully"

# Update or create the Lambda function
echo "🔧 Updating Lambda function..."
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} >/dev/null 2>&1; then
    echo "   Updating existing function: ${FUNCTION_NAME}"
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${ECR_REPO}:latest \
        --region ${REGION}
    
    # Update configuration with increased timeout and memory
    aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --timeout 900 \
        --memory-size 3008 \
        --region ${REGION}
else
    echo "   Creating new function: ${FUNCTION_NAME}"
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ECR_REPO}:latest \
        --role ${ROLE_ARN} \
        --timeout 900 \
        --memory-size 3008 \
        --region ${REGION}
fi

if [ $? -ne 0 ]; then
    echo "❌ Lambda function update failed"
    exit 1
fi

echo "✅ Lambda function updated successfully"

# Test the function
echo "🧪 Testing the function..."
PAYLOAD='{"year": 2019, "month": 10, "day": 1, "vehicle_type": "taxi", "methods": ["MinMaxCostFlow"], "acceptance_function": "PL", "training_id": "123456789"}'

aws lambda invoke \
    --function-name ${FUNCTION_NAME} \
    --payload "${PAYLOAD}" \
    --region ${REGION} \
    response.json

if [ $? -eq 0 ]; then
    echo "✅ Function test completed"
    echo "📋 Response:"
    cat response.json | jq .
    rm -f response.json
else
    echo "❌ Function test failed"
    if [ -f response.json ]; then
        echo "📋 Error response:"
        cat response.json
        rm -f response.json
    fi
fi

echo ""
echo "🎉 Deployment completed!"
echo "   Function name: ${FUNCTION_NAME}"
echo "   Region: ${REGION}"
echo "   Image URI: ${ECR_REPO}:latest"
echo ""
echo "You can now run experiments using:"
echo "python ../../run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=MinMaxCostFlow" 