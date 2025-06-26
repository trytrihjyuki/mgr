#!/bin/bash

# Ride-Hailing Pricing Benchmark Lambda Deployment Script
# Uses AWS Lambda Container Images for large dependencies

set -e

# Function to clean up existing deployment
cleanup_deployment() {
    echo "🧹 Cleaning up existing deployment..."
    
    # Delete Lambda function if it exists
    aws lambda delete-function --function-name $FUNCTION_NAME --region $AWS_REGION 2>/dev/null && \
        echo "✅ Deleted Lambda function" || echo "ℹ️ Lambda function didn't exist"
    
    # Delete ECR images
    aws ecr batch-delete-image \
        --repository-name $ECR_REPO_NAME \
        --region $AWS_REGION \
        --image-ids imageTag=$IMAGE_TAG 2>/dev/null && \
        echo "✅ Deleted ECR images" || echo "ℹ️ ECR images didn't exist"
    
    # Remove local Docker images
    docker rmi $ECR_REPO_NAME:$IMAGE_TAG 2>/dev/null && \
        echo "✅ Removed local Docker image" || echo "ℹ️ Local Docker image didn't exist"
    
    docker rmi $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG 2>/dev/null && \
        echo "✅ Removed tagged Docker image" || echo "ℹ️ Tagged Docker image didn't exist"
}

# Handle command line arguments
if [[ "$1" == "cleanup" ]]; then
    cleanup_deployment
    exit 0
fi

# Configuration
REGION=${AWS_REGION:-eu-north-1}
FUNCTION_NAME="rideshare-pricing-benchmark"
ROLE_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role"
IMAGE_TAG="pricing-benchmark:latest"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_NAME}"

echo "🚀 Deploying Ride-Hailing Pricing Benchmark Lambda Function - Hikima Environment"
echo "📊 Function: ${FUNCTION_NAME}"
echo "🌍 Region: ${REGION}"
echo "🏗️ Account: ${ACCOUNT_ID}"

# Check Docker platform support
echo "🔍 Checking Docker platform support..."
if command -v docker buildx >/dev/null 2>&1; then
    echo "✅ Docker buildx available for multi-platform builds"
else
    echo "❌ Docker buildx not available"
    exit 1
fi

# Check if we're on ARM64 and warn about build time
if [[ $(uname -m) == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon (ARM64). Building for linux/amd64 platform for AWS Lambda compatibility..."
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

# Build the Docker image with platform specification
echo "🐳 Building Docker image for AWS Lambda (linux/amd64)..."
docker buildx build \
    --platform linux/amd64 \
    --tag ${IMAGE_TAG} \
    --tag ${ECR_REPO}:latest \
    --load \
    .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Built with buildx"

# Verify image architecture
echo "🔍 Verifying image architecture..."
docker inspect ${IMAGE_TAG} --format='{{.Architecture}}'

# Tag for ECR
echo "📤 Tagging image for ECR..."
docker tag ${IMAGE_TAG} ${ECR_REPO}:latest

# Push the image to ECR
echo "📤 Pushing image to ECR..."
docker push ${ECR_REPO}:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi

# Verify image in ECR
echo "✅ Verifying image in ECR..."
aws ecr describe-images --repository-name ${FUNCTION_NAME} --region ${REGION} --query 'imageDetails[0].imageTags[0]' --output text

# Update or create the Lambda function
echo "⚡ Creating/updating Lambda function..."
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} >/dev/null 2>&1; then
    echo "🔄 Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${ECR_REPO}:latest \
        --region ${REGION}
    
    # Update configuration with optimized settings for Hikima environment
    aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --timeout 900 \
        --memory-size 10240 \
        --ephemeral-storage Size=2048 \
        --region ${REGION}
    
    # Update environment variables separately
    aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --environment Variables="{PYTHONPATH=/var/task,S3_BUCKET=magisterka,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2}" \
        --region ${REGION}
    
    # Set reserved concurrency separately
    aws lambda put-reserved-concurrency-configuration \
        --function-name ${FUNCTION_NAME} \
        --reserved-concurrent-executions 5 \
        --region ${REGION}
else
    echo "🆕 Creating new function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ECR_REPO}:latest \
        --role ${ROLE_ARN} \
        --timeout 900 \
        --memory-size 10240 \
        --ephemeral-storage Size=2048 \
        --environment Variables="{PYTHONPATH=/var/task,S3_BUCKET=magisterka,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2}" \
        --region ${REGION}
    
    # Set reserved concurrency after creation
    aws lambda put-reserved-concurrency-configuration \
        --function-name ${FUNCTION_NAME} \
        --reserved-concurrent-executions 5 \
        --region ${REGION}
fi

if [ $? -ne 0 ]; then
    echo "❌ Lambda function update failed"
    exit 1
fi

echo "✅ Deployment completed!"
echo "🔗 Function ARN: arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
echo "🧪 Testing function..."

# Test the function with Hikima environment parameters
aws lambda invoke \
    --function-name ${FUNCTION_NAME} \
    --payload '{"test_mode":true,"year":2019,"month":10,"day":1,"time_window":{"start_hour":10,"start_minute":0,"end_hour":10,"end_minute":5,"window_minutes":5},"vehicle_type":"green","acceptance_function":"PL","methods":["MinMaxCostFlow"],"training_id":"123456789"}' \
    --region ${REGION} \
    response.json

if [ $? -eq 0 ]; then
    echo "✅ Function test completed"
    echo "📋 Response:"
    cat response.json
    if command -v jq >/dev/null 2>&1; then
        echo "📄 Formatted response:"
        cat response.json | jq .
    fi
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
echo "   Memory: 10GB"
echo "   Timeout: 15 minutes"
echo "   Image URI: ${ECR_REPO}:latest"
echo ""
echo "You can now run Hikima-style experiments using:"
echo "python run_experiment.py --year=2019 --month=10 --days=1 --hours=10,20 --window=5 --func=PL --methods=MinMaxCostFlow"

echo "🎉 Lambda function deployment completed successfully!"
echo "📋 Usage examples:"
echo ""
echo "  # Test imports:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"test_mode\": true}' test_output.json"
echo ""
echo "  # Run experiment using CLI:"
echo "  python ../../run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=LP,MAPS,LinUCB"
echo ""
echo "  # Direct Lambda invocation:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{"
echo "    \"training_id\": \"123456789\","
echo "    \"year\": 2019,"
echo "    \"month\": 10,"
echo "    \"day\": 1,"
echo "    \"vehicle_type\": \"green\","
echo "    \"borough\": \"Manhattan\","
echo "    \"acceptance_function\": \"PL\","
echo "    \"methods\": [\"MinMaxCostFlow\", \"MAPS\", \"LinUCB\", \"LP\"]"
echo "  }' experiment_output.json"
echo ""
echo "🔧 Troubleshooting:"
echo "  # Clean up and redeploy if issues occur:"
echo "  ./deploy.sh cleanup"
echo "  ./deploy.sh"
echo ""
echo "  # Check function logs:"
echo "  aws logs tail /aws/lambda/$FUNCTION_NAME --follow --region $AWS_REGION" 