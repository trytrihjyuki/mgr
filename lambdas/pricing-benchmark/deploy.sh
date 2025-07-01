#!/bin/bash

# Ride-Hailing Pricing Benchmark Lambda Deployment Script
# Uses AWS Lambda Container Images for large dependencies

set -e

# Function to wait for Lambda function to be ready
wait_for_lambda_ready() {
    local function_name=$1
    local region=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for Lambda function to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        local state=$(aws lambda get-function --function-name "$function_name" --region "$region" --query 'Configuration.State' --output text 2>/dev/null || echo "NotFound")
        local last_update_status=$(aws lambda get-function --function-name "$function_name" --region "$region" --query 'Configuration.LastUpdateStatus' --output text 2>/dev/null || echo "NotFound")
        
        echo "   Attempt $attempt/$max_attempts: State=$state, LastUpdateStatus=$last_update_status"
        
        if [[ "$state" == "Active" && "$last_update_status" == "Successful" ]]; then
            echo "✅ Lambda function is ready"
            return 0
        elif [[ "$state" == "Failed" ]]; then
            echo "❌ Lambda function is in failed state"
            return 1
        fi
        
        echo "   Waiting 10 seconds before next check..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ Timeout waiting for Lambda function to be ready"
    return 1
}

# Function to retry AWS command with exponential backoff
retry_aws_command() {
    local max_attempts=5
    local attempt=1
    local delay=2
    
    while [ $attempt -le $max_attempts ]; do
        echo "   Attempt $attempt/$max_attempts..."
        
        if "$@"; then
            echo "✅ Command succeeded"
            return 0
        else
            local exit_code=$?
            echo "⚠️ Command failed with exit code $exit_code"
            
            if [ $attempt -eq $max_attempts ]; then
                echo "❌ Max attempts reached, command failed permanently"
                return $exit_code
            fi
            
            echo "   Waiting ${delay} seconds before retry..."
            sleep $delay
            delay=$((delay * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        fi
    done
}

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
    docker rmi rideshare-pricing-benchmark:$IMAGE_TAG 2>/dev/null && \
        echo "✅ Removed local Docker image" || echo "ℹ️ Local Docker image didn't exist"
    
    docker rmi $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG 2>/dev/null && \
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
ECR_REPO_NAME="rideshare-pricing-benchmark"
ROLE_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role"
IMAGE_TAG="latest"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

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
if ! aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${REGION} >/dev/null 2>&1; then
    echo "   Creating ECR repository: ${ECR_REPO_NAME}"
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${REGION}
else
    echo "   ECR repository already exists: ${ECR_REPO_NAME}"
fi

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

# Build the Docker image with platform specification for AWS Lambda compatibility
echo "🐳 Building Docker image for AWS Lambda (linux/amd64)..."

# Build for AWS Lambda (always linux/amd64)
echo "🖥️ Building single-architecture image for AWS Lambda compatibility..."

# Force single-architecture build without manifest list and attestations
if command -v docker buildx >/dev/null 2>&1; then
    echo "🔧 Using buildx with compatibility flags for AWS Lambda..."
    docker buildx build \
        --platform linux/amd64 \
        --provenance=false \
        --sbom=false \
        --output type=docker \
        --tag rideshare-pricing-benchmark:${IMAGE_TAG} \
        .
else
    echo "🔧 Using standard docker build..."
    docker build \
        --platform linux/amd64 \
        --tag rideshare-pricing-benchmark:${IMAGE_TAG} \
        .
fi

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker build completed"

# Verify image architecture
echo "🔍 Verifying image architecture..."
docker inspect rideshare-pricing-benchmark:${IMAGE_TAG} --format='{{.Architecture}}'

# Tag for ECR with a clean approach
echo "📤 Tagging image for ECR..."
docker tag rideshare-pricing-benchmark:${IMAGE_TAG} ${ECR_REPO}:latest

# Push the image to ECR
echo "📤 Pushing image to ECR..."
docker push ${ECR_REPO}:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi

# Verify image in ECR
echo "✅ Verifying image in ECR..."
aws ecr describe-images --repository-name ${ECR_REPO_NAME} --region ${REGION} --query 'imageDetails[0].imageTags[0]' --output text

# Update or create the Lambda function
echo "⚡ Creating/updating Lambda function..."
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} >/dev/null 2>&1; then
    echo "🔄 Updating existing function..."
    
    # Step 1: Update function code with retry logic
    echo "📝 Step 1/4: Updating function code..."
    retry_aws_command aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${ECR_REPO}:latest \
        --region ${REGION}
    
    # Wait for function to be ready before next update
    wait_for_lambda_ready ${FUNCTION_NAME} ${REGION}
    
    # Step 2: Update configuration with optimized settings for Hikima environment
    echo "📝 Step 2/4: Updating function configuration..."
    retry_aws_command aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --timeout 900 \
        --memory-size 10240 \
        --ephemeral-storage Size=2048 \
        --region ${REGION}
    
    # Wait for function to be ready before next update
    wait_for_lambda_ready ${FUNCTION_NAME} ${REGION}
    
    # Step 3: Update environment variables separately
    echo "📝 Step 3/4: Updating environment variables..."
    retry_aws_command aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --environment Variables="{PYTHONPATH=/var/task,S3_BUCKET=magisterka,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2}" \
        --region ${REGION}
    
    # Wait for function to be ready before next update
    wait_for_lambda_ready ${FUNCTION_NAME} ${REGION}
    
    # Step 4: Set reserved concurrency separately
    echo "📝 Step 4/4: Setting reserved concurrency..."
    retry_aws_command aws lambda put-function-concurrency \
        --function-name ${FUNCTION_NAME} \
        --reserved-concurrent-executions 5 \
        --region ${REGION}
else
    echo "🆕 Creating new function..."
    retry_aws_command aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ECR_REPO}:latest \
        --role ${ROLE_ARN} \
        --timeout 900 \
        --memory-size 10240 \
        --ephemeral-storage Size=2048 \
        --environment Variables="{PYTHONPATH=/var/task,S3_BUCKET=magisterka,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2}" \
        --region ${REGION}
    
    # Wait for function to be ready before setting concurrency
    wait_for_lambda_ready ${FUNCTION_NAME} ${REGION}
    
    # Set reserved concurrency after creation
    echo "📝 Setting reserved concurrency for new function..."
    retry_aws_command aws lambda put-function-concurrency \
        --function-name ${FUNCTION_NAME} \
        --reserved-concurrent-executions 5 \
        --region ${REGION}
fi

echo "✅ Deployment completed!"
echo "🔗 Function ARN: arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
echo "🧪 Testing function..."

# Test the function with Hikima environment parameters
# Use base64 encoding to avoid all encoding issues
TEST_JSON='{"test_mode":true}'
echo "📋 Using test payload: $TEST_JSON"

# Encode to base64 to avoid UTF-8 and compression issues
TEST_PAYLOAD_B64=$(echo -n "$TEST_JSON" | base64)
echo "📋 Base64 encoded payload: $TEST_PAYLOAD_B64"

aws lambda invoke \
    --function-name ${FUNCTION_NAME} \
    --payload "$TEST_PAYLOAD_B64" \
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
echo "  # Test imports (base64 encoded for reliability):"
echo "  PAYLOAD=\$(echo -n '{\"test_mode\":true}' | base64)"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload \"\$PAYLOAD\" test_output.json"
echo ""
echo "  # Run experiment using CLI (recommended):"
echo "  python ../../run_experiment.py --year=2019 --month=10 --days=1 --func=PL --methods=LP,MAPS,LinUCB"
echo ""
echo "  # Direct Lambda invocation with base64 encoding:"
echo "  JSON_PAYLOAD='{\"training_id\":\"123456789\",\"year\":2019,\"month\":10,\"day\":1,\"vehicle_type\":\"green\",\"acceptance_function\":\"PL\",\"methods\":[\"MinMaxCostFlow\"]}'"
echo "  PAYLOAD_B64=\$(echo -n \"\$JSON_PAYLOAD\" | base64)"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload \"\$PAYLOAD_B64\" experiment_output.json"
echo ""
echo "  # Quick one-liner for simple tests:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload \$(echo -n '{\"test_mode\":true}' | base64) test_output.json"
echo ""
echo "🔧 Troubleshooting:"
echo "  # Clean up and redeploy if issues occur:"
echo "  ./deploy.sh cleanup"
echo "  ./deploy.sh"
echo ""
echo "  # Check function logs:"
echo "  aws logs tail /aws/lambda/$FUNCTION_NAME --follow --region $AWS_REGION" 