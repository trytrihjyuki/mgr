#!/bin/bash

# Professional AWS Lambda Deployment with Lambda Layers
# This separates dependencies from business logic (best practice)

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not running. Please install and start Docker to continue."
    exit 1
fi

set -e

FUNCTION_NAME="rideshare-experiment-runner"
LAYER_NAME="scientific-packages-layer"
REGION="${REGION:-eu-north-1}"
BUCKET_NAME="${BUCKET_NAME:-magisterka}"
export REGION BUCKET_NAME

echo "🚀 Professional Lambda Deployment with Lambda Layers"
echo "📋 Function: $FUNCTION_NAME"
echo "📋 Layer: $LAYER_NAME"
echo "📋 Region: $REGION"

# Clean up any existing artifacts
rm -rf layer-build function-build *.zip

# Step 1: Build Lambda Layer with dependencies using Docker
echo ""
echo "===== STEP 1: Building Lambda Layer using Docker ====="

# Create Dockerfile for building the layer
cat > Dockerfile.layer << EOF
FROM public.ecr.aws/lambda/python:3.9

# Create a directory for the layer
RUN mkdir -p /tmp/layer/python

# Install system dependencies
RUN yum update -y && yum install -y gcc gcc-c++ make

# Install dependencies into the layer directory with specific versions that work well
RUN pip install --upgrade pip
RUN pip install numpy==1.21.6 -t /tmp/layer/python --no-cache-dir
RUN pip install pandas==1.5.3 -t /tmp/layer/python --no-cache-dir
RUN pip install scipy==1.9.3 networkx==2.8.8 pyproj==3.4.1 geopy==2.3.0 pulp==2.7.0 -t /tmp/layer/python --no-cache-dir

# Install zip utility
RUN yum install -y zip

# Perform aggressive cleaning to reduce size
RUN find /tmp/layer/python -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
RUN find /tmp/layer/python -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
RUN find /tmp/layer/python -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find /tmp/layer/python -name "*.pyc" -delete 2>/dev/null || true
RUN find /tmp/layer/python -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
RUN find /tmp/layer/python -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
RUN find /tmp/layer/python -name "*.so" -type f -exec strip {} \; 2>/dev/null || true

# Remove any problematic source directories
# RUN find /tmp/layer/python -name "numpy" -type d -path "*/site-packages/*" -prune -o -name "numpy" -type d -exec rm -rf {} + 2>/dev/null || true
# (Disabled) Previously removed numpy packages to reduce size; keep numpy for execution

# Zip the layer
RUN cd /tmp/layer && zip -r /tmp/scientific-layer.zip .
EOF

# Build the Docker image
echo "📦 Building Docker image for layer..."
docker build -t lambda-layer-builder -f Dockerfile.layer .

# Run the Docker container and copy the layer zip out
echo "📦 Creating layer ZIP from Docker container..."
docker run --rm --entrypoint "" -v "$(pwd)":/aws lambda-layer-builder cp /tmp/scientific-layer.zip /aws/scientific-layer.zip

LAYER_SIZE=$(du -h scientific-layer.zip | cut -f1)
echo "📏 Layer size: $LAYER_SIZE (built in a clean Linux environment)"

# Clean up Docker artifacts
rm Dockerfile.layer
docker rmi lambda-layer-builder

# Step 2: Upload Layer to S3 and Deploy Lambda Layer
echo ""
echo "===== STEP 2: Deploying Lambda Layer via S3 ====="

echo "📤 Uploading layer ZIP to S3..."
aws s3 cp scientific-layer.zip s3://$BUCKET_NAME/lambda-layers/scientific-layer.zip

echo "📤 Publishing Lambda Layer from S3..."
LAYER_VERSION_ARN=$(aws lambda publish-layer-version \
    --layer-name $LAYER_NAME \
    --content S3Bucket=$BUCKET_NAME,S3Key=lambda-layers/scientific-layer.zip \
    --compatible-runtimes python3.9 \
    --description "Scientific packages (pandas, numpy, scipy, etc.)" \
    --region $REGION \
    --query LayerVersionArn \
    --output text)

echo "✅ Layer published: $LAYER_VERSION_ARN"

# Step 3: Build Lambda Function (business logic only)
echo ""
echo "===== STEP 3: Building Lambda Function ====="
mkdir -p function-build

# No dependencies needed here, they are all in the layer.
# The function package will only contain our source code.

# Copy the clean pricing benchmark implementation and the src folder
echo "📋 Copying pricing benchmark implementation and source code..."
cp lambda_function.py function-build/lambda_function.py
cp -r ../../src function-build/

# Clean function package and remove any potential numpy conflicts
cd function-build
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
# Remove any potential numpy-related directories or files that might conflict
find . -name "*numpy*" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*numpy*" -type f -delete 2>/dev/null || true
cd ..

# Create function zip
echo "📦 Creating function ZIP..."
cd function-build
zip -r9 ../function.zip . -q
cd ..

FUNCTION_SIZE=$(du -h function.zip | cut -f1)
echo "📏 Function size: $FUNCTION_SIZE (lightweight - dependencies in layer)"

# Step 4: Deploy Lambda Function with Layer
echo ""
echo "===== STEP 4: Deploying Lambda Function ====="

if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "📝 Updating existing Lambda function..."
    
    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://function.zip \
        --region $REGION
    
    echo "⏳ Waiting for function update to complete..."
    aws lambda wait function-updated --function-name $FUNCTION_NAME --region $REGION
    
    echo "🔗 Attaching Lambda Layer..."
    # Update function configuration to use the layer
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --layers "$LAYER_VERSION_ARN" \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME,PYTHONPATH=/opt/python:/var/task}" \
        --timeout 900 \
        --memory-size 1024 \
        --runtime python3.9 \
        --region $REGION

else
    echo "🆕 Creating new Lambda function with layer..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION)
    
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://function.zip \
        --layers "$LAYER_VERSION_ARN" \
        --timeout 900 \
        --memory-size 1024 \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME,PYTHONPATH=/opt/python:/var/task}" \
        --description "Pricing methods benchmark experiment runner (4 methods)" \
        --region $REGION
fi

# Cleanup
echo ""
echo "===== CLEANUP ====="
rm -rf layer-build function-build *.zip

echo ""
echo "🎉 Professional Lambda Deployment Complete!"
echo "✅ Function Name: $FUNCTION_NAME"
echo "✅ Layer ARN: $LAYER_VERSION_ARN"
echo "✅ Function Size: $FUNCTION_SIZE (business logic only)"
echo "✅ Layer Size: $LAYER_SIZE (dependencies separated)"
echo "✅ Memory: 1024MB"
echo "✅ Timeout: 15 minutes"
echo ""
echo "📋 Architecture:"
echo "   📦 Lambda Layer: pandas, numpy (compiled for Linux)"
echo "   🧠 Lambda Function: 4 pricing methods benchmark system"
echo "   🔗 Clean separation of concerns"

# Test the function
echo ""
echo "===== TESTING ====="
echo "🧪 Testing pricing methods benchmark system..."

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
    test-output.json

echo "📄 Test output:"
cat test-output.json && echo

# Check if successful
if grep -q '"statusCode": 200' test-output.json; then
    echo "✅ SUCCESS: Pricing methods benchmark system working!"
else
    echo "❌ Test failed - checking output above"
fi

rm test-output.json

echo ""
echo "🎯 Ready for systematic pricing method benchmarks with 4 algorithms!" 