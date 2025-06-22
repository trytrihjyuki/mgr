#!/bin/bash

# Professional AWS Lambda Deployment with Lambda Layers
# This separates dependencies from business logic (best practice)
set -e

FUNCTION_NAME="rideshare-experiment-runner"
LAYER_NAME="scientific-packages-layer"
REGION="eu-north-1"
BUCKET_NAME="magisterka"

echo "🚀 Professional Lambda Deployment with Lambda Layers"
echo "📋 Function: $FUNCTION_NAME"
echo "📋 Layer: $LAYER_NAME"
echo "📋 Region: $REGION"

# Clean up any existing artifacts
rm -rf layer-build function-build *.zip

# Step 1: Build Lambda Layer with dependencies
echo ""
echo "===== STEP 1: Building Lambda Layer ====="
mkdir -p layer-build/python

echo "📦 Installing scientific packages for Lambda Layer..."
pip install pandas>=1.5.0 numpy>=1.21.0 -t layer-build/python/ \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --no-cache-dir \
    --upgrade

echo "🧹 Cleaning layer package..."
cd layer-build/python

# Remove unnecessary files from layer
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove numpy source directories (the root cause of import issues)
rm -rf numpy/core/src/ 2>/dev/null || true
rm -rf numpy/distutils/ 2>/dev/null || true
rm -rf numpy/f2py/src/ 2>/dev/null || true
rm -rf numpy/random/src/ 2>/dev/null || true

cd ../..

# Create layer zip
echo "📦 Creating layer ZIP..."
cd layer-build
zip -r9 ../scientific-layer.zip . -q
cd ..

LAYER_SIZE=$(du -h scientific-layer.zip | cut -f1)
echo "📏 Layer size: $LAYER_SIZE"

# Step 2: Deploy Lambda Layer
echo ""
echo "===== STEP 2: Deploying Lambda Layer ====="

echo "📤 Publishing Lambda Layer..."
LAYER_VERSION_ARN=$(aws lambda publish-layer-version \
    --layer-name $LAYER_NAME \
    --zip-file fileb://scientific-layer.zip \
    --compatible-runtimes python3.9 \
    --description "Scientific packages (pandas, numpy) for rideshare experiments" \
    --region $REGION \
    --query LayerVersionArn \
    --output text)

echo "✅ Layer published: $LAYER_VERSION_ARN"

# Step 3: Build Lambda Function (business logic only)
echo ""
echo "===== STEP 3: Building Lambda Function ====="
mkdir -p function-build

echo "📥 Installing minimal dependencies for function..."
# Only install boto3 for the function (pandas/numpy come from layer)
pip install boto3>=1.26.0 -t function-build/ --no-deps --quiet

# Copy the sophisticated Hikima implementation (NO CHANGES TO LOGIC)
echo "📋 Copying sophisticated Hikima implementation..."
cp lambda_function_heavy.py function-build/lambda_function.py

# Clean function package
cd function-build
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
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
    
    echo "🔗 Attaching Lambda Layer..."
    # Update function configuration to use the layer
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --layers "$LAYER_VERSION_ARN" \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
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
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --description "Sophisticated Hikima rideshare experiment runner" \
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
echo "   🧠 Lambda Function: sophisticated Hikima implementation"
echo "   🔗 Clean separation of concerns"

# Test the function
echo ""
echo "===== TESTING ====="
echo "🧪 Testing sophisticated Hikima implementation..."

aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{
        "vehicle_type": "green",
        "year": 2019,
        "month": 3,
        "place": "Manhattan",
        "simulation_range": 3,
        "acceptance_function": "PL",
        "start_hour": 15,
        "end_hour": 17
    }' \
    --region $REGION \
    test-output.json

echo "📄 Test output:"
cat test-output.json && echo

# Check if successful
if grep -q '"statusCode": 200' test-output.json; then
    echo "✅ SUCCESS: Sophisticated Hikima implementation working!"
else
    echo "❌ Test failed - checking output above"
fi

rm test-output.json

echo ""
echo "🎯 Ready for sophisticated rideshare experiments with real Hikima algorithms!" 