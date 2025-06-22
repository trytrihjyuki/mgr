#!/bin/bash

# AWS Lambda Deployment Script for Data Ingestion
set -e

FUNCTION_NAME="nyc-data-ingestion"
REGION="eu-north-1"
BUCKET_NAME="magisterka"

echo "🚀 Deploying Enhanced Data Ingestion Lambda Function"

# Clean up any previous deployment artifacts
rm -rf package lambda-deployment.zip 2>/dev/null || true

# Create deployment package with dependencies
echo "📦 Creating deployment package with dependencies..."

# Create package directory
mkdir -p package

# Install dependencies to package directory
echo "📥 Installing dependencies..."
pip install --target ./package requests boto3

# Copy our Lambda function
echo "📋 Copying Lambda function..."
cp lambda_function.py package/

# Create deployment zip
echo "🗜️  Creating deployment zip..."
cd package
zip -r ../lambda-deployment.zip . -q
cd ..

echo "📊 Package size: $(ls -lh lambda-deployment.zip | awk '{print $5}')"

# Check if function exists
echo "🔍 Checking if Lambda function exists..."
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION > /dev/null 2>&1; then
    echo "🔄 Function exists - updating code and configuration..."
    
    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $REGION
    
    # Update function configuration for bulk operations
    echo "⚙️  Updating function configuration for bulk operations..."
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout 900 \
        --memory-size 512 \
        --region $REGION \
        --environment Variables="{BUCKET_NAME=$BUCKET_NAME}" \
        --description "Enhanced NYC taxi data ingestion with multiple data sources and bulk download support"
    
    echo "✅ Function updated successfully!"
else
    echo "🆕 Function doesn't exist - creating new function..."
    
    # Create the function with appropriate settings for bulk operations
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://lambda-deployment.zip \
        --timeout 900 \
        --memory-size 512 \
        --region $REGION \
        --environment Variables="{BUCKET_NAME=$BUCKET_NAME}" \
        --description "Enhanced NYC taxi data ingestion with multiple data sources and bulk download support"
    
    echo "✅ Function created successfully!"
fi

# Wait for function to be active
echo "⏳ Waiting for function to be ready..."
aws lambda wait function-active --function-name $FUNCTION_NAME --region $REGION

# Get function info
echo "📋 Function Information:"
aws lambda get-function-configuration --function-name $FUNCTION_NAME --region $REGION | jq '{
    FunctionName,
    Runtime,
    Timeout,
    MemorySize,
    LastUpdateStatus,
    State
}'

echo ""
echo "🎉 Deployment completed successfully!"
echo "📝 Function Details:"
echo "   • Name: $FUNCTION_NAME"
echo "   • Timeout: 15 minutes (900 seconds)"
echo "   • Memory: 512 MB"
echo "   • Region: $REGION"
echo "   • S3 Bucket: $BUCKET_NAME"
echo ""
echo "⚡ Ready for bulk operations!"

# Clean up
rm -rf package lambda-deployment.zip

# Test the function
echo ""
echo "🧪 Testing enhanced function with sample data..."
sleep 5  # Give Lambda a moment to be ready

aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{
        "action": "download_single",
        "vehicle_type": "green", 
        "year": 2019,
        "month": 3,
        "limit": 100
    }' \
    --region $REGION \
    --cli-binary-format raw-in-base64-out \
    test-output.json > /dev/null

echo "📄 Test Results:"
if [[ -f test-output.json ]]; then
    # Parse response for better display
    status_code=$(cat test-output.json | jq -r '.statusCode // "unknown"' 2>/dev/null)
    response_body=$(cat test-output.json | jq -r '.body // ""' 2>/dev/null)
    
    echo "  Status Code: $status_code"
    if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
        echo "  Response:"
        echo "$response_body" | jq . 2>/dev/null || echo "$response_body"
    fi
    
    rm test-output.json
else
    echo "  ❌ No test output file generated"
fi

echo ""
echo "🎉 Deployment complete! Function is ready for use." 