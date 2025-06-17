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
zip -r ../lambda-deployment.zip .
cd ..

# Deploy or update Lambda function
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION > /dev/null 2>&1; then
    echo "📝 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $REGION > /dev/null
else
    echo "🆕 Creating new Lambda function..."
    # Get AWS Account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION)
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://lambda-deployment.zip \
        --timeout 900 \
        --memory-size 1024 \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --region $REGION > /dev/null
fi

# Update environment variables and configuration
echo "🔧 Setting environment variables and configuration..."
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
    --timeout 900 \
    --memory-size 1024 \
    --region $REGION > /dev/null

# Cleanup
rm -rf package lambda-deployment.zip

echo "✅ Lambda function deployed successfully!"
echo "📊 Function Details:"
echo "  Name: $FUNCTION_NAME"
echo "  Region: $REGION"
echo "  Runtime: python3.9"
echo "  Memory: 1024MB"
echo "  Timeout: 900s"

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
    local status_code=$(cat test-output.json | jq -r '.statusCode // "unknown"' 2>/dev/null)
    local response_body=$(cat test-output.json | jq -r '.body // ""' 2>/dev/null)
    
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