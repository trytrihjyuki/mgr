#!/bin/bash

# AWS Lambda Deployment Script for Simplified Experiment Runner
set -e

FUNCTION_NAME="rideshare-experiment-runner"
REGION="eu-north-1"
BUCKET_NAME="magisterka"

echo "🚀 Deploying Simplified Experiment Runner Lambda Function"

# Clean up any existing package  
rm -rf lambda-package lambda-deployment.zip

# Create deployment package (simplified - no heavy dependencies needed)
echo "📦 Creating simplified deployment package..."

mkdir -p lambda-package

# Install only boto3 (already available in Lambda runtime, but add for completeness)
echo "📥 Installing minimal dependencies..."
pip install boto3>=1.26.0 -t lambda-package/ --no-deps --quiet

# Copy minimal lambda function
cp lambda_function_minimal.py lambda-package/lambda_function.py

# Clean up the package
cd lambda-package
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
cd ..

# Create zip package
echo "📦 Creating deployment ZIP..."
cd lambda-package
zip -r9 ../lambda-deployment.zip . -q
cd ..

# Check package size
PACKAGE_SIZE=$(du -h lambda-deployment.zip | cut -f1)
echo "📏 Package size: $PACKAGE_SIZE (simplified version)"

# Deploy or update Lambda function
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "📝 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $REGION
        
    # Update function configuration
    echo "🔧 Updating function configuration..."
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --timeout 900 \
        --memory-size 512 \
        --runtime python3.9 \
        --region $REGION
        
else
    echo "🆕 Creating new Lambda function..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION)
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://lambda-deployment.zip \
        --timeout 900 \
        --memory-size 512 \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --region $REGION
fi

# Cleanup
rm -rf lambda-package lambda-deployment.zip

echo "✅ Simplified Lambda function deployed successfully!"
echo "Function Name: $FUNCTION_NAME"
echo "Region: $REGION"
echo "Package size: $PACKAGE_SIZE"
echo "Memory: 512MB (reduced - no heavy packages needed)"

# Test the function
echo "🧪 Testing simplified function..."
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
        "end_hour": 17,
        "methods": "hikima,maps"
    }' \
    --region $REGION \
    test-output.json

echo "📄 Test output:"
cat test-output.json && echo
rm test-output.json

echo "🎉 Deployment complete! No more import issues!" 