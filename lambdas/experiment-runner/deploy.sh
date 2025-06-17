#!/bin/bash

# AWS Lambda Deployment Script for Experiment Runner
set -e

FUNCTION_NAME="rideshare-experiment-runner"
REGION="eu-north-1"
BUCKET_NAME="magisterka"

echo "🚀 Deploying Experiment Runner Lambda Function"

# Create deployment package with dependencies
echo "📦 Creating deployment package with dependencies..."

# Create temporary directory for packaging
mkdir -p lambda-package

# Install dependencies to package directory
pip install -r requirements.txt -t lambda-package/

# Copy lambda function
cp lambda_function.py lambda-package/

# Create zip package
cd lambda-package
zip -r ../lambda-deployment.zip .
cd ..

# Deploy or update Lambda function
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "📝 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $REGION
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
        --region $REGION
fi

# Update environment variables
echo "🔧 Setting environment variables..."
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
    --timeout 900 \
    --memory-size 1024 \
    --region $REGION

# Cleanup
rm -rf lambda-package lambda-deployment.zip

echo "✅ Lambda function deployed successfully!"
echo "Function Name: $FUNCTION_NAME"
echo "Region: $REGION"

# Test the function
echo "🧪 Testing function with sample experiment..."
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{
        "vehicle_type": "green",
        "year": 2019,
        "month": 3,
        "place": "Manhattan",
        "simulation_range": 2,
        "acceptance_function": "PL"
    }' \
    --region $REGION \
    test-output.json

echo "📄 Test output:"
cat test-output.json && echo
rm test-output.json 