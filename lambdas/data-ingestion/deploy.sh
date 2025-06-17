#!/bin/bash

# AWS Lambda Deployment Script for Data Ingestion
set -e

FUNCTION_NAME="nyc-data-ingestion"
REGION="eu-north-1"
BUCKET_NAME="magisterka"

echo "🚀 Deploying Data Ingestion Lambda Function"

# Create deployment package
echo "📦 Creating deployment package..."
zip -r lambda-deployment.zip lambda_function.py

# Deploy or update Lambda function
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "📝 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $REGION
else
    echo "🆕 Creating new Lambda function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://lambda-deployment.zip \
        --timeout 900 \
        --memory-size 512 \
        --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
        --region $REGION
fi

# Update environment variables
echo "🔧 Setting environment variables..."
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
    --region $REGION

# Cleanup
rm lambda-deployment.zip

echo "✅ Lambda function deployed successfully!"
echo "Function Name: $FUNCTION_NAME"
echo "Region: $REGION"

# Test the function
echo "🧪 Testing function with sample data..."
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
    test-output.json

echo "📄 Test output:"
cat test-output.json && echo
rm test-output.json 