#!/bin/bash

# Simplified ZIP-based deployment for Lambda
# Avoids Docker platform issues

set -e

FUNCTION_NAME="rideshare-pricing-benchmark"
AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🚀 Deploying Lambda function with ZIP packaging..."
echo "📊 Function: $FUNCTION_NAME"
echo "🌍 Region: $AWS_REGION"

# Create deployment package
echo "📦 Creating deployment package..."
rm -f lambda-deployment.zip
zip -q lambda-deployment.zip lambda_function.py requirements.txt

# Create or update Lambda function
if aws lambda get-function --function-name $FUNCTION_NAME --region $AWS_REGION 2>/dev/null; then
    echo "🔄 Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-deployment.zip \
        --region $AWS_REGION
    
    # Update configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout 900 \
        --memory-size 3008 \
        --environment Variables='{S3_BUCKET=magisterka}' \
        --region $AWS_REGION
else
    echo "🆕 Creating new function..."
    
    # Create IAM role if needed
    ROLE_NAME="lambda-pricing-benchmark-role"
    ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME"
    
    if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
        echo "👤 Creating IAM role..."
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }'
        
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
        
        sleep 10
    fi
    
    # Create function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --role $ROLE_ARN \
        --code ZipFile=fileb://lambda-deployment.zip \
        --runtime python3.9 \
        --handler lambda_function.lambda_handler \
        --timeout 900 \
        --memory-size 3008 \
        --environment Variables='{S3_BUCKET=magisterka}' \
        --region $AWS_REGION
    
    # Install dependencies via layers
    echo "📚 Installing dependencies..."
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --layers arn:aws:lambda:eu-north-1:336392948345:layer:AWSSDKPandas-Python39:7 \
        --region $AWS_REGION
fi

# Test the function
echo "🧪 Testing function..."
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"test_mode": true}' \
    --region $AWS_REGION \
    test_output.json

echo "📊 Test result:"
cat test_output.json | jq . 2>/dev/null || cat test_output.json

echo "✅ ZIP deployment completed!"
rm -f lambda-deployment.zip test_output.json 