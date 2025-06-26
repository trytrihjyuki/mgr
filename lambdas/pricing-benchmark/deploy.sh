#!/bin/bash

# Ride-Hailing Pricing Benchmark Lambda Deployment Script
# Uses AWS Lambda Container Images for large dependencies

set -e

# Configuration
FUNCTION_NAME="rideshare-pricing-benchmark"
AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="rideshare-pricing-benchmark"
IMAGE_TAG="latest"

echo "🚀 Deploying Ride-Hailing Pricing Benchmark Lambda Function"
echo "📊 Function: $FUNCTION_NAME"
echo "🌍 Region: $AWS_REGION"
echo "🏗️ Account: $AWS_ACCOUNT_ID"

# Step 1: Create ECR repository if it doesn't exist
echo "📦 Setting up ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION

# Get ECR login token
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 2: Build Docker image
echo "🐳 Building Docker image..."
docker build -t $ECR_REPO_NAME:$IMAGE_TAG .

# Step 3: Tag and push image
echo "📤 Pushing image to ECR..."
docker tag $ECR_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG

# Step 4: Create or update Lambda function
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG"

echo "⚡ Creating/updating Lambda function..."

# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $AWS_REGION 2>/dev/null; then
    echo "🔄 Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $IMAGE_URI \
        --region $AWS_REGION
    
    # Update configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout 900 \
        --memory-size 3008 \
        --environment Variables='{S3_BUCKET=magisterka,PYTHONPATH=/var/task}' \
        --region $AWS_REGION
else
    echo "🆕 Creating new function..."
    
    # Create IAM role if it doesn't exist
    ROLE_NAME="lambda-pricing-benchmark-role"
    ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME"
    
    if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
        echo "👤 Creating IAM role..."
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }'
        
        # Attach policies
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
        
        # Wait for role to be available
        echo "⏳ Waiting for IAM role to be ready..."
        sleep 10
    fi
    
    # Create function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --role $ROLE_ARN \
        --code ImageUri=$IMAGE_URI \
        --package-type Image \
        --timeout 900 \
        --memory-size 3008 \
        --environment Variables='{S3_BUCKET=magisterka,PYTHONPATH=/var/task}' \
        --region $AWS_REGION
fi

echo "✅ Deployment completed!"
echo "🔗 Function ARN: arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME"

# Test the function
echo "🧪 Testing function..."
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"test_mode": true}' \
    --region $AWS_REGION \
    test_output.json

echo "📊 Test result:"
cat test_output.json
echo ""

echo "🎉 Lambda function deployment completed successfully!"
echo "📋 Usage examples:"
echo ""
echo "  # Test imports:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"test_mode\": true}' test_output.json"
echo ""
echo "  # Run experiment:"
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