#!/bin/bash

# Simplified Lambda Deployment Script
set -e

REGION="eu-north-1"
BUCKET_NAME="magisterka"
ACCOUNT_ID="167872550673"  # Your AWS account ID

echo "🚀 Simple Lambda Deployment"
echo "=========================="

# Function to create IAM role
create_iam_role() {
    echo "🔧 Creating IAM role..."
    
    cat > /tmp/trust-policy.json << 'EOF'
{
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
}
EOF

    cat > /tmp/execution-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF
    
    # Create role if it doesn't exist
    if ! /usr/local/bin/aws iam get-role --role-name lambda-execution-role --region $REGION >/dev/null 2>&1; then
        echo "Creating IAM role..."
        /usr/local/bin/aws iam create-role \
            --role-name lambda-execution-role \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --region $REGION
        
        /usr/local/bin/aws iam put-role-policy \
            --role-name lambda-execution-role \
            --policy-name lambda-execution-policy \
            --policy-document file:///tmp/execution-policy.json \
            --region $REGION
        
        echo "Waiting for role to propagate..."
        sleep 15
    else
        echo "IAM role already exists"
    fi
    
    rm -f /tmp/trust-policy.json /tmp/execution-policy.json
    echo "✅ IAM role ready"
}

# Function to deploy data ingestion Lambda
deploy_data_ingestion() {
    echo "📦 Deploying Data Ingestion Lambda..."
    cd lambdas/data-ingestion
    
    # Create package with dependencies
    rm -rf package
    mkdir package
    pip install -r requirements.txt -t package/ -q
    cp lambda_function.py package/
    cd package
    zip -q -r ../lambda-deployment.zip .
    cd ..
    
    # Check if function exists
    if /usr/local/bin/aws lambda get-function --function-name nyc-data-ingestion --region $REGION >/dev/null 2>&1; then
        echo "Updating existing function..."
        /usr/local/bin/aws lambda update-function-code \
            --function-name nyc-data-ingestion \
            --zip-file fileb://lambda-deployment.zip \
            --region $REGION
    else
        echo "Creating new function..."
        /usr/local/bin/aws lambda create-function \
            --function-name nyc-data-ingestion \
            --runtime python3.9 \
            --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
            --handler lambda_function.lambda_handler \
            --zip-file fileb://lambda-deployment.zip \
            --timeout 900 \
            --memory-size 512 \
            --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
            --region $REGION
    fi
    
    rm -rf package lambda-deployment.zip
    cd ../..
    echo "✅ Data Ingestion Lambda deployed"
}

# Function to deploy experiment runner Lambda
deploy_experiment_runner() {
    echo "📦 Deploying Experiment Runner Lambda..."
    cd lambdas/experiment-runner
    
    # Create package with dependencies
    rm -rf package
    mkdir package
    pip install -r requirements.txt -t package/ -q
    cp lambda_function.py package/
    cd package
    zip -q -r ../lambda-deployment.zip .
    cd ..
    
    # Check if function exists
    if /usr/local/bin/aws lambda get-function --function-name rideshare-experiment-runner --region $REGION >/dev/null 2>&1; then
        echo "Updating existing function..."
        /usr/local/bin/aws lambda update-function-code \
            --function-name rideshare-experiment-runner \
            --zip-file fileb://lambda-deployment.zip \
            --region $REGION
    else
        echo "Creating new function..."
        /usr/local/bin/aws lambda create-function \
            --function-name rideshare-experiment-runner \
            --runtime python3.9 \
            --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
            --handler lambda_function.lambda_handler \
            --zip-file fileb://lambda-deployment.zip \
            --timeout 900 \
            --memory-size 1024 \
            --environment Variables="{S3_BUCKET=$BUCKET_NAME}" \
            --region $REGION
    fi
    
    rm -rf package lambda-deployment.zip
    cd ../..
    echo "✅ Experiment Runner Lambda deployed"
}

# Function to test deployment
test_deployment() {
    echo "🧪 Testing deployment..."
    
    # Test data ingestion
    echo "Testing data ingestion..."
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload '{"action":"download_single","vehicle_type":"green","year":2019,"month":3,"limit":10}' \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        test-output.json
    
    echo "Response:"
    cat test-output.json
    echo ""
    rm test-output.json
}

# Main execution
case "${1:-all}" in
    "iam")
        create_iam_role
        ;;
    "data")
        create_iam_role
        deploy_data_ingestion
        ;;
    "experiment")
        create_iam_role
        deploy_experiment_runner
        ;;
    "test")
        test_deployment
        ;;
    "all")
        create_iam_role
        deploy_data_ingestion
        deploy_experiment_runner
        test_deployment
        echo ""
        echo "🎉 Deployment completed!"
        echo ""
        echo "📋 Next steps:"
        echo "1. Download data:"
        echo "   /usr/local/bin/aws lambda invoke --function-name nyc-data-ingestion --payload '{\"action\":\"download_single\",\"vehicle_type\":\"green\",\"year\":2019,\"month\":3,\"limit\":1000}' --region eu-north-1 --cli-binary-format raw-in-base64-out response.json"
        echo ""
        echo "2. Run experiment:"
        echo "   /usr/local/bin/aws lambda invoke --function-name rideshare-experiment-runner --payload '{\"vehicle_type\":\"green\",\"year\":2019,\"month\":3}' --region eu-north-1 --cli-binary-format raw-in-base64-out response.json"
        echo ""
        echo "3. Check results:"
        echo "   python local-manager/results_manager.py list --days 1"
        ;;
    *)
        echo "Usage: $0 [iam|data|experiment|test|all]"
        exit 1
        ;;
esac 