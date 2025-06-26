#!/bin/bash

# 🚀 MAIN DEPLOYMENT SCRIPT
# Unified Lambda Deployment Script for Rideshare Experiments
# This is the PRIMARY script for deploying both Lambda functions (data ingestion + experiment runner)
set -e

REGION="eu-north-1"
BUCKET_NAME="magisterka"
DATA_INGESTION_FUNCTION="nyc-data-ingestion"
EXPERIMENT_RUNNER_FUNCTION="rideshare-experiment-runner"

echo "🚀 Deploying Rideshare Experiment Infrastructure"
echo "================================================"

# Check AWS CLI installationq
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI is not installed. Please install it first."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "💡 Install via Homebrew: brew install awscli"
        fi
        exit 1
    fi
    echo "✅ AWS CLI found"
}

# Create IAM role for Lambda functions
create_lambda_role() {
    echo "🔧 Creating IAM role for Lambda functions..."
    
    # Create trust policy for Lambda
    cat > lambda-trust-policy.json << EOF
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

    # Create execution policy for Lambda
    cat > lambda-execution-policy.json << EOF
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

    # Check if role exists
    if aws iam get-role --role-name lambda-execution-role --region $REGION 2>/dev/null; then
        echo "📝 IAM role already exists"
    else
        echo "🆕 Creating new IAM role..."
        aws iam create-role \
            --role-name lambda-execution-role \
            --assume-role-policy-document file://lambda-trust-policy.json \
            --region $REGION
        
        aws iam put-role-policy \
            --role-name lambda-execution-role \
            --policy-name lambda-execution-policy \
            --policy-document file://lambda-execution-policy.json \
            --region $REGION
        
        echo "⏳ Waiting for role to be available..."
        sleep 10
    fi
    
    # Cleanup policy files
    rm lambda-trust-policy.json lambda-execution-policy.json
    echo "✅ IAM role ready"
}

# Deploy data ingestion Lambda
deploy_data_ingestion() {
    echo "📦 Deploying Data Ingestion Lambda..."
    cd lambdas/data-ingestion
    
    # Make deployment script executable
    chmod +x deploy.sh
    
    # Run deployment
    ./deploy.sh
    
    cd ../..
    echo "✅ Data Ingestion Lambda deployed"
}

# Deploy experiment runner Lambda
deploy_experiment_runner() {
    echo "📦 Deploying Experiment Runner Lambda..."
    cd lambdas/experiment-runner
    
    # Make deployment script executable
    chmod +x deploy.sh
    
    # Run deployment
    ./deploy.sh
    
    cd ../..
    echo "✅ Experiment Runner Lambda deployed"
}

# Test the deployment
test_deployment() {
    echo "🧪 Testing Lambda deployment..."
    
    # Test data ingestion
    echo "Testing data ingestion Lambda..."
    aws lambda invoke \
        --function-name $DATA_INGESTION_FUNCTION \
        --payload '{
            "action": "download_single",
            "vehicle_type": "green",
            "year": 2019,
            "month": 3,
            "limit": 10
        }' \
        --cli-binary-format raw-in-base64-out \
        --region $REGION \
        test-ingestion-output.json
    
    echo "Data ingestion test result:"
    cat test-ingestion-output.json && echo
    rm test-ingestion-output.json
    
    # Wait a moment for data to be available
    echo "⏳ Waiting for data to be ingested..."
    sleep 30
    
    # Test experiment runner
    echo "Testing experiment runner Lambda..."
    aws lambda invoke \
        --function-name $EXPERIMENT_RUNNER_FUNCTION \
        --payload '{
            "vehicle_type": "green",
            "year": 2019,
            "month": 3,
            "day": 1,
            "borough": "Manhattan",
            "scenario": "hikima_replication"
        }' \
        --cli-binary-format raw-in-base64-out \
        --region $REGION \
        test-experiment-output.json
    
    echo "Experiment runner test result:"
    cat test-experiment-output.json && echo
    rm test-experiment-output.json
}

# Create local configuration
setup_local_manager() {
    echo "🏠 Setting up local results manager..."
    
    cd local-manager
    
    # Install dependencies if virtual environment exists
    if [ -d "../venv" ]; then
        echo "📦 Installing local manager dependencies..."
        source ../venv/bin/activate
        pip install -r requirements.txt
        echo "✅ Local manager dependencies installed"
    else
        echo "⚠️  No virtual environment found. Please install requirements manually:"
        echo "   pip install -r local-manager/requirements.txt"
    fi
    
    cd ..
}

# Show usage information
show_usage() {
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    echo "📋 Available Commands:"
    echo ""
    echo "🔄 Data Ingestion:"
    echo "  aws lambda invoke --function-name $DATA_INGESTION_FUNCTION \\"
    echo "    --payload '{\"action\":\"download_single\",\"vehicle_type\":\"green\",\"year\":2019,\"month\":3}' \\"
    echo "    --region $REGION output.json"
    echo ""
    echo "🧪 Run Experiment:"
    echo "  aws lambda invoke --function-name $EXPERIMENT_RUNNER_FUNCTION \\"
    echo "    --payload '{\"vehicle_type\":\"green\",\"year\":2019,\"month\":3,\"simulation_range\":5}' \\"
    echo "    --region $REGION output.json"
    echo ""
    echo "📊 Local Results Management:"
    echo "  python local-manager/results_manager.py list --days 7"
    echo "  python local-manager/results_manager.py report --days 7"
    echo "  python local-manager/results_manager.py show <experiment_id>"
    echo ""
    echo "💾 Bulk Data Download:"
    echo "  aws lambda invoke --function-name $DATA_INGESTION_FUNCTION \\"
    echo "    --payload '{\"action\":\"download_bulk\",\"vehicle_types\":[\"green\",\"yellow\",\"fhv\"],\"year\":2019,\"start_month\":1,\"end_month\":3}' \\"
    echo "    --region $REGION output.json"
    echo ""
    echo "📚 Documentation:"
    echo "  • Lambda functions: lambdas/"
    echo "  • Local manager: local-manager/"
    echo "  • AWS config: aws_config.py"
}

# Main execution
main() {
    case "${1:-all}" in
        "check")
            check_aws_cli
            ;;
        "iam")
            check_aws_cli
            create_lambda_role
            ;;
        "data-ingestion")
            check_aws_cli
            create_lambda_role
            deploy_data_ingestion
            ;;
        "experiment-runner")
            check_aws_cli
            create_lambda_role
            deploy_experiment_runner
            ;;
        "local")
            setup_local_manager
            ;;
        "test")
            test_deployment
            ;;
        "all")
            check_aws_cli
            create_lambda_role
            deploy_data_ingestion
            deploy_experiment_runner
            setup_local_manager
            test_deployment
            show_usage
            ;;
        *)
            echo "Usage: $0 [check|iam|data-ingestion|experiment-runner|local|test|all]"
            echo ""
            echo "Commands:"
            echo "  check            - Check AWS CLI installation"
            echo "  iam              - Create IAM roles"
            echo "  data-ingestion   - Deploy data ingestion Lambda"
            echo "  experiment-runner - Deploy experiment runner Lambda"
            echo "  local            - Setup local results manager"
            echo "  test             - Test deployment"
            echo "  all              - Full deployment (default)"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@" 