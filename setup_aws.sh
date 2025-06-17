#!/bin/bash

# AWS Setup Script for Bipartite Matching Optimization Project
# This script helps initialize the AWS environment and upload initial data

set -e  # Exit on any error

echo "🚀 AWS Bipartite Matching Setup Script"
echo "======================================="

# Check if AWS credentials are set
check_aws_credentials() {
    echo "🔍 Checking AWS credentials..."
    
    if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
        echo "⚠️  AWS credentials not found in environment variables"
        echo "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        echo ""
        echo "export AWS_ACCESS_KEY_ID=\"your_access_key\""
        echo "export AWS_SECRET_ACCESS_KEY=\"your_secret_key\""
        echo "export AWS_REGION=\"us-east-1\""
        exit 1
    fi
    
    echo "✅ AWS credentials found"
}

# Install Python dependencies
install_dependencies() {
    echo "📦 Installing Python dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    pip install -r requirements_aws.txt
    echo "✅ Dependencies installed"
}

# Test AWS connection
test_aws_connection() {
    echo "🔗 Testing AWS connection..."
    
    python3 -c "from aws_config import AWSConfig; from aws_s3_manager import S3DataManager; s3 = S3DataManager(); print('✅ AWS connection successful')"
}

# Upload sample data if available
upload_sample_data() {
    echo "📤 Checking for sample data to upload..."
    
    # Check for rideshare data
    if [[ -d "Rideshare_experiment/data" ]]; then
        data_files=$(find Rideshare_experiment/data -name "*.csv" -o -name "*.parquet" | wc -l)
        if [[ $data_files -gt 0 ]]; then
            echo "📊 Found $data_files rideshare data files"
            echo "Uploading to S3..."
            python3 aws_deploy.py upload-datasets --data-dir Rideshare_experiment/data
        else
            echo "📂 Rideshare data directory exists but no data files found"
        fi
    else
        echo "📂 No rideshare data directory found (Rideshare_experiment/data)"
    fi
    
    # Check for crowdsourcing data
    if [[ -f "Crowd_sourcing_experiment/work/trec-rf10-data.csv" ]]; then
        echo "📊 Found crowdsourcing data file"
        echo "Uploading to S3..."
        python3 aws_deploy.py upload-crowdsourcing
    else
        echo "📂 No crowdsourcing data found (Crowd_sourcing_experiment/work/trec-rf10-data.csv)"
    fi
}

# Run a simple test experiment
run_test_experiment() {
    echo "🧪 Running test experiment..."
    
    echo "Running rideshare simulation..."
    python3 aws_experiment_runner.py rideshare \
        --vehicle-type green \
        --year 2019 \
        --month 3 \
        --place Manhattan \
        --simulation-range 5 \
        --acceptance-function PL
    
    echo "✅ Test experiment completed"
}

# Show next steps
show_next_steps() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Run experiments:"
    echo "   python3 aws_experiment_runner.py rideshare --vehicle-type green --year 2019 --month 3"
    echo "   python3 aws_experiment_runner.py crowdsourcing --phi 0.8 --psi 0.6"
    echo ""
    echo "2. Manage data:"
    echo "   python3 aws_deploy.py list-data"
    echo "   python3 aws_deploy.py upload-datasets --data-dir /path/to/data"
    echo ""
    echo "3. Create analysis dashboard:"
    echo "   python3 aws_deploy.py create-dashboard"
    echo ""
    echo "4. Docker deployment:"
    echo "   docker build -t bipartite-matching ."
    echo "   docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY bipartite-matching"
    echo ""
    echo "📚 See README_COMPLETE.md for full documentation"
}

# Main execution
main() {
    case "${1:-full}" in
        "check")
            check_aws_credentials
            ;;
        "install")
            install_dependencies
            ;;
        "test")
            check_aws_credentials
            test_aws_connection
            ;;
        "upload")
            check_aws_credentials
            upload_sample_data
            ;;
        "experiment")
            check_aws_credentials
            run_test_experiment
            ;;
        "full")
            check_aws_credentials
            install_dependencies
            test_aws_connection
            upload_sample_data
            run_test_experiment
            show_next_steps
            ;;
        *)
            echo "Usage: $0 [check|install|test|upload|experiment|full]"
            echo ""
            echo "Commands:"
            echo "  check      - Check AWS credentials"
            echo "  install    - Install Python dependencies"
            echo "  test       - Test AWS connection"
            echo "  upload     - Upload sample data to S3"
            echo "  experiment - Run test experiment"
            echo "  full       - Run complete setup (default)"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@"