# Ride-Hailing Pricing Benchmark Makefile

.PHONY: deploy upload-config test-lambda run-hikima run-comprehensive run-extended clean help

# Default target
help:
	@echo "🚗 Ride-Hailing Pricing Benchmark Framework"
	@echo ""
	@echo "Available targets:"
	@echo "  deploy          - Deploy Lambda function with container support"
	@echo "  upload-config   - Upload configuration to S3"
	@echo "  test-lambda     - Test Lambda function imports"
	@echo "  run-hikima      - Run Hikima replication experiment"
	@echo "  run-comprehensive - Run comprehensive benchmark (all 4 methods)"
	@echo "  run-extended    - Run extended multi-day analysis"
	@echo "  clean           - Clean up temporary files"
	@echo ""
	@echo "Examples:"
	@echo "  make deploy           # Deploy the system"
	@echo "  make run-hikima       # Replicate Hikima experiment"
	@echo "  make run-comprehensive # Test all 4 methods"

# Deploy Lambda function
deploy:
	@echo "🚀 Deploying Lambda function..."
	cd lambdas/pricing-benchmark && ./deploy.sh

# Upload configuration to S3
upload-config:
	@echo "📋 Uploading configuration to S3..."
	aws s3 cp configs/experiment_config.json s3://magisterka/configs/

# Test Lambda function
test-lambda:
	@echo "🧪 Testing Lambda function imports..."
	aws lambda invoke \
		--function-name rideshare-pricing-benchmark \
		--payload '{"test_mode": true}' \
		test_output.json
	@echo "📊 Test result:"
	@cat test_output.json | jq .
	@rm test_output.json

# Run Hikima replication experiment
run-hikima:
	@echo "🔬 Running Hikima replication experiment..."
	python run_experiment.py \
		--year=2019 --month=10 --days=1,6 \
		--func=PL,Sigmoid \
		--methods=MinMaxCostFlow,MAPS,LinUCB \
		--scenario=hikima_replication

# Run comprehensive benchmark (all 4 methods)
run-comprehensive:
	@echo "📊 Running comprehensive benchmark..."
	python run_experiment.py \
		--year=2019 --month=10 --days=1 \
		--func=PL \
		--methods=MinMaxCostFlow,MAPS,LinUCB,LP \
		--scenario=comprehensive

# Run extended multi-day analysis
run-extended:
	@echo "📈 Running extended multi-day analysis..."
	python run_experiment.py \
		--year=2019 --month=10 --days=1,2,3,4,5,6,7 \
		--func=PL,Sigmoid \
		--methods=LP,MAPS,LinUCB \
		--scenario=scalability

# Custom experiment examples
run-multi-month:
	@echo "🗓️ Running multi-month experiment..."
	python run_experiment.py \
		--year=2019 --months=3,4,5 --days=1,15 \
		--func=PL \
		--methods=LP,MAPS

run-full-day:
	@echo "🌅 Running full-day analysis..."
	python run_experiment.py \
		--year=2019 --month=10 --days=1 \
		--func=PL --methods=LP,MAPS \
		--time-range=full_day

# Setup and installation
setup:
	@echo "⚙️ Setting up development environment..."
	pip install boto3 pandas numpy networkx pulp geopy scipy pytz
	chmod +x lambdas/pricing-benchmark/deploy.sh

# Check AWS configuration
check-aws:
	@echo "🔍 Checking AWS configuration..."
	@aws sts get-caller-identity
	@aws s3 ls s3://magisterka/ 2>/dev/null || echo "❌ S3 bucket 'magisterka' not accessible"

# Full deployment pipeline
deploy-all: upload-config deploy test-lambda
	@echo "🎉 Full deployment completed!"

# Clean up temporary files
clean:
	@echo "🧹 Cleaning up..."
	rm -f test_output.json
	rm -f experiment_output.json
	rm -f *.log
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Development helpers
lint:
	@echo "🔍 Running linter..."
	pylint run_experiment.py lambdas/pricing-benchmark/lambda_function.py || true

format:
	@echo "🎨 Formatting code..."
	black run_experiment.py lambdas/pricing-benchmark/lambda_function.py

# Documentation
docs:
	@echo "📚 Opening documentation..."
	@echo "Main README: README.md"
	@echo "Configuration: configs/experiment_config.json"
	@echo "Lambda function: lambdas/pricing-benchmark/lambda_function.py" 