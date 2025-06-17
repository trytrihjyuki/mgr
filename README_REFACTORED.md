# Rideshare Bipartite Matching Experiments - Cloud Architecture

## Overview

This project implements a cloud-native architecture for running bipartite matching experiments on NYC taxi and rideshare data. The system has been refactored to focus exclusively on rideshare experiments using a serverless Lambda-based approach.

### 🏗️ Architecture

```
NYC Open Data API → Data Ingestion Lambda → S3 Storage → Experiment Runner Lambda → Results
                                                     ↓
                                             Local Results Manager
```

## 🚀 Quick Start

### Prerequisites

1. **AWS CLI**: Install and configure with your credentials
   ```bash
   # macOS
   brew install awscli
   
   # Configure credentials
   aws configure
   ```

2. **Python Environment**: Set up virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements_aws.txt
   ```

3. **Environment Variables**: Ensure `.env` file contains:
   ```
   AWS_ACCESS_KEY_ID="your_access_key"
   AWS_SECRET_ACCESS_KEY="your_secret_key"
   AWS_REGION="eu-north-1"
   ```

### 🚀 Deploy Infrastructure

```bash
# Make deployment script executable
chmod +x deploy_lambdas.sh

# Full deployment (recommended)
./deploy_lambdas.sh

# Or deploy components individually
./deploy_lambdas.sh iam              # Create IAM roles
./deploy_lambdas.sh data-ingestion   # Deploy data ingestion Lambda
./deploy_lambdas.sh experiment-runner # Deploy experiment runner Lambda
./deploy_lambdas.sh local            # Setup local manager
./deploy_lambdas.sh test             # Test deployment
```

## 📦 Components

### 1. Data Ingestion Lambda (`lambdas/data-ingestion/`)

Downloads NYC taxi data directly from NYC Open Data API to S3.

**Supported Data Types:**
- 🟢 Green Taxi (`green`)
- 🟡 Yellow Taxi (`yellow`)  
- 🚗 For-Hire Vehicles (`fhv`)

**Usage:**
```bash
# Download single dataset
aws lambda invoke --function-name nyc-data-ingestion \
  --payload '{
    "action": "download_single",
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "limit": 10000
  }' \
  output.json

# Bulk download multiple datasets
aws lambda invoke --function-name nyc-data-ingestion \
  --payload '{
    "action": "download_bulk",
    "vehicle_types": ["green", "yellow", "fhv"],
    "year": 2019,
    "start_month": 1,
    "end_month": 6
  }' \
  output.json
```

### 2. Experiment Runner Lambda (`lambdas/experiment-runner/`)

Runs bipartite matching experiments on rideshare data stored in S3.

**Features:**
- Piecewise Linear (PL) and Sigmoid acceptance functions
- Configurable simulation scenarios
- Automatic results upload to S3
- Error handling and logging

**Usage:**
```bash
aws lambda invoke --function-name rideshare-experiment-runner \
  --payload '{
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "place": "Manhattan",
    "simulation_range": 5,
    "acceptance_function": "PL"
  }' \
  output.json
```

### 3. Local Results Manager (`local-manager/`)

Python-based tool for loading, analyzing, and visualizing experiment results.

**Features:**
- 📊 Load and analyze results from S3
- 📈 Generate comparison reports
- 🎨 Create visualizations
- 💾 Local caching for performance

**Usage:**
```bash
# List recent experiments
python local-manager/results_manager.py list --days 7

# Generate full report
python local-manager/results_manager.py report --days 7 --output report.txt

# Show specific experiment
python local-manager/results_manager.py show rideshare_green_2019_03_20241217_143022

# Compare multiple experiments
python local-manager/results_manager.py compare exp1 exp2 exp3
```

## 📋 Experiment Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `vehicle_type` | Type of taxi data | `green`, `yellow`, `fhv` | `green` |
| `year` | Year of data | 2015-2024 | 2019 |
| `month` | Month of data | 1-12 | 3 |
| `place` | Location (metadata) | Any string | `Manhattan` |
| `simulation_range` | Number of scenarios | 1-10 | 5 |
| `acceptance_function` | Acceptance model | `PL`, `Sigmoid` | `PL` |

## 🗂️ S3 Data Structure

```
s3://magisterka/
├── datasets/
│   ├── green/
│   │   └── year=2019/month=03/
│   │       └── green_tripdata_2019-03.csv
│   ├── yellow/
│   │   └── year=2019/month=03/
│   │       └── yellow_tripdata_2019-03.csv
│   └── fhv/
│       └── year=2020/month=01/
│           └── fhv_tripdata_2020-01.csv
└── experiments/
    ├── results/
    │   └── rideshare/
    │       └── rideshare_green_2019_03_20241217_143022_results.json
    ├── analysis/
    └── logs/
```

## 🔧 Configuration

### AWS Configuration (`aws_config.py`)
- S3 bucket settings
- Key naming conventions
- Region configuration

### Environment Variables (`.env`)
- AWS credentials
- Regional settings

## 🧪 Running Experiments

### 1. Download Data
```bash
# Download sample data for testing
aws lambda invoke --function-name nyc-data-ingestion \
  --payload '{
    "action": "download_single",
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "limit": 1000
  }' \
  data-output.json
```

### 2. Run Experiment
```bash
# Run experiment on downloaded data
aws lambda invoke --function-name rideshare-experiment-runner \
  --payload '{
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "simulation_range": 3,
    "acceptance_function": "PL"
  }' \
  experiment-output.json
```

### 3. Analyze Results
```bash
# Generate analysis report
python local-manager/results_manager.py report --days 1
```

## 📊 Example Workflow

```bash
# 1. Deploy infrastructure
./deploy_lambdas.sh

# 2. Download multiple datasets
aws lambda invoke --function-name nyc-data-ingestion \
  --payload '{
    "action": "download_bulk",
    "vehicle_types": ["green", "yellow"],
    "year": 2019,
    "start_month": 1,
    "end_month": 3,
    "limit": 5000
  }' output.json

# 3. Run experiments with different parameters
for vehicle in green yellow; do
  for acceptance in PL Sigmoid; do
    aws lambda invoke --function-name rideshare-experiment-runner \
      --payload "{
        \"vehicle_type\": \"$vehicle\",
        \"year\": 2019,
        \"month\": 3,
        \"acceptance_function\": \"$acceptance\",
        \"simulation_range\": 5
      }" \
      experiment_${vehicle}_${acceptance}.json
  done
done

# 4. Generate comprehensive analysis
python local-manager/results_manager.py report --days 1 --output analysis_report.txt
```

## 🛠️ Development

### Local Testing

```bash
# Test data ingestion locally
cd lambdas/data-ingestion
python lambda_function.py

# Test experiment runner locally
cd lambdas/experiment-runner
python lambda_function.py
```

### Adding New Features

1. **New Data Sources**: Modify `NYCDataIngester` in data ingestion Lambda
2. **New Algorithms**: Extend `BipartiteMatchingExperiment` in experiment runner
3. **New Analysis**: Add methods to `ExperimentResultsManager` in local manager

## 🔍 Monitoring and Debugging

### CloudWatch Logs
```bash
# View data ingestion logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/nyc-data-ingestion"

# View experiment runner logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/rideshare-experiment-runner"
```

### S3 Data Verification
```bash
# List all datasets
python aws_s3_manager.py list-datasets

# Check experiment results
python aws_s3_manager.py list-experiment-results rideshare
```

## 🔒 Security

- Lambda functions use least-privilege IAM roles
- S3 bucket access is restricted to necessary operations
- Environment variables for sensitive configuration

## 💡 Tips

- Use `limit` parameter in data ingestion for testing
- Start with small `simulation_range` values
- Cache results locally for repeated analysis
- Monitor Lambda execution time and memory usage

## 📚 Related Files

- `aws_config.py` - AWS configuration
- `aws_s3_manager.py` - S3 operations (legacy, used by local manager)
- `deploy_lambdas.sh` - Infrastructure deployment
- `requirements_aws.txt` - Python dependencies

## 🆘 Troubleshooting

### Common Issues

1. **Lambda Timeout**: Increase timeout or reduce data size
2. **Memory Issues**: Increase Lambda memory allocation
3. **Permissions**: Check IAM roles and S3 bucket policies
4. **Data Not Found**: Verify data ingestion completed successfully

### Support Commands

```bash
# Check AWS configuration
aws sts get-caller-identity

# Verify S3 bucket access
aws s3 ls s3://magisterka/

# Test Lambda functions
./deploy_lambdas.sh test
``` 