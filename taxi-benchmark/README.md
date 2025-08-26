# 🚕 Taxi Pricing Benchmark Framework

A comprehensive, production-ready framework for running large-scale taxi pricing experiments using Docker containers on AWS EC2 with S3 data storage.

## 🌟 Features

- **🚀 High Performance**: Parallel processing with configurable workers
- **☁️ Cloud Native**: Built for AWS (EC2 + S3) with local mode support
- **🐳 Docker-Based**: One container per experiment day for isolation
- **📊 Multiple Methods**: LP (NEW), MinMaxCostFlow, MAPS, LinUCB
- **📈 Comprehensive Monitoring**: Real-time progress tracking and detailed logging
- **🔧 Highly Configurable**: Support for multiple boroughs, time windows, and parameters
- **📦 S3 Integration**: Automatic data loading and result storage

## 📋 Table of Contents

1. [Architecture](#-architecture)
2. [Installation](#-installation)
3. [Quick Start](#-quick-start)
4. [Configuration](#-configuration)
5. [Running Experiments](#-running-experiments)
6. [AWS Setup](#-aws-setup)
7. [Methods Overview](#-methods-overview)
8. [Results Analysis](#-results-analysis)
9. [Monitoring](#-monitoring)
10. [Troubleshooting](#-troubleshooting)

## 🏗 Architecture

The framework follows a clean, modular architecture:

```
taxi-benchmark/
├── src/
│   ├── core/           # Configuration, types, logging
│   ├── data/           # S3 data loading and processing
│   ├── methods/        # Pricing method implementations
│   │   ├── lp.py       # NEW: Linear Programming method
│   │   ├── minmax_costflow.py
│   │   ├── maps.py
│   │   └── linucb.py
│   ├── experiments/    # Experiment runner and workers
│   └── utils/          # Helper utilities
├── scripts/
│   ├── local/          # Local development scripts
│   ├── ec2/            # EC2 deployment scripts
│   └── deploy/         # Docker and ECR scripts
├── notebooks/          # Analysis notebooks
├── configs/            # Experiment configurations
└── run_experiment.py   # Main CLI entry point
```

## 💻 Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- AWS CLI configured with credentials
- Access to S3 buckets with taxi data

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/taxi-benchmark.git
cd taxi-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region (eu-north-1)

# Create .env file
cat > .env << EOF
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=eu-north-1
S3_BASE=magisterka
S3_RESULTS_BUCKET=taxi-benchmark
SUBNET_ID=your_subnet_id
SECURITY_GROUP_IDS=your_security_group
KEY_NAME=your_ec2_key
IAM_INSTANCE_PROFILE=PricingExperimentRole
EOF
```

## 🚀 Quick Start

### Local Testing

```bash
# Test with a small experiment locally
python run_experiment.py \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan \
    --methods LP \
    --num-iter 10 \
    --start-hour 10 \
    --end-hour 11 \
    --time-delta 30 \
    --local-mode

# Dry run to validate configuration
python run_experiment.py \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan Brooklyn \
    --methods LP MinMaxCostFlow MAPS LinUCB \
    --dry-run
```

### Docker Local

```bash
# Build Docker image
docker build -t taxi-benchmark:latest .

# Run experiment in Docker
docker run -it \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_REGION=eu-north-1 \
    taxi-benchmark:latest \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan \
    --methods LP MAPS
```

## ⚙️ Configuration

### Command Line Arguments

| Argument | Description | Example | Required |
|----------|-------------|---------|----------|
| `--processing-date` | Date to process (YYYY-MM-DD) | `2019-10-06` | ✅ |
| `--vehicle-type` | Vehicle type | `green`, `yellow`, `fhv` | ✅ |
| `--boroughs` | Boroughs to process | `Manhattan Brooklyn` | ✅ |
| `--methods` | Pricing methods | `LP MinMaxCostFlow MAPS LinUCB` | ✅ |
| `--eval` | Acceptance function | `PL`, `Sigmoid` | ❌ (default: PL) |
| `--num-iter` | Monte Carlo iterations | `100` | ❌ (default: 100) |
| `--start-hour` | Start hour (0-23) | `6` | ❌ (default: 0) |
| `--end-hour` | End hour (0-23) | `22` | ❌ (default: 23) |
| `--time-delta` | Window size (minutes) | `30` | ❌ (default: 5) |
| `--num-workers` | Parallel workers | `8` | ❌ (default: 4) |

### Example Configurations

```bash
# Full day experiment with all methods
python run_experiment.py \
    --processing-date 2019-10-01 \
    --vehicle-type green \
    --boroughs Manhattan Brooklyn Queens Bronx \
    --methods LP MinMaxCostFlow MAPS LinUCB \
    --eval Sigmoid \
    --num-iter 100 \
    --time-delta 60

# Rush hour analysis
python run_experiment.py \
    --processing-date 2019-10-15 \
    --vehicle-type yellow \
    --boroughs Manhattan \
    --methods LP MAPS \
    --start-hour 7 \
    --end-hour 10 \
    --time-delta 15 \
    --num-iter 200

# Weekend comparison
python run_experiment.py \
    --processing-date 2019-10-05 \
    --vehicle-type green \
    --boroughs Brooklyn Queens \
    --methods MinMaxCostFlow LinUCB \
    --start-hour 10 \
    --end-hour 22 \
    --time-delta 30
```

## 🌐 AWS Setup

### 1. ECR Setup (Docker Registry)

```bash
# Create ECR repository
aws ecr create-repository --repository-name taxi-benchmark

# Get login token
aws ecr get-login-password --region eu-north-1 | \
    docker login --username AWS --password-stdin \
    YOUR_ACCOUNT.dkr.ecr.eu-north-1.amazonaws.com

# Build and push image
docker build -t taxi-benchmark:latest .
docker tag taxi-benchmark:latest \
    YOUR_ACCOUNT.dkr.ecr.eu-north-1.amazonaws.com/taxi-benchmark:latest
docker push YOUR_ACCOUNT.dkr.ecr.eu-north-1.amazonaws.com/taxi-benchmark:latest
```

### 2. EC2 Instance Setup

```bash
# Launch EC2 instance (t3.xlarge or larger recommended)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.xlarge \
    --key-name your-key \
    --subnet-id subnet-xxx \
    --security-group-ids sg-xxx \
    --iam-instance-profile Name=PricingExperimentRole \
    --user-data file://scripts/ec2/user-data.sh

# SSH to instance
ssh -i your-key.pem ec2-user@YOUR_INSTANCE_IP

# On EC2: Pull and run Docker image
$(aws ecr get-login --no-include-email --region eu-north-1)
docker pull YOUR_ACCOUNT.dkr.ecr.eu-north-1.amazonaws.com/taxi-benchmark:latest

# Run experiment in tmux for persistence
tmux new -s experiment
docker run -d \
    --name taxi-exp-$(date +%Y%m%d) \
    YOUR_ACCOUNT.dkr.ecr.eu-north-1.amazonaws.com/taxi-benchmark:latest \
    --processing-date 2019-10-01 \
    --vehicle-type green \
    --boroughs Manhattan Brooklyn \
    --methods LP MinMaxCostFlow MAPS LinUCB \
    --num-iter 100 \
    --num-workers 8

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t experiment
```

### 3. S3 Data Structure

```
s3://magisterka/
├── datasets/
│   ├── area_information.csv
│   ├── green/
│   │   └── year=2019/month=10/
│   │       └── green_tripdata_2019-10.parquet
│   ├── yellow/
│   │   └── year=2019/month=10/
│   │       └── yellow_tripdata_2019-10.parquet
│   └── fhv/
│       └── year=2019/month=10/
│           └── fhv_tripdata_2019-10.parquet
└── models/
    └── linucb/
        └── green_Manhattan_201910/
            └── trained_model.pkl

s3://taxi-benchmark/
└── experiments/
    └── run_20191001_green_Man_Bro_LP_MAPS_20231025_143022/
        ├── results_batch_001.parquet
        ├── results_batch_002.parquet
        └── experiment_summary.json
```

## 📊 Methods Overview

### 1. **LP Method (NEW)**
- Based on Gupta & Nagarajan's linearization of Myerson's optimal mechanism
- Discretizes price space and uses probing variables
- Guaranteed approximation ratio
- Best for: Theoretical optimality

### 2. **MinMaxCostFlow**
- Original method from Hikima et al. paper
- Uses minimum cost flow algorithms
- 3-approximation guarantee
- Best for: Balanced performance

### 3. **MAPS**
- Market-based Adaptive Pricing System
- Greedy allocation with dynamic pricing
- Fast computation
- Best for: Real-time systems

### 4. **LinUCB**
- Contextual bandit approach
- Learns from historical data
- Requires pre-trained models
- Best for: Adaptive learning scenarios

## 📈 Results Analysis

### Reading Results

```python
import pandas as pd
import json
import boto3

# Load experiment summary
s3 = boto3.client('s3')
exp_id = "run_20191001_green_Man_Bro_LP_MAPS_20231025_143022"

# Get summary
obj = s3.get_object(
    Bucket='taxi-benchmark',
    Key=f'experiments/{exp_id}/experiment_summary.json'
)
summary = json.loads(obj['Body'].read())

# Load detailed results
results_df = pd.read_parquet(
    f's3://taxi-benchmark/experiments/{exp_id}/results_batch_001.parquet'
)

# Analyze by method
method_comparison = results_df.groupby('method').agg({
    'profit': ['mean', 'std'],
    'matching_rate': 'mean',
    'acceptance_rate': 'mean',
    'computation_time': 'mean'
})
print(method_comparison)
```

### Visualization Notebook

See `notebooks/analyze_results.ipynb` for comprehensive analysis tools.

## 📊 Monitoring

### Real-time Logs

```bash
# View Docker logs
docker logs -f taxi-exp-20231025

# Monitor system resources
docker stats taxi-exp-20231025

# Check S3 uploads
aws s3 ls s3://taxi-benchmark/experiments/run_xxx/ --recursive
```

### Progress Tracking

The framework provides detailed progress information:

```
2023-10-25 14:30:15 - INFO - Starting experiment run_20191001_xxx
2023-10-25 14:30:16 - INFO - Total scenarios to process: 2880
2023-10-25 14:30:16 - INFO - Number of Time Windows: 288
2023-10-25 14:30:20 - INFO - Progress: 100/2880 (3.5%) - Rate: 2.5 scenarios/s - ETA: 1112s
```

## 🐛 Troubleshooting

### Common Issues

#### 1. AWS Credentials Error
```bash
# Fix: Ensure credentials are configured
aws configure
# Or use environment variables
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
```

#### 2. Out of Memory
```bash
# Fix: Reduce batch size or use larger instance
--num-workers 2  # Reduce parallel workers
```

#### 3. S3 Access Denied
```bash
# Fix: Check IAM role permissions
# Required: s3:GetObject, s3:PutObject, s3:ListBucket
```

#### 4. Docker Build Fails
```bash
# Fix: Clear cache and rebuild
docker system prune -a
docker build --no-cache -t taxi-benchmark:latest .
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{taxi_benchmark_2023,
  title = {Taxi Pricing Benchmark Framework},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/taxi-benchmark}
}
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact:
- Email: your.email@example.com
- GitHub: @yourusername

---

**Happy Experimenting! 🚕💰**