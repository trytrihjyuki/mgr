# Ride-Hailing Pricing Experiment System

A comprehensive experimental framework for evaluating dynamic pricing algorithms in ride-hailing platforms using real NYC taxi data.

## 🎯 Overview

This system implements and compares four pricing algorithms:

- **MinMaxCostFlow** - Min-cost flow algorithm with capacity scaling
- **MAPS** - Area-based pricing with bipartite matching  
- **LinUCB** - Contextual bandit learning with Upper Confidence Bound
- **LP** - Gupta-Nagarajan Linear Program optimization

## 🚀 Quick Start

### Prerequisites

- AWS Account with Lambda and S3 access
- Python 3.8+ with required packages
- NYC Taxi & Limousine Commission (TLC) data in S3

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ride-hailing-pricing
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_aws.txt
   ```

3. **Configure AWS credentials**:
   ```bash
   aws configure
   ```

4. **Deploy the Lambda function**:
   ```bash
   cd lambdas/pricing-benchmark
   ./deploy.sh
   ```

## 🎮 Basic Usage

### Single Day Experiment

Run pricing experiments for a specific day:

```bash
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=6 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL,Sigmoid \
  --methods=LP,MinMaxCostFlow,LinUCB,MAPS
```

### Multiple Days

Run experiments across multiple days:

```bash
python run_pricing_experiment.py \
  --year=2019 --month=10 --days=1,6,10,15 \
  --borough=Manhattan --vehicle_type=green \
  --eval=PL,Sigmoid \
  --methods=LinUCB,MAPS
```

### Date Range

Run experiments for a continuous date range:

```bash
python run_pricing_experiment.py \
  --year=2019 --month=10 --start_day=1 --end_day=31 \
  --borough=Queens --vehicle_type=yellow \
  --eval=PL \
  --methods=LinUCB
```

### Fast LinUCB Experiments (Skip Training)

Use pre-trained models for quick experiments:

```bash
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=6 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL,Sigmoid --methods=LinUCB,MAPS \
  --skip_training
```

### Force LinUCB Retraining

Retrain LinUCB models from scratch:

```bash
python run_pricing_experiment.py \
  --year=2019 --month=10 --day=6 \
  --borough=Manhattan --vehicle_type=yellow \
  --eval=PL --methods=LinUCB \
  --force_training
```

## 📊 Data & Results

### Input Data Structure

The system expects TLC data in S3 with this structure:
```
s3://magisterka/datasets/
├── yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet
├── green/year=2019/month=10/green_tripdata_2019-10.parquet
└── area_information.csv
```

### Output Results Structure

Results are stored in S3 following this pattern:
```
s3://magisterka/experiments/
└── type=yellow/eval=PL/borough=Manhattan/year=2019/month=10/day=01/
    └── 20250627_pricing_exp_yellow_Manhattan_20250627.json
```

Each result file contains:
- **120 scenarios** (every 5 minutes from 10:00-20:00)
- **Day-level statistics** (mean, std, min, max across scenarios)
- **Method performance comparison**
- **Detailed scenario results**

## 🔧 System Components

### Lambda Function (`lambdas/pricing-benchmark/`)

The core processing engine that:
- Loads real taxi data from S3
- Implements pricing algorithms
- Runs Monte Carlo simulations
- Stores aggregated results

### CLI Tool (`run_pricing_experiment.py`)

Command-line interface that:
- Manages experiment configuration
- Handles LinUCB training automatically
- Orchestrates 120 scenarios per day
- Provides progress tracking

### Training System

LinUCB requires training on historical data:
- **Pre-trained models** available for 2019-07/08/09 periods
- **Automatic training** when needed (10-20 minutes)
- **Skippable training** with `--skip_training` flag for fast experiments
- **Force retraining** with `--force_training` flag
- **July 2019 data** used as default for other months
- **Trained models** stored in S3 for reuse

## 🏙️ Supported Regions

- **Manhattan** - High-density urban area
- **Brooklyn** - Mixed urban/suburban
- **Queens** - Suburban with airport traffic  
- **Bronx** - Urban residential

## 🚗 Vehicle Types

- **Yellow Taxi** - Traditional NYC taxis
- **Green Taxi** - Outer borough taxis
- **FHV** - For-Hire Vehicles (Uber, Lyft, etc.)

## 📈 Evaluation Functions

- **Piecewise Linear (PL)** - Deterministic price-acceptance relationship
- **Sigmoid** - Smooth probabilistic acceptance function

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--year` | Year to analyze | Required |
| `--month` | Month (1-12) | Required |
| `--day/days/start_day` | Day specification | Required |
| `--borough` | NYC borough | Required |
| `--vehicle_type` | Vehicle type | Required |
| `--eval` | Evaluation functions | Required |
| `--methods` | Pricing methods | Required |
| `--training_period` | LinUCB training period | 2019-07 |
| `--skip_training` | Skip LinUCB training (use pre-trained) | False |
| `--force_training` | Force LinUCB retraining | False |
| `--hour_start` | Experiment start hour | 10 |
| `--hour_end` | Experiment end hour | 20 |
| `--time_interval` | Scenario interval (minutes) | 5 |
| `--dry_run` | Preview without execution | False |

## 🎯 Use Cases

### Research & Development
- **Algorithm comparison** across different conditions
- **Parameter sensitivity** analysis
- **Performance benchmarking** on real data

### Operations Analysis  
- **Demand pattern** analysis by time/location
- **Supply-demand matching** optimization
- **Revenue impact** assessment

### Academic Studies
- **Reproducible experiments** with standardized methodology
- **Large-scale evaluation** across multiple days/regions
- **Statistical significance** testing

## 📋 Next Steps

1. **Set up pre-trained LinUCB models** (optional but recommended):
   ```bash
   python prepare_hikima_matrices.py
   ```

2. **Review technical documentation** for detailed usage examples
3. **Set up data pipeline** with your TLC data
4. **Run test experiments** to validate setup
5. **Scale to production** experiments

## 🆘 Support

For technical details, data setup, and advanced usage examples, see:
- **[TECHNICAL.md](TECHNICAL.md)** - Comprehensive technical documentation
- **[Issues](../../issues)** - Report bugs or request features

## 📄 License

See LICENSE file for licensing information. 