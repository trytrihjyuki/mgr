# Docker Scripts for Taxi Benchmark

This directory contains scripts for building, pushing, pulling, and running the taxi-benchmark Docker container both locally and on AWS EC2.

## Prerequisites

- Docker installed
- AWS CLI configured with credentials
- `.env` file configured with AWS settings (see root directory)

## Scripts Overview

### 🚀 Main Scripts

| Script | Description | Usage |
|--------|-------------|--------|
| `docker_manager.sh` | Interactive menu-driven Docker management | `./scripts/docker_manager.sh` |
| `push_docker.sh` | Build and push Docker image to ECR | `./scripts/push_docker.sh [tag]` |
| `pull_docker.sh` | Pull Docker image from ECR | `./scripts/pull_docker.sh [tag]` |
| `run_docker.sh` | Run experiments in Docker container | `./scripts/run_docker.sh [options] -- [args]` |

### 📦 push_docker.sh

Builds the Docker image locally and pushes it to Amazon ECR (Elastic Container Registry).

```bash
# Push with latest tag
./scripts/push_docker.sh

# Push with specific tag
./scripts/push_docker.sh v1.0.0
```

**Features:**
- Creates ECR repository if it doesn't exist
- Builds for linux/amd64 platform (compatible with EC2)
- Tags and pushes both specified tag and 'latest'
- Uses credentials from `.env` file

### ⬇️ pull_docker.sh

Pulls a Docker image from Amazon ECR.

```bash
# Pull latest
./scripts/pull_docker.sh

# Pull specific tag
./scripts/pull_docker.sh v1.0.0
```

**Features:**
- Authenticates with ECR
- Checks if image exists before pulling
- Tags pulled image for local use
- Lists available tags if requested tag not found

### 🏃 run_docker.sh

Runs experiments using the Docker container. **Automatically detects** whether you're running locally or on EC2.

```bash
# Show help
./scripts/run_docker.sh --help

# Auto-detect environment and run
./scripts/run_docker.sh -- \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan \
    --methods MinMaxCostFlow \
    --num-iter 100

# Force local mode (build from source)
./scripts/run_docker.sh --local -- \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Queens \
    --methods MAPS LinUCB

# Force remote mode (pull from ECR)
./scripts/run_docker.sh --remote -- \
    --processing-date 2019-10-06 \
    --vehicle-type yellow \
    --boroughs Brooklyn \
    --methods LP
```

**Options:**
- `--local` - Force local mode (builds from Dockerfile)
- `--remote` - Force remote mode (pulls from ECR)
- `--tag TAG` - Specify Docker image tag
- `--no-cache` - Build without cache (local mode only)
- `--interactive` - Run container interactively
- `--` - Separator before experiment arguments

**Auto-detection:**
- On EC2: Automatically uses remote mode (pulls from ECR)
- Locally: Automatically uses local mode (builds from source)

### 🎮 docker_manager.sh

Interactive menu-driven interface for all Docker operations.

```bash
./scripts/docker_manager.sh
```

**Menu Options:**
1. **Build & Push** - Build and push to ECR
2. **Pull** - Pull from ECR
3. **Run Experiment** - Instructions for running experiments
4. **Quick Local Run** - Interactive quick test
5. **Example Experiments** - Show example commands
6. **Check Environment** - Verify Docker/AWS setup
7. **Clean Up** - Remove local images and containers
8. **Exit**

## Environment Variables

The scripts use the following environment variables from `.env`:

```bash
AWS_REGION="eu-north-1"
S3_BUCKET="magisterka"
AWS_ACCESS_KEY_ID="your-key-id"
AWS_SECRET_ACCESS_KEY="your-secret-key"
```

## Examples

### Example 1: Complete Workflow (Local Development)

```bash
# 1. Build and test locally
./scripts/run_docker.sh --local -- \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan \
    --methods MinMaxCostFlow \
    --num-iter 10

# 2. If successful, push to ECR
./scripts/push_docker.sh v1.0.0

# 3. Test pulling from ECR
./scripts/pull_docker.sh v1.0.0
```

### Example 2: EC2 Production Run

```bash
# On EC2 instance (auto-detects and pulls from ECR)
./scripts/run_docker.sh -- \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan Brooklyn Queens Bronx \
    --methods LP MinMaxCostFlow MAPS LinUCB \
    --num-iter 100 \
    --start-hour 6 \
    --end-hour 22
```

### Example 3: Quick Test

```bash
# Interactive quick test
./scripts/docker_manager.sh
# Select option 4 (Quick Local Run)
# Follow prompts
```

### Example 4: Comparing Methods

```bash
# Compare all methods for Queens during lunch hour
./scripts/run_docker.sh -- \
    --processing-date 2019-10-06 \
    --vehicle-type yellow \
    --boroughs Queens \
    --methods LP MinMaxCostFlow MAPS LinUCB \
    --num-iter 100 \
    --start-hour 12 \
    --end-hour 13 \
    --time-delta 15
```

## Dual Evaluation

**Important:** The framework now automatically evaluates **BOTH** PL and Sigmoid acceptance functions for every experiment. You don't need to run experiments twice.

Results are stored separately for:
- **PL (Piecewise Linear)**: Linear acceptance function
- **Sigmoid**: Smooth acceptance function

## S3 Integration

### LinUCB Pre-trained Models
LinUCB automatically loads pre-trained matrices from:
```
s3://taxi-benchmark/models/work/learned_matrix_PL/{month}_{borough}/
```

### Results Storage
Experiment results are automatically saved to:
```
s3://{S3_BUCKET}/results/
```

### Download Results
```bash
# Download all results
aws s3 sync s3://magisterka/results/ ./results/ --region eu-north-1

# Download specific experiment
aws s3 cp s3://magisterka/results/experiment_20240101_120000.json ./results/
```

## Troubleshooting

### Docker Issues

```bash
# Check Docker is running
docker info

# Clean up Docker resources
docker system prune -a

# Or use the cleanup option
./scripts/docker_manager.sh
# Select option 7 (Clean Up)
```

### AWS Issues

```bash
# Check AWS credentials
aws sts get-caller-identity

# Configure AWS CLI
aws configure

# Test ECR access
aws ecr describe-repositories --region eu-north-1
```

### Build Issues

```bash
# Build with no cache
./scripts/run_docker.sh --local --no-cache -- [args]

# Check Dockerfile syntax
docker build --check .
```

## Performance Tips

1. **Use appropriate number of iterations**
   - Testing: 10-50 iterations
   - Production: 100-1000 iterations

2. **Time window selection**
   - Peak hours: 7-9 AM, 5-7 PM
   - Off-peak: 10 AM-4 PM, 8 PM-6 AM

3. **EC2 Instance Types**
   - Development: t3.medium
   - Production: c5.xlarge or larger
   - Memory-intensive: r5.large or larger

## Support

For issues or questions:
1. Check the main README in the root directory
2. Review the `IMPLEMENTATION_STATUS.md` file
3. Check Docker and AWS logs
4. Ensure all dependencies are installed

## License

See the LICENSE file in the root directory. 