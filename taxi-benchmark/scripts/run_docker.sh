#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
AWS_REGION="${AWS_REGION:-eu-north-1}"
S3_BUCKET="${S3_BUCKET:-magisterka}"
IMAGE_NAME="taxi-benchmark"
IMAGE_TAG="latest"
CONTAINER_NAME="taxi-benchmark-experiment"

# Detect if we're on EC2
IS_EC2=false
if [ -f /sys/devices/virtual/dmi/id/product_uuid ]; then
    if grep -qi ec2 /sys/devices/virtual/dmi/id/product_uuid 2>/dev/null; then
        IS_EC2=true
    fi
fi
# Alternative EC2 detection
if [ -f /sys/hypervisor/uuid ]; then
    if grep -qi ec2 /sys/hypervisor/uuid 2>/dev/null; then
        IS_EC2=true
    fi
fi
# Check for EC2 metadata service
if curl -s -m 1 http://169.254.169.254/latest/meta-data/ >/dev/null 2>&1; then
    IS_EC2=true
fi

# Function to show usage
show_usage() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Taxi Benchmark Docker Runner${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] -- [EXPERIMENT_ARGS]"
    echo ""
    echo "Options:"
    echo "  --local          Force local mode (build from source)"
    echo "  --remote         Force remote mode (pull from ECR)"
    echo "  --tag TAG        Docker image tag (default: latest)"
    echo "  --no-cache       Build without cache (local mode only)"
    echo "  --interactive    Run container in interactive mode"
    echo "  --help           Show this help message"
    echo ""
    echo "Experiment Arguments (after --):"
    echo "  --processing-date DATE    Date to process (YYYY-MM-DD)"
    echo "  --vehicle-type TYPE       Vehicle type (green/yellow)"
    echo "  --boroughs BOROUGH...     Boroughs to process"
    echo "  --methods METHOD...       Pricing methods to use"
    echo "  --num-iter N             Number of iterations"
    echo "  --start-hour H           Start hour (0-23)"
    echo "  --end-hour H             End hour (0-23)"
    echo "  --time-delta M           Time delta in minutes"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Run locally with specific experiment parameters:"
    echo "  $0 --local -- --processing-date 2019-10-06 --vehicle-type green \\"
    echo "                 --boroughs Manhattan Brooklyn --methods LP MinMaxCostFlow MAPS LinUCB \\"
    echo "                 --num-iter 100 --start-hour 10 --end-hour 12"
    echo ""
    echo "  # Run on EC2 (pulls from ECR):"
    echo "  $0 --remote -- --processing-date 2019-10-06 --vehicle-type yellow \\"
    echo "                  --boroughs Queens --methods MAPS --num-iter 50"
    echo ""
    echo "  # Auto-detect environment:"
    echo "  $0 -- --processing-date 2019-10-06 --vehicle-type green \\"
    echo "        --boroughs Manhattan --methods MinMaxCostFlow"
    echo ""
    exit 0
}

# Parse options
LOCAL_MODE=false
REMOTE_MODE=false
NO_CACHE=""
INTERACTIVE=""
EXPERIMENT_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --remote)
            REMOTE_MODE=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --interactive)
            INTERACTIVE="-it"
            shift
            ;;
        --help)
            show_usage
            ;;
        --)
            shift
            EXPERIMENT_ARGS="$@"
            break
            ;;
        *)
            EXPERIMENT_ARGS="$@"
            break
            ;;
    esac
done

# Determine mode if not explicitly set
if [ "$LOCAL_MODE" = true ] && [ "$REMOTE_MODE" = true ]; then
    echo -e "${RED}Error: Cannot specify both --local and --remote${NC}"
    exit 1
fi

if [ "$LOCAL_MODE" = false ] && [ "$REMOTE_MODE" = false ]; then
    if [ "$IS_EC2" = true ]; then
        echo -e "${BLUE}ℹ Detected EC2 environment, using remote mode${NC}"
        REMOTE_MODE=true
    else
        echo -e "${BLUE}ℹ Detected local environment, using local mode${NC}"
        LOCAL_MODE=true
    fi
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Running Taxi Benchmark Experiment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Handle image based on mode
if [ "$REMOTE_MODE" = true ]; then
    echo -e "${YELLOW}Mode: Remote (ECR)${NC}"
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo -e "${RED}Error: Unable to get AWS account ID. Check your AWS credentials.${NC}"
        exit 1
    fi
    
    ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    FULL_IMAGE_NAME="${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Pull from ECR
    echo -e "${YELLOW}Pulling image from ECR...${NC}"
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
    docker pull ${FULL_IMAGE_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Image pulled successfully${NC}"
        DOCKER_IMAGE="${FULL_IMAGE_NAME}"
    else
        echo -e "${RED}✗ Failed to pull image from ECR${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Mode: Local (Build from source)${NC}"
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}Error: Dockerfile not found in current directory${NC}"
        exit 1
    fi
    
    # Build locally
    echo -e "${YELLOW}Building Docker image locally...${NC}"
    docker build \
        ${NO_CACHE} \
        --platform linux/amd64 \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        --build-arg AWS_REGION=${AWS_REGION} \
        --build-arg S3_BUCKET=${S3_BUCKET} \
        -f Dockerfile \
        .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Image built successfully${NC}"
        DOCKER_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    else
        echo -e "${RED}✗ Failed to build image${NC}"
        exit 1
    fi
fi

# Remove existing container if it exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm -f ${CONTAINER_NAME}
fi

# Prepare AWS credentials for container
AWS_CREDS=""
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    AWS_CREDS="-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}"
elif [ "$IS_EC2" = true ]; then
    echo -e "${BLUE}ℹ Using EC2 instance IAM role for AWS credentials${NC}"
else
    echo -e "${YELLOW}⚠ No AWS credentials found. S3 operations may fail.${NC}"
fi

# Run the container
echo -e "${YELLOW}Starting container...${NC}"
echo -e "${BLUE}Command: python run_experiment.py ${EXPERIMENT_ARGS}${NC}"
echo ""

docker run \
    ${INTERACTIVE} \
    --rm \
    --name ${CONTAINER_NAME} \
    -e AWS_REGION=${AWS_REGION} \
    -e S3_BUCKET=${S3_BUCKET} \
    ${AWS_CREDS} \
    -v /var/run/docker.sock:/var/run/docker.sock \
    ${DOCKER_IMAGE} \
    python run_experiment.py ${EXPERIMENT_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Experiment Completed Successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [ -n "${S3_BUCKET}" ]; then
        echo ""
        echo -e "${YELLOW}Results should be available in S3:${NC}"
        echo -e "  ${BLUE}s3://${S3_BUCKET}/results/${NC}"
        echo ""
        echo "To download results:"
        echo -e "  ${YELLOW}aws s3 sync s3://${S3_BUCKET}/results/ ./results/ --region ${AWS_REGION}${NC}"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Experiment Failed (Exit code: $EXIT_CODE)${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE 