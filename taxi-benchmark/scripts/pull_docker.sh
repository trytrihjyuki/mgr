#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
AWS_REGION="${AWS_REGION:-eu-north-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
IMAGE_NAME="taxi-benchmark"
IMAGE_TAG="${1:-latest}"
ECR_REPOSITORY="${IMAGE_NAME}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Docker Pull from ECR${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Check AWS credentials
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}Error: Unable to get AWS account ID. Check your AWS credentials.${NC}"
    exit 1
fi

echo -e "${YELLOW}AWS Account ID:${NC} ${AWS_ACCOUNT_ID}"
echo -e "${YELLOW}AWS Region:${NC} ${AWS_REGION}"
echo -e "${YELLOW}ECR Repository:${NC} ${ECR_REPOSITORY}"
echo -e "${YELLOW}Image Tag:${NC} ${IMAGE_TAG}"
echo ""

# Get ECR login token
echo -e "${YELLOW}Authenticating with ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Authenticated${NC}"
else
    echo -e "${RED}✗ Authentication failed${NC}"
    exit 1
fi

# Check if image exists in ECR
echo -e "${YELLOW}Checking if image exists in ECR...${NC}"
if aws ecr describe-images --repository-name ${ECR_REPOSITORY} --image-ids imageTag=${IMAGE_TAG} --region ${AWS_REGION} >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Image found in ECR${NC}"
else
    echo -e "${RED}✗ Image not found in ECR${NC}"
    echo -e "${YELLOW}Available tags:${NC}"
    aws ecr describe-images --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION} --query 'imageDetails[*].imageTags[*]' --output text 2>/dev/null | tr '\t' '\n' | sort -u
    exit 1
fi

# Pull image from ECR
echo -e "${YELLOW}Pulling image from ECR...${NC}"
docker pull ${FULL_IMAGE_NAME}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image pulled successfully${NC}"
    
    # Also tag as local image for convenience
    echo -e "${YELLOW}Tagging as local image...${NC}"
    docker tag ${FULL_IMAGE_NAME} ${IMAGE_NAME}:${IMAGE_TAG}
    echo -e "${GREEN}✓ Tagged as ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Success!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Image: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo -e "Full URI: ${YELLOW}${FULL_IMAGE_NAME}${NC}"
    echo ""
    echo "To run this image:"
    echo -e "  ${YELLOW}./scripts/run_docker.sh --help${NC}"
    echo ""
    echo "Available local images:"
    docker images | grep ${IMAGE_NAME} | head -5
else
    echo -e "${RED}✗ Pull failed${NC}"
    exit 1
fi 