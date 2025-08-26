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
echo -e "${GREEN}  Docker Build & Push to ECR${NC}"
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

# Check if repository exists, create if not
echo -e "${YELLOW}Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating ECR repository ${ECR_REPOSITORY}...${NC}"
    aws ecr create-repository \
        --repository-name ${ECR_REPOSITORY} \
        --region ${AWS_REGION} \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo -e "${GREEN}✓ Repository created${NC}"
else
    echo -e "${GREEN}✓ Repository exists${NC}"
fi

# Get ECR login token
echo -e "${YELLOW}Authenticating with ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
echo -e "${GREEN}✓ Authenticated${NC}"

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    --platform linux/amd64 \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${FULL_IMAGE_NAME} \
    --build-arg AWS_REGION=${AWS_REGION} \
    --build-arg S3_BUCKET=${S3_BUCKET:-magisterka} \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Tag image for ECR
echo -e "${YELLOW}Tagging image for ECR...${NC}"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE_NAME}
echo -e "${GREEN}✓ Image tagged${NC}"

# Push to ECR
echo -e "${YELLOW}Pushing image to ECR...${NC}"
docker push ${FULL_IMAGE_NAME}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image pushed successfully${NC}"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Success!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Image URI: ${YELLOW}${FULL_IMAGE_NAME}${NC}"
    echo ""
    echo "To pull this image:"
    echo -e "  ${YELLOW}./scripts/pull_docker.sh ${IMAGE_TAG}${NC}"
    echo ""
    echo "To run this image:"
    echo -e "  ${YELLOW}./scripts/run_docker.sh --help${NC}"
else
    echo -e "${RED}✗ Push failed${NC}"
    exit 1
fi

# Optional: Also tag and push as 'latest' if not already
if [ "${IMAGE_TAG}" != "latest" ]; then
    echo -e "${YELLOW}Also pushing as 'latest' tag...${NC}"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest
    docker push ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest
    echo -e "${GREEN}✓ 'latest' tag pushed${NC}"
fi 