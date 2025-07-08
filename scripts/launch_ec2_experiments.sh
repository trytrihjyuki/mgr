#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# Interactive EC2 Experiment Launcher
#
# This script launches a dedicated EC2 instance to run a pricing experiment,
# streams the container logs directly to your terminal via SSH, and ensures
# the instance is terminated afterward.
#
# Usage:
# ./scripts/launch_ec2_experiments.sh --method LinUCB --ec2-type small
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Configuration ---

# Load environment variables from .env file. This is mandatory.
if [ -f .env ]; then
    echo "🔑 Sourcing configuration from .env file..."
    set -a
    source .env
    set +a
else
    echo "❌ Configuration file .env not found."
    echo "   Please copy .env.example to .env and fill in your AWS details."
    exit 1
fi

# --- User-tunable defaults ---
REGION="${AWS_REGION}"
AMI_ID="${AMI_ID:-}"  # Default is empty to auto-resolve latest Amazon Linux
SUBNET_ID="${SUBNET_ID}"
SECURITY_GROUP_IDS=("${SECURITY_GROUP_IDS}")
KEY_NAME="${KEY_NAME}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE}"
REPO_NAME="pricing-experiments"
TAG="latest"
DOCKERFILE="Dockerfile.ec2"
SSH_USER="ec2-user" # User for Amazon Linux 2 AMIs

# --- Experiment Parameters (with default values) ---
START_DATE=$(date -v-1d +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)
START_HOUR=0
END_HOUR=23
BOROUGH="Manhattan"
VEHICLE_TYPE="green"
METHOD="LinUCB"
EVAL_FUNCTIONS="PL,Sigmoid"
NUM_ITER=1000
NUM_PARALLEL=4
EC2_TYPE="small"
SEED=42

# --- Argument Parsing ---
# (Usage function and argument parsing loop remains the same)
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --start-date <YYYY-MM-DD>     Start date (default: yesterday)"
    echo "  --end-date <YYYY-MM-DD>       End date (default: today)"
    echo "  --start-hour <HH>             Start hour (0-23) (default: 0)"
    echo "  --end-hour <HH>               End hour (0-23) (default: 23)"
    echo "  --borough <name>              NYC Borough (default: Manhattan)"
    echo "  --vehicle-type <type>         Taxi type (default: green)"
    echo "  --method <name>               Pricing method(s) (default: LinUCB)"
    echo "  --eval <functions>            Evaluation functions (default: PL,Sigmoid)"
    echo "  --num-iter <n>                Monte Carlo iterations (default: 1000)"
    echo "  --num-parallel <n>            Number of parallel jobs (default: 4)"
    echo "  --ec2-type <size>             Instance size: small, medium, large, xlarge (default: small)"
    echo "  --seed <n>                    Random seed (default: 42)"
    echo "  --help                        Display this help and exit"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start-date) START_DATE="$2"; shift ;;
        --end-date) END_DATE="$2"; shift ;;
        --start-hour) START_HOUR="$2"; shift ;;
        --end-hour) END_HOUR="$2"; shift ;;
        --borough) BOROUGH="$2"; shift ;;
        --vehicle-type) VEHICLE_TYPE="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --eval) EVAL_FUNCTIONS="$2"; shift ;;
        --num-iter) NUM_ITER="$2"; shift ;;
        --num-parallel) NUM_PARALLEL="$2"; shift ;;
        --ec2-type) EC2_TYPE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# --- EC2 Instance Type Mapping ---
case "$EC2_TYPE" in
    small) INSTANCE_TYPE="t3.large" ;;
    medium) INSTANCE_TYPE="m5.2xlarge" ;;
    large) INSTANCE_TYPE="r5dn.8xlarge" ;;
    xlarge) INSTANCE_TYPE="d2.8xlarge" ;;
    *) echo "Error: Invalid EC2 type '$EC2_TYPE'." >&2; exit 1 ;;
esac

# --- Derived variables ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "${REGION}")
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"
EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)_${RANDOM}"

# --- Cleanup Function ---
# This function will be called on script exit to ensure the instance is terminated.
function cleanup() {
    if [ -n "${INSTANCE_ID-}" ]; then
        echo "---"
        echo "🧹 Terminating instance ${INSTANCE_ID}..."
        aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" --no-cli-pager >/dev/null
        echo "✅ Instance termination command issued. Please verify in the AWS console."
    fi
}
# Register the cleanup function to run on script exit (normal or error)
trap cleanup EXIT


########################################
# Main Execution
########################################

# Construct container arguments from script parameters
CONTAINER_ARGS="--start-date ${START_DATE} --end-date ${END_DATE} "
CONTAINER_ARGS+="--start-hour ${START_HOUR} --end-hour ${END_HOUR} "
CONTAINER_ARGS+="--borough ${BOROUGH} --vehicle-type ${VEHICLE_TYPE} "
CONTAINER_ARGS+="--method \"${METHOD}\" --eval ${EVAL_FUNCTIONS} "
CONTAINER_ARGS+="--num-iter ${NUM_ITER} --num-parallel ${NUM_PARALLEL} --seed ${SEED} "
CONTAINER_ARGS+="--experiment-id ${EXPERIMENT_ID}"

# --- Confirmation ---
echo "🚀 Launching experiment with the following configuration:"
echo "-----------------------------------------------------"
echo "Experiment ID:     ${EXPERIMENT_ID}"
echo "EC2 Instance Type: ${EC2_TYPE} (${INSTANCE_TYPE})"
echo "Start Date:        ${START_DATE}"
echo "End Date:          ${END_DATE}"
echo "Borough:           ${BOROUGH}"
echo "Method(s):         ${METHOD}"
echo "Container Args:    ${CONTAINER_ARGS}"
echo "-----------------------------------------------------"

read -p "Do you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# --- Step 1: Build Docker Image ---
echo ""
echo "--- Step 1: Building and Pushing Docker Image ---"
python3 aws_ec2_launcher.py \
    --repo-name "${REPO_NAME}" --tag "${TAG}" --dockerfile "${DOCKERFILE}" \
    --region "${REGION}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --build-only
echo "✅ Docker image build complete."
echo ""

# --- Step 2: Launch EC2 Instance ---
echo "--- Step 2: Launching EC2 Instance ---"
# The python script will print the instance ID to stdout
INSTANCE_ID=$(python3 aws_ec2_launcher.py \
    --no-build --repo-name "${REPO_NAME}" --tag "${TAG}" --region "${REGION}" \
    $( [[ -n "${AMI_ID}" ]] && echo "--ami-id ${AMI_ID}" ) --instance-type "${INSTANCE_TYPE}" \
    --subnet-id "${SUBNET_ID}" --security-group-ids "${SECURITY_GROUP_IDS[@]}" \
    --key-name "${KEY_NAME}" --iam-instance-profile "${IAM_INSTANCE_PROFILE}" \
    --experiment-id "${EXPERIMENT_ID}")

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to launch EC2 instance. Aborting." >&2
    exit 1
fi
echo "✅ EC2 instance ${INSTANCE_ID} launched."
echo ""

# --- Step 3: Wait for Instance to be Ready ---
echo "--- Step 3: Waiting for Instance to be Ready ---"
echo "⏳ Waiting for instance to pass health checks..."
aws ec2 wait instance-status-ok --instance-ids "${INSTANCE_ID}" --region "${REGION}"
echo "✅ Instance is running and healthy."

echo "⏳ Waiting for SSH to be available..."
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region "${REGION}")
while ! nc -z -w5 "${INSTANCE_IP}" 22; do
    echo "   (SSH port not open yet. Retrying in 5 seconds...)"
    sleep 5
done
echo "✅ SSH is available at ${INSTANCE_IP}."
echo ""


# --- Step 4: Run Experiment via SSH ---
echo "--- Step 4: Running Experiment via SSH ---"
echo "👀 Streaming logs from container..."
echo "---"

# Strict host key checking is disabled to avoid interactive prompts.
# This is generally safe in this context as the instance is new and ephemeral.
SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/${KEY_NAME}.pem ${SSH_USER}@${INSTANCE_IP}"

# The remote command to be executed on the EC2 instance
REMOTE_CMD="
set -e;
echo '--- On Instance ---';
echo '1. Logging into ECR...';
aws ecr get-login-password --region ${REGION} | sudo docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com;
echo '2. Pulling container image...';
sudo docker pull ${IMAGE_URI};
echo '3. Running experiment container...';
echo '--- Container Logs Start ---';
sudo docker run --rm ${IMAGE_URI} ${CONTAINER_ARGS};
echo '--- Container Logs End ---';
"

# Execute the remote command and stream output
${SSH_CMD} "${REMOTE_CMD}"
SSH_EXIT_CODE=$?

echo "---"
if [ ${SSH_EXIT_CODE} -eq 0 ]; then
    echo "🎉 Experiment finished successfully!"
else
    echo "⚠️ Experiment finished with exit code ${SSH_EXIT_CODE}." >&2
    echo "   The instance will now be terminated. You may need to check S3 for partial results." >&2
fi
# The 'trap' will handle termination automatically.
exit ${SSH_EXIT_CODE} 