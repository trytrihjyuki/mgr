#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# Batch EC2 Experiment Launcher
#
# This script provides a flexible way to launch pricing experiments on EC2.
# It allows for detailed configuration of experiment parameters, including
# date ranges, pricing methods, vehicle types, and EC2 instance sizes.
#
# It wraps `aws_ec2_launcher.py`, building the image once and then launching
# an instance with the specified parameters.
#
# Usage:
# ./scripts/launch_ec2_experiments.sh --start-date 2019-10-01 --end-date 2019-10-03 --method LinUCB --ec2-type medium
# ---------------------------------------------------------------------------

set -euo pipefail

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "🔑 Sourcing configuration from .env file..."
    # Export the variables to make them available to sub-processes
    set -a
    source .env
    set +a
fi

########################################
# User-tunable defaults
########################################

REGION="${AWS_REGION:-us-east-1}"
AMI_ID="${AMI_ID:-}"  # leave empty to auto-resolve via SSM
SUBNET_ID="${SUBNET_ID:-subnet-0123456789abcdef0}"
SECURITY_GROUP_IDS=("${SECURITY_GROUP_IDS_OVERRIDE:-${SECURITY_GROUP_IDS:-sg-0123456789abcdef0}}")
KEY_NAME="${KEY_NAME:-myKeyPair}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-PricingExperimentRole}"
S3_BUCKET_NAME="${S3_BUCKET:-magisterka}"

REPO_NAME="pricing-experiments"
TAG="latest"
DOCKERFILE="Dockerfile.ec2"

# Experiment Parameters with default values
START_DATE=$(date -v-1d +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)
START_HOUR=0
END_HOUR=23
BOROUGH="Manhattan"
VEHICLE_TYPE="green"
METHOD="LinUCB"
ACCEPTANCE_FUNCTION="PL"
NUM_ITER=1000
NUM_PARALLEL=4
EC2_TYPE="small"
SEED=42

########################################
# Argument Parsing
########################################

function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --start-date <YYYY-MM-DD>     Start date for the experiment (default: yesterday)"
    echo "  --end-date <YYYY-MM-DD>       End date for the experiment (default: today)"
    echo "  --start-hour <HH>             Start hour (0-23) (default: 0)"
    echo "  --end-hour <HH>               End hour (0-23) (default: 23)"
    echo "  --borough <name>              NYC Borough (default: Manhattan)"
    echo "  --vehicle-type <type>         Taxi type: green, yellow, fhv (default: green)"
    echo "  --method <name>               Pricing method: LinUCB, LP, etc. (default: LinUCB)"
    echo "  --acceptance-function <name>  Acceptance function: PL, Sigmoid (default: PL)"
    echo "  --num-iter <n>                Number of Monte Carlo iterations (default: 1000)"
    echo "  --num-parallel <n>            Number of parallel jobs (default: 4)"
    echo "  --ec2-type <size>             Instance size: small, medium, large, xlarge, extra-large (default: small)"
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
        --acceptance-function) ACCEPTANCE_FUNCTION="$2"; shift ;;
        --num-iter) NUM_ITER="$2"; shift ;;
        --num-parallel) NUM_PARALLEL="$2"; shift ;;
        --ec2-type) EC2_TYPE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

########################################
# EC2 Instance Type Mapping
########################################
case "$EC2_TYPE" in
    small)
        INSTANCE_TYPE="t3.large"
        ;;
    medium)
        INSTANCE_TYPE="m5.2xlarge"
        ;;
    large)
        INSTANCE_TYPE="r5dn.8xlarge"
        ;;
    xlarge)
        INSTANCE_TYPE="d2.8xlarge"
        ;;
    extra-large)
        INSTANCE_TYPE="u-6tb1.112xlarge"
        ;;
    *)
        echo "Error: Invalid EC2 type '$EC2_TYPE'. Please use one of: small, medium, large, xlarge, extra-large"
        exit 1
        ;;
esac

########################################
# Derived variables (do not modify)
########################################

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"
EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)_${RANDOM}"


########################################
# Helper functions
########################################

function build_image_once() {
  echo "[+] Building & pushing image to ECR (${IMAGE_URI})"
  python3 aws_ec2_launcher.py \
    --repo-name "${REPO_NAME}" \
    --tag "${TAG}" \
    --dockerfile "${DOCKERFILE}" \
    --region "${REGION}" \
    --build-only
}

function launch_job() {
  local INSTANCE_TYPE=$1
  shift
  local CONTAINER_ARGS=$*

  python3 aws_ec2_launcher.py \
    --no-build --repo-name "${REPO_NAME}" --tag "${TAG}" --region "${REGION}" \
    $( [[ -n "${AMI_ID}" ]] && echo "--ami-id ${AMI_ID}" ) --instance-type "${INSTANCE_TYPE}" \
    --subnet-id "${SUBNET_ID}" --security-group-ids "${SECURITY_GROUP_IDS[@]}" \
    --key-name "${KEY_NAME}" --iam-instance-profile "${IAM_INSTANCE_PROFILE}" \
    --container-args "${CONTAINER_ARGS}"
}

function monitor_experiment() {
    local instance_id=$1
    local experiment_id=$2
    local bucket_name="${S3_BUCKET_NAME}"
    local success_s3_key="experiments/${experiment_id}/_SUCCESS"
    local progress_s3_key="experiments/${experiment_id}/_PROGRESS"
    local timeout_seconds=10800 # 3 hours
    local check_interval_seconds=60

    echo "👀 Monitoring experiment on instance ${instance_id}..."
    echo "⏳ Will wait for _SUCCESS file at: s3://${bucket_name}/${success_s3_key}"

    local start_time=$(date +%s)

    while true; do
        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))

        if (( elapsed_time > timeout_seconds )); then
            echo "❌ Timeout of ${timeout_seconds} seconds reached. Aborting monitoring."
            return 1
        fi

        # Check for _SUCCESS file
        if aws s3api head-object --bucket "${bucket_name}" --key "${success_s3_key}" >/dev/null 2>&1; then
            echo "✅ SUCCESS file found. Experiment completed successfully."
            return 0
        fi

        # Check instance status
        local instance_status=$(aws ec2 describe-instance-status --instance-ids "${instance_id}" --include-all-instances --query "InstanceStatuses[0].InstanceState.Name" --output text 2>/dev/null)

        if [[ "$instance_status" == "stopped" || "$instance_status" == "terminated" || "$instance_status" == "shutting-down" ]]; then
            echo "❌ Instance ${instance_id} is in state: ${instance_status}. Experiment may have failed."
            return 1
        fi
        
        # Check for _PROGRESS file
        local progress_info="--"
        local progress_json=$(aws s3api get-object --bucket "${bucket_name}" --key "${progress_s3_key}" /dev/null 2>/dev/null || echo "{}")
        if [[ -n "$progress_json" && "$progress_json" != "{}" ]]; then
            local processed=$(echo "$progress_json" | grep "processed_scenarios" | sed 's/[^0-9]*//g')
            local total=$(echo "$progress_json" | grep "total_scenarios" | sed 's/[^0-9]*//g')
            local percent=$(echo "$progress_json" | grep "progress_percentage" | sed 's/[^0-9.]*//g')
            if [[ -n "$processed" && -n "$total" ]]; then
                progress_info="Progress: ${processed}/${total} (${percent}%)"
            fi
        fi

        echo "   (Elapsed: ${elapsed_time}s) ${progress_info}. Instance status: ${instance_status:-running}. Checking again in ${check_interval_seconds}s..."
        sleep "${check_interval_seconds}"
    done
}


########################################
# Main Execution
########################################

# Construct container arguments from script parameters
CONTAINER_ARGS="--start-date ${START_DATE} --end-date ${END_DATE} "
CONTAINER_ARGS+="--start-hour ${START_HOUR} --end-hour ${END_HOUR} "
CONTAINER_ARGS+="--borough ${BOROUGH} --vehicle-type ${VEHICLE_TYPE} "
CONTAINER_ARGS+="--method ${METHOD} --acceptance-function ${ACCEPTANCE_FUNCTION} "
CONTAINER_ARGS+="--num-iter ${NUM_ITER} --num-parallel ${NUM_PARALLEL} --seed ${SEED} "
CONTAINER_ARGS+="--experiment-id ${EXPERIMENT_ID}"

echo "🚀 Launching experiment with the following configuration:"
echo "-----------------------------------------------------"
echo "Experiment ID:     ${EXPERIMENT_ID}"
echo "EC2 Instance Type: ${EC2_TYPE} (${INSTANCE_TYPE})"
echo "Start Date:        ${START_DATE}"
echo "End Date:          ${END_DATE}"
echo "Time Range:        ${START_HOUR}:00 - ${END_HOUR}:00"
echo "Borough:           ${BOROUGH}"
echo "Vehicle Type:      ${VEHICLE_TYPE}"
echo "Method:            ${METHOD}"
echo "Acceptance Func:   ${ACCEPTANCE_FUNCTION}"
echo "Iterations:        ${NUM_ITER}"
echo "Parallel Jobs:     ${NUM_PARALLEL}"
echo "Seed:              ${SEED}"
echo "-----------------------------------------------------"
echo "Container Args:    ${CONTAINER_ARGS}"
echo "-----------------------------------------------------"

read -p "Do you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "--- Step 1: Building Docker Image ---"
build_image_once
echo "✅ Docker image build complete."
echo ""

echo "--- Step 2: Launching EC2 Instance ---"
INSTANCE_ID=$(launch_job "${INSTANCE_TYPE}" "${CONTAINER_ARGS}" | grep "Instance launched:" | awk '{print $3}')

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to launch EC2 instance. Aborting."
    exit 1
fi
echo "✅ EC2 instance ${INSTANCE_ID} launched successfully."
echo ""


echo "--- Step 3: Monitoring Experiment ---"
monitor_experiment "$INSTANCE_ID" "$EXPERIMENT_ID"
MONITOR_EXIT_CODE=$?

echo ""
echo "--- Experiment Complete ---"
if [ $MONITOR_EXIT_CODE -eq 0 ]; then
    echo "🎉 Experiment finished and verified successfully!"
else
    echo "⚠️ Experiment finished with an unexpected status or timed out."
    echo "   Please check the instance logs in CloudWatch and the results in S3 manually."
fi

echo "ℹ️  The instance should now be stopped or terminated. Please verify in the EC2 console." 