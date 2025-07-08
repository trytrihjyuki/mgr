#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# Force Cleanup All Running EC2 Instances
#
# This script finds and terminates ALL running or pending EC2 instances
# in the specified region. It is designed as a powerful tool to clear out
# orphaned resources that may be consuming quota, regardless of their tags.
#
# WARNING: This script is destructive and will list ALL running instances
#          in the account and region. Use with extreme caution.
#
# Usage:
# ./scripts/force_cleanup.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Configuration ---
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
REGION="${AWS_REGION:-us-east-1}"

echo "🔎 Searching for ALL running or pending EC2 instances in region ${REGION}..."

# Find all instances that are not yet terminated
INSTANCE_INFO=$(aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].{ID:InstanceId, Type:InstanceType, State:State.Name, LaunchTime:LaunchTime, Name:Tags[?Key==`Name`].Value | [0]}' \
  --output table)

INSTANCE_IDS=$(aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' \
  --output text)

if [ -z "$INSTANCE_IDS" ]; then
    echo "✅ No running, pending, or stopped instances found. Nothing to do."
    exit 0
fi

echo ""
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "The following EC2 instances from your account will be PERMANENTLY terminated:"
echo "This includes ANY non-terminated instance, not just those from this project."
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "${INSTANCE_INFO}"
echo ""

read -p "Are you sure you want to terminate ALL of these instances? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted. No instances were terminated."
    exit 1
fi

echo ""
echo "🛑 Terminating instances: ${INSTANCE_IDS}"
aws ec2 terminate-instances --region "${REGION}" --instance-ids ${INSTANCE_IDS} > /dev/null

echo "✅ Termination command sent successfully. It may take a few minutes for the instances to disappear from the console." 