#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# AWS Resource Cleanup Script
#
# This script finds and terminates all EC2 instances tagged for this project.
# It provides a safety prompt before taking any action.
#
# Usage:
# ./scripts/cleanup_resources.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# The tag key and value used to identify project resources.
# This should match the tag set in aws_ec2_launcher.py
TAG_KEY="Name"
TAG_VALUE="pricing-experiment"
REGION="${AWS_REGION:-us-east-1}"

echo "🔎 Searching for EC2 instances with tag ${TAG_KEY}=${TAG_VALUE} in region ${REGION}..."

# Find instances and format the output for display
INSTANCE_INFO=$(aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=tag:${TAG_KEY},Values=${TAG_VALUE}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].{ID:InstanceId, Type:InstanceType, State:State.Name, LaunchTime:LaunchTime}' \
  --output table)

INSTANCE_IDS=$(aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=tag:${TAG_KEY},Values=${TAG_VALUE}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' \
  --output text)

if [ -z "$INSTANCE_IDS" ]; then
    echo "✅ No running or stopped instances found with the specified tag. Nothing to do."
    exit 0
fi

echo ""
echo "The following EC2 instances will be permanently terminated:"
echo "${INSTANCE_INFO}"
echo ""

read -p "Are you sure you want to terminate these instances? [y/N] " -n 1 -r
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