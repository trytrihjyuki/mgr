#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# EC2 Experiment Launcher
#
# This script is a wrapper around the `aws_ec2_launcher.py` Python script.
# It simplifies the process of launching a pricing experiment on a dedicated
# EC2 instance by sourcing environment variables and passing them as
# command-line arguments.
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
    # Disable unbound variable errors temporarily
    set +u
    source .env
    # Re-enable unbound variable errors
    set -u
    set +a
else
    echo "❌ Configuration file .env not found."
    echo "   Please copy .env.example to .env and fill in your AWS details."
    exit 1
fi

# --- Main Execution ---
# The Python script handles all logic, including argument parsing and execution.
echo "🚀 Launching experiment via aws_ec2_launcher.py..."
echo "Passing all script arguments: $@"

python3 aws_ec2_launcher.py "$@"

echo "✅ Script execution finished." 