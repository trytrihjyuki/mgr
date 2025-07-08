import argparse
import base64
import os
import subprocess
import sys
import textwrap
import time
from typing import List

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def run(cmd: List[str], check: bool = True, capture_output: bool = False, **kwargs):
    """Thin wrapper around subprocess.run that prints the command to stderr."""
    print("🚀", " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True, **kwargs)


def ensure_ecr_repository(ecr_client, repo_name: str):
    """Create ECR repository if it does not yet exist."""
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        print(f"💿 ECR repo '{repo_name}' already exists", file=sys.stderr)
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(f"💿 Creating ECR repo '{repo_name}'…", file=sys.stderr)
        ecr_client.create_repository(repositoryName=repo_name)


def docker_login_to_ecr(region: str, account_id: str):
    """Authenticate the local Docker daemon against ECR."""
    print("🔑 Logging in to ECR…", file=sys.stderr)
    login_cmd = [
        "aws",
        "ecr",
        "get-login-password",
        "--region",
        region,
    ]
    password = run(login_cmd, capture_output=True).stdout.strip()
    run(
        [
            "docker",
            "login",
            f"{account_id}.dkr.ecr.{region}.amazonaws.com",
            "--username",
            "AWS",
            "--password-stdin",
        ],
        input=password,
    )


def build_and_push_image(repo_name: str, tag: str, dockerfile: str, region: str):
    """Build the Docker image locally and push it to ECR."""
    sts = boto3.client("sts")
    ecr = boto3.client("ecr", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    ensure_ecr_repository(ecr, repo_name)
    docker_login_to_ecr(region, account_id)

    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}"

    print("🐳 Building image…", file=sys.stderr)
    run(["docker", "build", "-f", dockerfile, "-t", image_uri, "."])

    print("📤 Pushing image…", file=sys.stderr)
    run(["docker", "push", image_uri])

    return image_uri


# ---------------------------------------------------------------------------
# EC2 launching helpers
# ---------------------------------------------------------------------------

def generate_user_data() -> str:
    """
    Return a base64-encoded user-data script that prepares the instance
    by installing and starting Docker.
    """
    bash_script = textwrap.dedent(
        f"""#!/bin/bash
        # Log everything to a file on the instance for debugging
        exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
        
        echo "--- User Data Script Start ---"
        set -euxo pipefail
        
        # --- Install and start Docker ---
        echo "Updating yum and installing docker..."
        yum update -y
        # Use the amazon-linux-extras installer for AL2, fall back to yum for other versions
        amazon-linux-extras install docker -y || yum install -y docker
        
        echo "Starting docker service..."
        service docker start
        
        echo "--- User Data Script End ---"
        """
    )
    return base64.b64encode(bash_script.encode()).decode()


def launch_instance(
    region: str,
    ami_id: str,
    instance_type: str,
    key_name: str,
    security_group_ids: List[str],
    subnet_id: str,
    iam_instance_profile: str,
    experiment_id: str,
):
    """Launch a bare EC2 instance ready for SSH commands."""
    ec2 = boto3.client("ec2", region_name=region)
    user_data_b64 = generate_user_data()

    print("📦 Launching EC2 instance…", file=sys.stderr)
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=security_group_ids,
        SubnetId=subnet_id,
        IamInstanceProfile={"Name": iam_instance_profile},
        UserData=user_data_b64,
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior='terminate', # Ensure it terminates if shutdown from within
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"pricing-experiment-{experiment_id}"},
                    {"Key": "ExperimentID", "Value": experiment_id},
                ],
            }
        ],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"✅ Instance {instance_id} is being launched.", file=sys.stderr)
    return instance_id


def resolve_default_ami(region: str) -> str:
    """Return an Amazon Linux AMI ID valid in the provided region via SSM."""
    ssm = boto3.client("ssm", region_name=region)
    param_candidates = [
        "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64",
        "/aws/service/ami-amazon-linux-latest/al2023-ami-minimal-kernel-6.1-x86_64",
        "/aws/service/ami-amazon-linux-latest/al2-ami-hvm-x86_64-gp2",
    ]
    for name in param_candidates:
        try:
            return ssm.get_parameter(Name=name)["Parameter"]["Value"]
        except ssm.exceptions.ParameterNotFound:
            continue
    raise RuntimeError(f"Could not resolve a default Amazon Linux AMI for region {region}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Launch pricing experiment on EC2 from the command-line")

    # Docker / ECR
    parser.add_argument("--repo-name", default="pricing-experiments", help="ECR repository name")
    parser.add_argument("--tag", default="latest", help="Docker tag to use")
    parser.add_argument("--dockerfile", default="Dockerfile.ec2")
    parser.add_argument("--no-build", action="store_true", help="Skip building & pushing the image")
    parser.add_argument("--build-only", action="store_true", help="Only build & push the image, do not launch an instance")

    # EC2 networking & instance
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--ami-id", help="AMI ID to use (will auto-resolve Amazon Linux if omitted)")
    parser.add_argument("--instance-type", default="r5dn.8xlarge")
    parser.add_argument("--subnet-id", help="Subnet ID to launch the instance in")
    parser.add_argument("--security-group-ids", nargs="+", help="One or more security group IDs")
    parser.add_argument("--key-name", help="Name of the EC2 key pair to use")
    parser.add_argument("--iam-instance-profile", help="Instance profile name with ECR + S3 permissions")

    # Experiment parameters
    parser.add_argument("--experiment-id", required=True, help="Unique ID for this experiment run")
    
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.build_only and not args.no_build:
        build_and_push_image(args.repo_name, args.tag, args.dockerfile, args.region)
    elif args.no_build:
        print(f"⚠️ Skipping build, will use existing image.", file=sys.stderr)

    if args.build_only:
        print("✅ Build only mode. Exiting without launching instance.", file=sys.stderr)
        return

    # --- Validate arguments for instance launch ---
    required_for_launch = {
        "subnet-id": args.subnet_id,
        "security-group-ids": args.security_group_ids,
        "key-name": args.key_name,
        "iam-instance-profile": args.iam_instance_profile,
    }
    missing_args = [f"--{arg}" for arg, value in required_for_launch.items() if not value]

    if missing_args:
        print(f"❌ Error: The following arguments are required to launch an instance: {', '.join(missing_args)}", file=sys.stderr)
        sys.exit(1)


    ami_id = args.ami_id
    if ami_id and ami_id.startswith("ami-0abcdef"):
        ami_id = None
    ami_id = ami_id or resolve_default_ami(args.region)

    instance_id = launch_instance(
        region=args.region,
        ami_id=ami_id,
        instance_type=args.instance_type,
        key_name=args.key_name,
        security_group_ids=args.security_group_ids,
        subnet_id=args.subnet_id,
        iam_instance_profile=args.iam_instance_profile,
        experiment_id=args.experiment_id,
    )
    
    # Print instance ID to stdout for the calling script to capture
    print(instance_id)


if __name__ == "__main__":
    try:
        main()
    except ClientError as ce:
        print(f"AWS error: {ce}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as cpe:
        print(f"Subprocess error: {cpe.cmd} exited with {cpe.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) 