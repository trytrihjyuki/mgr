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
    """Thin wrapper around subprocess.run that prints the command."""
    print("🚀", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True, **kwargs)


def ensure_ecr_repository(ecr_client, repo_name: str):
    """Create ECR repository if it does not yet exist."""
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        print(f"💿 ECR repo '{repo_name}' already exists")
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(f"💿 Creating ECR repo '{repo_name}'…")
        ecr_client.create_repository(repositoryName=repo_name)


def docker_login_to_ecr(region: str, account_id: str):
    """Authenticate the local Docker daemon against ECR."""
    print("🔑 Logging in to ECR…")
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

    print("🐳 Building image…")
    run(["docker", "build", "-f", dockerfile, "-t", image_uri, "."])

    print("📤 Pushing image…")
    run(["docker", "push", image_uri])

    return image_uri


# ---------------------------------------------------------------------------
# EC2 launching helpers
# ---------------------------------------------------------------------------

def generate_user_data(image_uri: str, region: str, container_args: str = "") -> str:
    """Return base64-encoded user-data that pulls the image and runs the container."""

    bash_script = textwrap.dedent(
        f"""#!/bin/bash
        set -euxo pipefail
        yum update -y
        amazon-linux-extras install docker -y || yum install docker -y
        service docker start

        # ECR login
        aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri.split('/')[0]}

        # Pull & run container
        docker pull {image_uri}
        CPUs=$(nproc)
        docker run --rm --cpus=$CPUs {image_uri} {container_args}

        shutdown -h now
        """
    )
    return base64.b64encode(bash_script.encode()).decode()


def launch_instance(
    image_uri: str,
    region: str,
    ami_id: str,
    instance_type: str,
    key_name: str,
    security_group_ids: List[str],
    subnet_id: str,
    iam_instance_profile: str,
    container_args: str,
):
    """Launch an EC2 instance that immediately runs the pricing experiment container."""
    ec2 = boto3.client("ec2", region_name=region)

    user_data_b64 = generate_user_data(image_uri, region, container_args)

    print("📦 Launching EC2 instance…")
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
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "pricing-experiment"}],
            }
        ],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"🆔 Instance launched: {instance_id}")
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
    parser.add_argument("--container-args", default="--day 15 --hour_start 10 --hour_end 11 --time_interval 5 --parallel 2",
                        help="Arguments forwarded to run_pricing_experiment.py inside the container")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.no_build:
        image_uri = build_and_push_image(args.repo_name, args.tag, args.dockerfile, args.region)
    else:
        # Derive account ID to construct URI
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        image_uri = f"{account_id}.dkr.ecr.{args.region}.amazonaws.com/{args.repo_name}:{args.tag}"
        print(f"⚠️ Skipping build, will use existing image {image_uri}")

    if args.build_only:
        print("✅ Build only mode. Exiting without launching instance.")
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
        print(f"❌ Error: The following arguments are required to launch an instance: {', '.join(missing_args)}")
        sys.exit(1)


    ami_id = args.ami_id
    if ami_id and ami_id.startswith("ami-0abcdef"):
        ami_id = None
    ami_id = ami_id or resolve_default_ami(args.region)

    instance_id = launch_instance(
        image_uri=image_uri,
        region=args.region,
        ami_id=ami_id,
        instance_type=args.instance_type,
        key_name=args.key_name,
        security_group_ids=args.security_group_ids,
        subnet_id=args.subnet_id,
        iam_instance_profile=args.iam_instance_profile,
        container_args=args.container_args,
    )

    print("🎉 All done. Monitor progress via EC2 console or CloudWatch logs. Remember to terminate the instance when finished!")


if __name__ == "__main__":
    try:
        main()
    except ClientError as ce:
        print(f"AWS error: {ce}")
        sys.exit(1)
    except subprocess.CalledProcessError as cpe:
        print(f"Subprocess error: {cpe.cmd} exited with {cpe.returncode}")
        sys.exit(cpe.returncode) 