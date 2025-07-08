# Technical Guide

This document provides a detailed technical overview of the ride-hailing pricing experiment framework. It covers the system architecture, advanced usage scenarios, and instructions for configuration and troubleshooting.

## 1. System Architecture

The framework is designed to run pricing experiments in a containerized environment on AWS EC2. This provides a flexible and powerful way to execute large-scale simulations without the limitations of a serverless architecture.

The core components of the system are:

- **`scripts/launch_ec2_experiments.sh`**: The main user-facing script for launching experiments. It's a bash wrapper that simplifies the process of configuring and running a new experiment.
- **`aws_ec2_launcher.py`**: A Python script that handles the AWS infrastructure automation. It builds and pushes the Docker image to ECR, and then provisions an EC2 instance to run the experiment.
- **`run_pricing_experiment.py`**: The heart of the experiment logic. This Python script runs inside the Docker container and is responsible for loading data, executing the pricing algorithms, and saving the results.
- **`Dockerfile.ec2`**: Defines the container environment, ensuring all necessary dependencies are installed for the experiment to run correctly.

### Architectural Workflow

The following diagram illustrates the workflow from launching an experiment to retrieving the results:

```mermaid
graph TD
    A[User] -- runs --> B(scripts/launch_ec2_experiments.sh);
    B -- calls --> C(aws_ec2_launcher.py);
    C -- builds & pushes --> D[ECR Repository];
    C -- launches --> E[EC2 Instance];
    E -- pulls image from --> D;
    E -- runs container with --> F(run_pricing_experiment.py);
    F -- reads data from --> G[S3 Bucket];
    F -- runs pricing algorithms --> F;
    F -- writes results to --> G;
    E -- self-terminates --> H((Instance Terminated));
```

## 2. Scenario Showcase

The `launch_ec2_experiments.sh` script is highly flexible. Here are some examples of different experiment scenarios you can run.

### Scenario 1: Quick Sanity Check

**Goal:** Verify that the entire pipeline is working correctly with a minimal, low-cost run.
**Configuration:** One day of data, the default `LinUCB` method, and a `small` EC2 instance.

```bash
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-01 \
  --end-date 2019-10-01 \
  --ec2-type small
```

### Scenario 2: Method Comparison (Weekend Analysis)

**Goal:** Compare the performance of the `LP` and `LinUCB` pricing methods over a weekend.
**Configuration:** This requires two separate runs, one for each method.

**Run 1: LP Method**
```bash
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-05 \
  --end-date 2019-10-06 \
  --method LP \
  --vehicle-type yellow \
  --ec2-type medium \
  --num-parallel 8
```

**Run 2: LinUCB Method**
```bash
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-05 \
  --end-date 2019-10-06 \
  --method LinUCB \
  --vehicle-type yellow \
  --ec2-type medium \
  --num-parallel 8
```

### Scenario 3: Borough Performance Analysis

**Goal:** Analyze the impact of a pricing method in different parts of the city.
**Configuration:** Run the same experiment configuration in two different boroughs.

**Run 1: Manhattan**
```bash
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-10 \
  --end-date 2019-10-10 \
  --borough Manhattan \
  --ec2-type medium
```

**Run 2: Brooklyn**
```bash
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-10 \
  --end-date 2019-10-10 \
  --borough Brooklyn \
  --ec2-type medium
```

### Scenario 4: High-Resolution Rush Hour Analysis

**Goal:** Focus on a narrow time window (e.g., evening rush hour) to analyze pricing dynamics with greater detail.
**Configuration:** Use the `--start-hour` and `--end-hour` parameters to define the time window.

```bash
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-21 \
  --end-date 2019-10-25 \
  --start-hour 16 \
  --end-hour 20 \
  --method MAPS \
  --ec2-type large \
  --num-parallel 16
```

## 3. Infrastructure Configuration

Before running the launcher script, you need to provide your AWS infrastructure details. The recommended method is to create a `.env` file in the root of the project.

1.  **Create the `.env` file:** Copy the provided template.
```bash
    cp .env.example .env
    ```
2.  **Edit `.env`:** Open the `.env` file and fill in the values for your AWS environment. The launch script will automatically source this file.

Here are some one-liner commands to help you find these values.

- **`REGION`**: The AWS region where the resources will be created.

- **`SUBNET_ID`**: Find subnets in your default VPC. This subnet must have internet access to pull the Docker image.
  ```sh
  aws ec2 describe-subnets --filters "Name=vpc-id,Values=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)" --query "Subnets[].SubnetId" --output text
  ```

- **`SECURITY_GROUP_IDS`**: Find security groups, preferably one named `default` or `pricing-experiment`. Must allow outbound HTTPS traffic (for ECR and S3) and inbound SSH if you need to debug.
  ```sh
  aws ec2 describe-security-groups --query "SecurityGroups[?GroupName=='default' || contains(GroupName, 'pricing')].GroupId" --output text
  ```

- **`KEY_NAME`**: List your available EC2 key pairs.
  ```sh
  aws ec2 describe-key-pairs --query "KeyPairs[].KeyName" --output text
  ```
  
- **`IAM_INSTANCE_PROFILE`**: The name of the IAM instance profile for the EC2 instance. It needs permissions for ECR and S3.
  
  To find a suitable profile, run:
  ```sh
  aws iam list-instance-profiles --query "InstanceProfiles[?contains(InstanceProfileName, 'Pricing')].InstanceProfileName" --output text
  ```

> **Handling Multiple Resource IDs:** The CLI commands provided are for guidance. If they return multiple, untagged resources, it can be difficult to identify the correct one. In this situation, the most reliable approach is to:
> 1.  Log in to the **AWS Management Console**.
> 2.  Navigate to the specific service (e.g., VPC, EC2 Security Groups).
> 3.  Manually identify the correct resource and copy its ID.
> 4.  **Best Practice:** Apply a consistent `Name` tag to all resources related to this project (e.g., `pricing-experiment-vpc`, `pricing-experiment-sg`) to make them easily discoverable via the CLI in the future.

## 4. Monitoring & Debugging

While fully automated monitoring is not yet implemented, you can manually track the progress of your experiments.

1.  **EC2 Console:** After launching an experiment, the instance ID will be printed to the console. You can use this ID to find the instance in the AWS EC2 console and monitor its state.
2.  **CloudWatch Logs:** The `user-data` script configures the instance to send container logs to Amazon CloudWatch. Look for a log group named after your ECR repository (e.g., `/aws/docker/pricing-experiments`) to view the real-time output of `run_pricing_experiment.py`.
3.  **S3 Results:** The experiment results are saved to the specified S3 bucket. A `_SUCCESS` file mechanism is planned to be implemented to signal the successful completion of an experiment. You can monitor the target S3 path to see the output files as they are generated.

## 5. Extending the Framework

The framework is designed to be extensible. Here’s a brief guide to adding a new pricing method:

1.  **Implement the Logic:** Create a new Python class for your pricing method, inheriting from a base class (if one exists) in the `pricing_logic.py` or a similar module.
2.  **Integrate into the Runner:** In `run_pricing_experiment.py`, import your new class and add it to the dictionary of available methods.
3.  **Update the Launcher:** Add your new method's name to the list of choices for the `--method` argument in `scripts/launch_ec2_experiments.sh`.
4.  **Rebuild the Image:** The next time you run an experiment, the launcher will automatically rebuild the Docker image with your new code. 