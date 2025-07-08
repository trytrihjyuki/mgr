# Ride-Hailing Pricing Experiment Framework

> **2024-07 Update** – The project has been streamlined to use a flexible EC2-based architecture.  
> The `scripts/launch_ec2_experiments.sh` script is the main entry point for running experiments.

## 🚀 Quick Start

The primary way to run experiments is through the `scripts/launch_ec2_experiments.sh` script. It provides a wide range of options to configure your experiment and launches the necessary cloud infrastructure.

**Note:** You will need to configure your AWS credentials and some basic infrastructure parameters (like `SUBNET_ID`, `KEY_NAME`, etc.) at the top of the script before the first run.

Here are some one-liner commands to help you find these values. It's recommended to use resources specifically created for this project, which may be tagged with a name like `pricing-experiment`.

- **`SUBNET_ID`**: Find subnets in your default VPC.
  ```sh
  aws ec2 describe-subnets --filters "Name=vpc-id,Values=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)" --query "Subnets[].SubnetId" --output text
  ```
- **`SECURITY_GROUP_IDS`**: Find security groups, preferably one named `default` or `pricing-experiment`.
  ```sh
  aws ec2 describe-security-groups --query "SecurityGroups[?GroupName=='default' || contains(GroupName, 'pricing')].GroupId" --output text
  ```
- **`KEY_NAME`**: List your available EC2 key pairs.
  ```sh
  aws ec2 describe-key-pairs --query "KeyPairs[].KeyName" --output text
  ```
- **`IAM_INSTANCE_PROFILE`**: Find instance profiles with "Pricing" in their name.
  ```sh
  aws iam list-instance-profiles --query "InstanceProfiles[?contains(InstanceProfileName, 'Pricing')].InstanceProfileName" --output text
  ```

### Example 1: Basic Sanity Check

This command launches a small EC2 instance to run the LinUCB method for a single day on a small sample of green taxi data.

```bash
# Launch a small, one-day experiment
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-06 \
  --end-date 2019-10-06 \
  --ec2-type small
```

### Example 2: Comprehensive Weekly Experiment

This command runs a larger experiment over a week, using a more powerful EC2 instance. It tests the Linear Programming method on yellow taxi data in Manhattan.

```bash
# Launch a week-long experiment on a medium instance
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-01 \
  --end-date 2019-10-07 \
  --method LP \
  --vehicle-type yellow \
  --borough Manhattan \
  --ec2-type medium
```

### Example 3: Multi-Iteration Monte Carlo

This example demonstrates how to run a Monte Carlo simulation with a specific number of iterations and a custom seed.

```bash
# Run a Monte Carlo simulation with 5000 iterations
./scripts/launch_ec2_experiments.sh \
  --start-date 2019-10-15 \
  --end-date 2019-10-15 \
  --num-iter 5000 \
  --seed 123 \
  --ec2-type large
```

After the container exits, the instance will shut itself down. Remember to **terminate** it in the EC2 console to stop billing.

---

## Core Features

### **Pricing Algorithms**
- **LP**: Gupta-Nagarajan Linear Program optimization
- **MinMaxCostFlow**: Capacity scaling min-cost flow algorithm  
- **LinUCB**: Contextual bandit learning with pre-trained models
- **MAPS**: Area-based pricing with bipartite matching

### **Acceptance Functions**
- **PL**: Piecewise Linear (`acceptance = -2.0/trip_amount * price + 3.0`)
- **Sigmoid**: Sigmoid function with Hikima parameters (`β=1.3`, `γ=0.3*√3/π`)

### **Vehicle Types**
- **yellow**: Yellow taxi data (largest dataset)
- **green**: Green taxi data (outer boroughs)
- **fhv**: For-hire vehicle data

### **Boroughs**
- **Manhattan**: Highest density, uses 30s time intervals in Hikima
- **Bronx/Queens/Brooklyn**: Lower density, uses 300s time intervals in Hikima

## Script Arguments

The `scripts/launch_ec2_experiments.sh` script accepts the following arguments:

| Argument                | Description                                                 | Default      |
|-------------------------|-------------------------------------------------------------|--------------|
| `--start-date`          | Start date for the experiment (YYYY-MM-DD)                  | Yesterday    |
| `--end-date`            | End date for the experiment (YYYY-MM-DD)                    | Today        |
| `--start-hour`          | Start hour (0-23)                                           | 0            |
| `--end-hour`            | End hour (0-23)                                             | 23           |
| `--borough`             | NYC Borough                                                 | Manhattan    |
| `--vehicle-type`        | Taxi type: `green`, `yellow`, `fhv`                           | `green`      |
| `--method`              | Pricing method: `LinUCB`, `LP`, etc.                          | `LinUCB`     |
| `--acceptance-function` | Acceptance function: `PL`, `Sigmoid`                        | `PL`         |
| `--num-iter`            | Number of Monte Carlo iterations                            | 1000         |
| `--num-parallel`        | Number of parallel jobs within the container                | 4            |
| `--ec2-type`            | `small`, `medium`, `large`, `xlarge`, `extra-large`           | `small`      |
| `--seed`                | Random seed                                                 | 42           |

## Results Structure

Results are saved to S3 with a structured path:
```
s3://magisterka/experiments/
  type={vehicle_type}/
    year={year}/month={month}/day={day}/
      {training_id}.json
```

Each file contains detailed results from the experiment run, including performance metrics for each pricing method and scenario.

## See Also

- **TECHNICAL.md**: Detailed setup, validation, and troubleshooting guide.
- **`run_pricing_experiment.py`**: The core Python script that runs inside the Docker container.
- **`aws_ec2_launcher.py`**: The Python script responsible for the cloud infrastructure automation. 