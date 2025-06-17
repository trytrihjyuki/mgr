# Practical Usage Examples

## 🎯 **Easy Method: Use Helper Script (Recommended)**

The `run_experiment.sh` script eliminates JSON parsing issues and provides a clean interface:

```bash
# Make executable (one time)
chmod +x run_experiment.sh

# Show all available commands
./run_experiment.sh

# Examples
./run_experiment.sh download-single green 2019 3
./run_experiment.sh download-bulk 2019 1 3 green,yellow
./run_experiment.sh run-experiment green 2019 3 5 PL
./run_experiment.sh list-data
./run_experiment.sh list-experiments 7
```

## 🔍 Understanding Lambda Responses vs S3 Data

When you invoke a Lambda function, there are **two different things**:

1. **Lambda Response** - Just the function's HTTP response (properly formatted by helper script)
2. **Actual Data** - Stored directly in S3

### Helper Script Output Example
```
🧪 Running experiment: green taxi 2019-03 with 5 scenarios...
📄 Response:
{
    "statusCode": 200,
    "body": {
        "experiment_id": "rideshare_green_2019_03_20250617_115920",
        "total_requests": 32930,
        "total_successful_matches": 19721,
        "average_match_rate": 0.599
    }
}
```

### Actual Data Location
```
s3://magisterka/datasets/green/year=2019/month=03/green_tripdata_2019-03.parquet
s3://magisterka/experiments/results/rideshare/rideshare_green_2019_03_20250617_115920_results.json
```

## 📊 Complete Workflow Examples

### Example 1: Download and Experiment with Green Taxi Data (Using Helper Script)

```bash
# 1. Download green taxi data for March 2019
./run_experiment.sh download-single green 2019 3

# 2. Verify data was uploaded to S3
./run_experiment.sh list-data

# 3. Run bipartite matching experiment
./run_experiment.sh run-experiment green 2019 3 5 PL

# 4. View all recent experiments
./run_experiment.sh list-experiments 1
```

### Example 1b: Manual AWS CLI Commands (If Helper Script Issues)

```bash
# 1. Download data (single-line JSON, no parsing issues)
/usr/local/bin/aws lambda invoke \
  --function-name nyc-data-ingestion \
  --payload '{"action":"download_single","vehicle_type":"green","year":2019,"month":3}' \
  --region eu-north-1 \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json | python3 -m json.tool

# 2. Run experiment (single-line JSON)
/usr/local/bin/aws lambda invoke \
  --function-name rideshare-experiment-runner \
  --payload '{"vehicle_type":"green","year":2019,"month":3,"simulation_range":5,"acceptance_function":"PL"}' \
  --region eu-north-1 \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json | python3 -m json.tool
```

### Example 2: Using Local Results Manager

```bash
# Instead of parsing JSON responses, use the local manager
source venv/bin/activate

# List all recent experiments
python local-manager/results_manager.py list --days 1

# Generate a comprehensive report
python local-manager/results_manager.py report --days 1 --output today_report.txt

# Show specific experiment details
python local-manager/results_manager.py show rideshare_green_2019_03_20241217_143022
```

### Example 3: Bulk Data Processing

```bash
# Download multiple datasets (6 files: green & yellow for Jan-Mar 2019)
./run_experiment.sh download-bulk 2019 1 3 green,yellow

# Check what was downloaded
./run_experiment.sh list-data

# Run experiments on all downloaded data
for vehicle in green yellow; do
  for month in 1 2 3; do
    ./run_experiment.sh run-experiment $vehicle 2019 $month 5 PL
    echo "Started experiment: $vehicle taxi, month $month"
  done
done

# Analyze all results
python local-manager/results_manager.py report --days 1
```

### Example 3b: Manual Bulk Processing (If Needed)

```bash
# Bulk download with manual command
/usr/local/bin/aws lambda invoke \
  --function-name nyc-data-ingestion \
  --payload '{"action":"download_bulk","vehicle_types":["green","yellow"],"year":2019,"start_month":1,"end_month":3}' \
  --region eu-north-1 \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json | python3 -m json.tool
```

## 🎯 Key Points

### ✅ DO:
- Use the **local results manager** to analyze data
- Check **S3 directly** for actual data files
- Parse the **Lambda response body** to get S3 keys
- Use **bulk operations** for efficiency

### ❌ DON'T:
- Expect actual data in `output.json` files
- Try to download data through Lambda responses
- Ignore the local results manager tools

## 🔧 Troubleshooting

### Check if Data Ingestion Worked
```bash
# Check S3 for data
aws s3 ls s3://magisterka/datasets/ --recursive --region eu-north-1

# Or use the existing tools
python aws_s3_manager.py list-datasets
```

### Check if Experiments Completed
```bash
# Check S3 for results
aws s3 ls s3://magisterka/experiments/results/rideshare/ --region eu-north-1

# Or use the local manager
python local-manager/results_manager.py list --days 1
```

### Parse Lambda Response for S3 Keys
```bash
# Extract S3 key from Lambda response
cat experiment_response.json | jq -r '.body' | jq -r '.s3_key'

# Download the actual result file
S3_KEY=$(cat experiment_response.json | jq -r '.body' | jq -r '.s3_key')
aws s3 cp s3://magisterka/$S3_KEY ./result.json --region eu-north-1
```

## 📈 Performance Tips

1. **Use limits for testing**: Start with `"limit": 1000` for quick tests
2. **Bulk operations**: Download multiple datasets at once
3. **Local caching**: The results manager caches data locally
4. **Monitor Lambda logs**: Check CloudWatch for detailed execution logs

```bash
# View Lambda logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/" --region eu-north-1
``` 