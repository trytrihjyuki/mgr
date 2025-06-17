# Practical Usage Examples

## 🔍 Understanding Lambda Responses vs S3 Data

When you invoke a Lambda function, there are **two different things**:

1. **Lambda Response** (`output.json`) - Just the function's HTTP response
2. **Actual Data** - Stored directly in S3

### Lambda Response Example
```json
{
  "statusCode": 200,
  "body": "{\"status\":\"success\",\"s3_key\":\"datasets/green/year=2019/month=03/green_tripdata_2019-03.csv\",\"size_bytes\":15420}"
}
```

### Actual Data Location
```
s3://magisterka/datasets/green/year=2019/month=03/green_tripdata_2019-03.csv
```

## 📊 Complete Workflow Examples

### Example 1: Download and Experiment with Green Taxi Data

```bash
# 1. Download data (response shows success, data goes to S3)
aws lambda invoke \
  --function-name nyc-data-ingestion \
  --payload '{
    "action": "download_single",
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "limit": 1000
  }' \
  --region eu-north-1 \
  ingestion_response.json

# 2. Check the Lambda response
echo "Lambda Response:"
cat ingestion_response.json

# 3. Verify data was uploaded to S3
echo "S3 Data:"
aws s3 ls s3://magisterka/datasets/green/year=2019/month=03/ --region eu-north-1

# 4. Run experiment (results go to S3)
aws lambda invoke \
  --function-name rideshare-experiment-runner \
  --payload '{
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "simulation_range": 3,
    "acceptance_function": "PL"
  }' \
  --region eu-north-1 \
  experiment_response.json

# 5. Check experiment response
echo "Experiment Response:"
cat experiment_response.json

# 6. View actual results in S3
echo "Experiment Results in S3:"
aws s3 ls s3://magisterka/experiments/results/rideshare/ --region eu-north-1
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
# Download multiple datasets
aws lambda invoke \
  --function-name nyc-data-ingestion \
  --payload '{
    "action": "download_bulk",
    "vehicle_types": ["green", "yellow"],
    "year": 2019,
    "start_month": 1,
    "end_month": 3,
    "limit": 5000
  }' \
  --region eu-north-1 \
  bulk_response.json

# Check what was downloaded
aws s3 ls s3://magisterka/datasets/ --recursive --region eu-north-1

# Run experiments on all downloaded data
for vehicle in green yellow; do
  for month in 1 2 3; do
    aws lambda invoke \
      --function-name rideshare-experiment-runner \
      --payload "{
        \"vehicle_type\": \"$vehicle\",
        \"year\": 2019,
        \"month\": $month,
        \"simulation_range\": 5,
        \"acceptance_function\": \"PL\"
      }" \
      --region eu-north-1 \
      exp_${vehicle}_${month}_response.json
    
    echo "Started experiment: $vehicle taxi, month $month"
  done
done

# Analyze all results
python local-manager/results_manager.py report --days 1
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