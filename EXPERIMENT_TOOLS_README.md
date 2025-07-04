# Enhanced Experiment Tools

This directory contains enhanced tools for running and auditing S3 experiments with improved parallel execution, tracking, and monitoring capabilities.

## 🚀 Quick Start

1. **S3 Audit Script** - Track and audit all experiments in S3
2. **Enhanced Parallel Runner** - Run experiments with better tracking and daily saves
3. **Improved Experiment Script** - Enhanced logging and circuit breaker pattern

## 📋 Available Tools

### 1. S3 Audit Script (`s3_audit_script.py`)

Comprehensive audit tool for tracking all experiments in S3.

**Features:**
- Tracks experiment files and metadata
- Provides daily/weekly summaries
- Monitors experiment health and completion status
- Identifies missing or corrupted files
- Tracks parallel execution progress

**Usage:**
```bash
# List recent experiments
python s3_audit_script.py --list-recent

# Get daily summary
python s3_audit_script.py --daily-summary 2024-01-15

# Health check
python s3_audit_script.py --health-check

# Check parallel execution status
python s3_audit_script.py --parallel-status

# Generate full audit report
python s3_audit_script.py --full-report
```

### 2. Enhanced Parallel Runner (`improved_parallel_runner.py`)

Advanced parallel experiment runner with comprehensive tracking.

**Features:**
- Daily save logic with check-if-saved-then-proceed
- Comprehensive timestamps in all logs
- Status monitoring and health checks
- Recovery from failures
- Progress reporting
- Circuit breaker pattern to prevent spam on broken lambdas

**Usage:**
```bash
# Run with configuration
python improved_parallel_runner.py --config example_experiment_config.json --scenarios scenarios.json

# Resume a previous experiment
python improved_parallel_runner.py --resume experiment_id --scenarios scenarios.json

# Check experiment status
python improved_parallel_runner.py --status

# Dry run to test configuration
python improved_parallel_runner.py --config example_experiment_config.json --scenarios scenarios.json --dry-run
```

### 3. Enhanced Shell Script (`enhanced_parallel_experiments.sh`)

Improved version of the parallel experiments shell script.

**Features:**
- Comprehensive tracking of parallel lambda executions
- Daily save logic with S3 upload
- Detailed timestamps in all logs
- Health monitoring and status tracking
- Recovery from failures
- Progress reporting

**Usage:**
```bash
# Run enhanced parallel experiments
./enhanced_parallel_experiments.sh

# Check logs
tail -f parallel_experiments_*/logs/*.log

# Monitor status
cat parallel_experiments_*/status/*.json

# Check S3 uploads
aws s3 ls s3://magisterka/parallel_experiments/
```

### 4. Improved Main Script (`run_pricing_experiment.py`)

Enhanced with better logging, daily save checking, and circuit breaker pattern.

**New Features:**
- Enhanced logging with timestamps and file output
- Daily save checking (skips already completed days)
- Circuit breaker pattern to prevent spam on broken lambdas
- Better error tracking and classification
- Lambda health monitoring

## 🔧 Configuration

### Example Configuration (`example_experiment_config.json`)

```json
{
    "experiment_name": "enhanced_parallel_pricing_experiment",
    "aws": {
        "bucket": "magisterka",
        "region": "eu-north-1"
    },
    "execution": {
        "max_concurrent": 3,
        "lambda_timeout": 900,
        "max_retries": 3,
        "circuit_breaker": {
            "failure_threshold": 10,
            "timeout_duration": 300
        }
    },
    "experiments": {
        "daily_save_enabled": true,
        "health_check_interval": 300,
        "progress_report_interval": 60,
        "auto_resume": true,
        "skip_completed": true
    }
}
```

## 📊 Daily Save Logic

The enhanced tools implement a "check-if-saved-then-proceed" logic:

1. **Local Cache Check**: Checks local cache files for completed days
2. **S3 Verification**: Verifies results exist in S3
3. **Skip Completed**: Automatically skips already completed experiments
4. **Resume Support**: Can resume interrupted experiments

### Cache Structure
```
daily_cache_2019_10/
├── day_01_PL_saved.json
├── day_01_Sigmoid_saved.json
├── day_02_PL_saved.json
└── ...
```

## 🏥 Health Monitoring

### Circuit Breaker Pattern
- **Closed**: Normal operation
- **Open**: Too many failures, blocks requests
- **Half-Open**: Testing if service recovered

### Health Status
- **Healthy**: Normal operation
- **Degraded**: Some failures but still functional
- **Unhealthy**: Circuit breaker open, service unavailable

## 📈 Progress Tracking

### Real-time Monitoring
- Progress reports every minute
- Health checks every 5 minutes
- Automatic status persistence
- Recovery from interruptions

### Status Files
```
experiment_status_parallel_exp_20240115_120000.json
```

Contains:
- Scenario-level tracking
- Execution times
- Error messages
- S3 locations

## 🚨 Error Handling

### Error Classification
- **timeout**: Lambda timeout errors
- **rate_limit**: AWS rate limiting
- **memory**: Out of memory errors
- **permission**: Access denied errors
- **invalid_request**: Bad request format
- **unknown**: Other errors

### Anti-Spam Features
- Circuit breaker prevents repeated failures
- Error pattern tracking
- Automatic backoff on repeated errors
- Health status monitoring

## 📁 Output Structure

### Local Directory Structure
```
parallel_experiments_20240115_120000/
├── logs/
│   ├── master.log
│   ├── process_1.log
│   └── ...
├── status/
│   ├── day_01_status.json
│   └── ...
├── daily_cache/
│   ├── day_01_PL_saved.json
│   └── ...
└── final_status.json
```

### S3 Structure
```
s3://magisterka/
├── parallel_experiments/
│   └── parallel_exp_20240115_120000/
│       ├── day_01/
│       │   ├── daily_summary.json
│       │   └── process_1.log
│       └── final_status.json
├── daily_summaries/
│   └── parallel_exp_20240115_120000/
│       ├── 2024-01-15_summary.json
│       └── ...
└── experiments/
    └── type=yellow/
        └── eval=pl/
            └── year=2019/
                └── month=10/
                    └── day=01/
                        └── results/
```

## 🛠️ Maintenance

### Cleanup Commands
```bash
# Clean old experiment directories
find . -name "parallel_experiments_*" -type d -mtime +30 -exec rm -rf {} \;

# Clean old logs
find . -name "*.log" -mtime +7 -delete

# Clean daily cache
find . -name "daily_cache_*" -type d -mtime +7 -exec rm -rf {} \;
```

### Monitoring Commands
```bash
# Monitor active experiments
ps aux | grep -E "(python|bash)" | grep -i experiment

# Check disk space
df -h

# Check S3 usage
aws s3api list-objects-v2 --bucket magisterka --prefix experiments/ --query 'Contents[?LastModified>`2024-01-01`]' --output table

# View recent logs
tail -f parallel_experiments_*/logs/master.log
```

## 🔍 Troubleshooting

### Common Issues

1. **Circuit Breaker Open**
   - Check lambda function health
   - Verify AWS permissions
   - Wait for timeout or fix underlying issue

2. **S3 Permission Errors**
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Ensure bucket exists

3. **Lambda Timeouts**
   - Check lambda function configuration
   - Verify memory allocation
   - Check for hanging processes

4. **Daily Save Failures**
   - Check S3 connectivity
   - Verify cache directory permissions
   - Check disk space

### Debug Commands
```bash
# Check lambda health
python -c "from run_pricing_experiment import ExperimentRunner; r = ExperimentRunner(); print(r.get_lambda_health_summary())"

# Test S3 connection
python s3_audit_script.py --bucket magisterka --list-recent

# Check experiment status
python improved_parallel_runner.py --status
```

## 📚 Additional Resources

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [S3 Best Practices](https://docs.aws.amazon.com/s3/latest/userguide/best-practices.html)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

## 🤝 Contributing

1. Test changes with `--dry-run` flag
2. Monitor health metrics during experiments
3. Update configuration examples
4. Add new error classifications as needed
5. Update documentation for new features 