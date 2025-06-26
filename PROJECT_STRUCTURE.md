# 📁 Project Structure Overview

## 🎯 **Main Entry Points**

### **🚀 Deployment**
```bash
./deploy_lambdas.sh all    # Deploy entire system (both Lambda functions)
```

### **🧪 Running Experiments**
```bash
./run_experiment.sh run-comparative green 2019 3    # Standard experiment
./run_experiment.sh run-experiment-24h 30m Manhattan 30s 10 6 2019    # 24-hour experiment
```

## 📚 **Documentation (Start Here)**

| File | Purpose | When to Read |
|------|---------|--------------|
| **[docs/README.md](docs/README.md)** | Project overview & quick start | **START HERE** |
| **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)** | Complete system architecture | Understanding the system |
| **[docs/EXPERIMENT_USAGE_GUIDE.md](docs/EXPERIMENT_USAGE_GUIDE.md)** | Practical examples & API usage | Running experiments |

## 🏗️ **Core System Structure**

```
mgr/
├── 🚀 deploy_lambdas.sh              # MAIN DEPLOYMENT SCRIPT
├── 🧪 run_experiment.sh              # MAIN EXPERIMENT RUNNER
├── 📚 docs/                          # Clean documentation (3 files)
├── ⚡ lambdas/
│   ├── data-ingestion/               # NYC data download Lambda
│   └── experiment-runner/            # Rideshare experiment Lambda
├── 🔧 local-manager/                 # Results analysis tools
└── 📄 aws_config.py                  # AWS configuration
```

## 🎯 **Key Lambda Functions**

### **Data Ingestion Lambda** (`lambdas/data-ingestion/`)
- **Purpose**: Download NYC taxi data from official sources
- **Deploy**: `cd lambdas/data-ingestion && ./deploy.sh`
- **Usage**: Via `run_experiment.sh download-single/download-bulk`

### **Experiment Runner Lambda** (`lambdas/experiment-runner/`)
- **Purpose**: Run Hikima-compliant pricing optimization experiments  
- **Deploy**: `cd lambdas/experiment-runner && ./deploy.sh`
- **Usage**: Via `run_experiment.sh` or direct Lambda API calls

## 📊 **Data Flow**

```
1. 📥 Data Download:    run_experiment.sh → data-ingestion Lambda → S3
2. 🧪 Run Experiment:   run_experiment.sh → experiment-runner Lambda → S3  
3. 📊 Get Results:      Auto-analysis with S3 URLs provided
```

## 🔧 **Development Workflow**

### **1. First Time Setup**
```bash
# Deploy system
./deploy_lambdas.sh all

# Test with simple experiment
./run_experiment.sh run-experiment 10 12 30m Manhattan 30s 10 6 2019 green "hikima" PL
```

### **2. Regular Development**
```bash
# Download data
./run_experiment.sh download-single green 2019 3

# Run experiments
./run_experiment.sh run-comparative green 2019 3 PL 5

# 24-hour analysis
./run_experiment.sh run-experiment-24h 30m Manhattan 30s 10 6 2019
```

### **3. Production Research**
```bash
# Multi-day studies
./run_experiment.sh run-multi-month 10 20 5m Manhattan 30s "3,4,5" "6,10" 2019

# Direct API calls
aws lambda invoke --function-name rideshare-experiment-runner --payload '{...}'
```

## 🗂️ **AWS Resources Created**

- **Lambda Functions**: `nyc-data-ingestion`, `rideshare-experiment-runner`
- **IAM Role**: `lambda-execution-role` (with S3 permissions)
- **S3 Bucket**: `magisterka` (configurable in scripts)
- **CloudWatch Logs**: Automatic logging for all Lambda executions

## 🎯 **Experiment Types Supported**

| Type | Command | Purpose |
|------|---------|---------|
| **Standard** | `run-comparative` | Business hours (10:00-20:00) |
| **24-Hour** | `run-experiment-24h` | Full day analysis |
| **Multi-Day** | `run-multi-month` | Weekly/seasonal patterns |
| **Custom** | Direct Lambda API | Full control over parameters |

## 📈 **Results & Analysis**

- **Auto-Analysis**: Results analyzed immediately after experiment completion
- **S3 Storage**: All results stored with clear S3 URLs provided
- **Local Tools**: `local-manager/results_manager.py` for advanced analysis
- **Format**: Clean JSON structure with performance metrics

## 🔗 **External Dependencies**

- **Data Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **AWS Services**: Lambda, S3, IAM, CloudWatch
- **Python Libraries**: pandas, numpy, boto3, geopy
- **Geographic Data**: Official NYC 264 taxi zones

## 🎉 **Quick Navigation**

- **📖 Getting Started**: [docs/README.md](docs/README.md)
- **🏗️ System Architecture**: [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)
- **🧪 Running Experiments**: [docs/EXPERIMENT_USAGE_GUIDE.md](docs/EXPERIMENT_USAGE_GUIDE.md)
- **🚀 Deploy System**: `./deploy_lambdas.sh all`
- **🔬 Run Experiment**: `./run_experiment.sh run-comparative green 2019 3`

---

**🏆 This structure provides a clean, scalable platform for rideshare pricing optimization research with real NYC data!** 