# Rideshare Pricing Optimization Experiment Platform

## 🎯 **What is This?**

A **comprehensive cloud-based research platform** for rideshare pricing optimization experiments. Built with AWS Lambda and Python, implementing the **Hikima et al. methodology** using real NYC taxi data.

**🚀 Quick Start**: Jump straight to [EXPERIMENT_USAGE_GUIDE.md](EXPERIMENT_USAGE_GUIDE.md) for practical examples!

## 📚 **Documentation Structure**

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** | Complete system overview, architecture, AWS setup | Understanding the full system |
| **[EXPERIMENT_USAGE_GUIDE.md](EXPERIMENT_USAGE_GUIDE.md)** | Practical usage examples, API calls, best practices | Running experiments |
| **[README.md](README.md)** | This overview and quick start | Starting point |

## 🚀 **Quick Start Examples**

### **Deploy the System**
```bash
# Deploy both Lambda functions (data ingestion + experiment runner)
./deploy_lambdas.sh all
```

### **Run Standard Experiment**
```bash
# Basic business hours experiment (10:00-20:00)
./run_experiment.sh run-comparative green 2019 3 PL 5
```

### **Run 24-Hour Experiment**
```bash
# Full day analysis with all time periods
./run_experiment.sh run-experiment-24h 30m Manhattan 30s 10 6 2019 green "hikima,maps" PL
```

### **Via Direct API**
```bash
# 24-hour experiment via Lambda API
aws lambda invoke \
  --function-name rideshare-experiment-runner \
  --payload '{
    "vehicle_type": "green",
    "year": 2019,
    "month": 3,
    "full_day_experiment": true,
    "simulation_range": 6,
    "acceptance_function": "PL"
  }' \
  response.json
```

## 🔬 **Research Capabilities**

### **Experiment Types**
- **Standard Business Hours** (10:00-20:00, Hikima paper standard)
- **24-Hour Analysis** (00:00-23:59, full day patterns)
- **Multi-Day Studies** (weekly patterns, seasonal analysis)
- **Geographic Comparison** (Manhattan vs outer boroughs)

### **Algorithm Methods**
- **Hikima**: Min-cost flow optimization (paper method)
- **MAPS**: Area-based pricing approximation
- **LinUCB**: Contextual bandit learning
- **Linear Program**: Mathematical optimization baseline

### **Data Sources**
- **NYC TLC Official Data** (2009-2025)
- **264 Official Taxi Zones** (all boroughs + airports)
- **Yellow/Green/FHV vehicles** (comprehensive coverage)

## ⚡ **AWS Lambda Configuration**

### **Memory Requirements by Experiment Type**
| Experiment Type | Memory | Timeout | Use Case |
|----------------|--------|---------|-----------|
| **Standard** | 512 MB | 5 min | Development, quick tests |
| **24-Hour** | 1024 MB | 10 min | Full day analysis |
| **Multi-Day** | 2048-3008 MB | 15 min | Week/month studies |

### **Main Scripts**
- **`deploy_lambdas.sh`** - Main deployment script (both lambdas)
- **`run_experiment.sh`** - Main experiment runner script
- **`lambdas/experiment-runner/upload_area_info.py`** - Upload taxi zone data

## 📊 **Technical Understanding**

### **Key Concept: Scenarios vs Evaluations**
- **Scenarios** = Different time periods (e.g., 120 scenarios = every 5 min from 10:00-20:00)
- **num_eval** = Monte Carlo evaluations per scenario (100 runs to reduce randomness)
- **Why both**: Test different market conditions (scenarios) with statistical confidence (evaluations)

### **24-Hour Time Periods**
- **Night (00:00-06:00)**: Low demand, limited supply
- **Early Morning (06:00-10:00)**: Commuter patterns
- **Morning Rush (10:00-14:00)**: High business demand
- **Afternoon (14:00-18:00)**: Steady demand
- **Evening Rush (18:00-22:00)**: Peak demand/supply mismatch
- **Late Night (22:00-24:00)**: Entertainment/nightlife

## 🌍 **Real-World Applications**

### **Academic Research**
- Algorithm performance comparison
- Geographic demand modeling
- Temporal pattern analysis
- Parameter sensitivity studies

### **Industry Applications**
- Dynamic pricing optimization
- Demand forecasting
- Driver allocation strategies
- Market expansion planning

### **Policy Research**
- Congestion pricing impact (2025+ data)
- Geographic equity analysis
- Transportation planning
- Economic impact studies

## 🎯 **Expected Results**

### **Typical Performance Ranking**
1. **Hikima**: ~1250 avg objective value (paper method)
2. **MAPS**: ~1180 avg objective value (area-based)
3. **LinUCB**: ~1145 avg objective value (learning-based)

### **Geographic Patterns**
- **Manhattan**: Highest revenue potential, complex patterns
- **Brooklyn/Queens**: Moderate demand, suburban patterns
- **Bronx**: Lower density, different optimization challenges

## 🔧 **Development Workflow**

### **1. Start Small**
```bash
# Test with 2-hour window, single method
./run_experiment.sh run-experiment 10 12 30m Manhattan 30s 10 6 2019 green "hikima" PL
```

### **2. Scale Gradually**
```bash
# Add more methods
./run_experiment.sh run-experiment 10 20 5m Manhattan 30s 10 6 2019 green "hikima,maps,linucb" PL
```

### **3. Go Full Scale**
```bash
# 24-hour multi-day analysis
./run_experiment.sh run-multi-month 10 20 5m Manhattan 30s "3,4,5" "6,10" 2019 green "hikima,maps" PL
```

## 📈 **Data & Performance**

### **NYC TLC Data Characteristics**
| Year | File Size | Reason |
|------|-----------|--------|
| **2009** | 500MB-1GB | Legacy format, quality issues |
| **2019** | 50-150MB | Optimized Parquet, clean data |
| **2024+** | 80-200MB | Enhanced data + congestion pricing |

### **Processing Performance**
- **Single day**: 2-3 minutes
- **24-hour analysis**: 5-8 minutes  
- **Multi-day (week)**: 10-15 minutes
- **Full month**: 15+ minutes

## 🔗 **Key Links**

- **NYC TLC Official Data**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **AWS Lambda Docs**: https://docs.aws.amazon.com/lambda/
- **Original Hikima Paper**: Dynamic pricing for ride-hailing platforms (AAAI)

## 🎉 **Getting Started**

1. **📖 Read**: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for system overview
2. **🚀 Deploy**: Run `./deploy_lambdas.sh all` to deploy system
3. **🧪 Experiment**: Follow examples in [EXPERIMENT_USAGE_GUIDE.md](EXPERIMENT_USAGE_GUIDE.md)
4. **📊 Analyze**: Results auto-generated with S3 URLs

---

**🏆 This platform provides research-grade rideshare pricing optimization experiments using real NYC data with full scientific rigor and AWS cloud scalability!**
