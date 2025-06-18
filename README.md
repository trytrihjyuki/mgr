# 🚀 Rideshare Bipartite Matching Optimization

## 📋 **Complete Documentation Available**

**📖 See [`COMPLETE_DOCUMENTATION.md`](./COMPLETE_DOCUMENTATION.md) for the comprehensive guide** that includes:

- ✅ **All parameters explained** with ranges and defaults
- ✅ **Complete JSON output reference** with all KPIs and fields  
- ✅ **Step-by-step examples** for all experiment types
- ✅ **Troubleshooting guide** and data availability info
- ✅ **Algorithm comparison** and performance benchmarks

---

## ⚡ **Quick Start**

```bash
# Essential workflow: Check → Download → Experiment → Analyze
./run_experiment.sh check-availability green 2019 3
./run_experiment.sh download-single green 2019 3  
./run_experiment.sh run-comparative green 2019 3 PL 5
./run_experiment.sh analyze <experiment_id>
```

---

## 🔬 **Four Advanced Algorithms**

| Algorithm | Efficiency | Best For |
|-----------|------------|----------|
| **LINEAR_PROGRAM** | 88-91% | Maximum revenue |
| **PROPOSED** | 85-91% | Balanced performance |
| **MAPS** | 65-77% | Market dynamics |
| **LINUCB** | 75-84% | Learning scenarios |

---

## 📊 **System Overview**

This implements **sophisticated bipartite matching algorithms** for rideshare optimization using real NYC taxi data on **AWS Lambda architecture**.

### **Architecture**
```
NYC TLC Data → Lambda Ingestion → S3 Storage → Lambda Experiments → JSON Results → Local Analysis
```

### **Key Features**
- 🌩️ **Serverless AWS Lambda** - Auto-scaling experiment execution
- 🚕 **Real NYC Taxi Data** - Green, Yellow, FHV vehicle types  
- 📊 **Four Algorithms** - Comparative analysis and ranking
- 🎯 **Acceptance Functions** - PL (Piecewise Linear) and Sigmoid
- 📈 **Rich Analytics** - Detailed KPIs, statistics, and visualizations

---

## 🎯 **Most Common Commands**

```bash
# Data Management
./run_experiment.sh check-availability <type> <year> <month>    # Check before download
./run_experiment.sh download-bulk 2019 1 3 green,yellow,fhv    # Bulk download

# Run Experiments  
./run_experiment.sh run-comparative green 2019 3 PL 5          # All 4 algorithms
./run_experiment.sh run-single green 2019 3 linear_program     # Single algorithm
./run_experiment.sh parameter-sweep green 2019 3 proposed      # 54 combinations

# Analysis
./run_experiment.sh list-experiments 30                        # Recent experiments
./run_experiment.sh analyze <experiment_id>                    # Detailed analysis
```

---

## 📈 **JSON Output Structure**

Each experiment produces comprehensive JSON with:

### **Core Sections**
- **`parameters`**: All input configuration
- **`method_results`**: Per-algorithm detailed results  
- **`comparative_stats`**: Performance ranking and comparison
- **`execution_times`**: Runtime performance metrics

### **Key KPIs** (per algorithm)
- **`objective_value`**: Primary optimization target
- **`match_rate`**: Success percentage (0-1)
- **`algorithm_efficiency`**: Performance score (0-1)  
- **`total_revenue`**: Monetary value generated
- **`supply_demand_ratio`**: Driver/request balance

**📖 See [`COMPLETE_DOCUMENTATION.md`](./COMPLETE_DOCUMENTATION.md) for full JSON field reference**

---

## ✅ **Data Availability**

| Vehicle Type | Years Available | Recommended |
|--------------|----------------|-------------|
| **Green** | 2017-2023 ✅ | All years work |
| **Yellow** | 2018-2023 ✅ | 2019-2023 optimal |
| **FHV** | 2018-2023 ✅ | 2019-2023 optimal |

**⚠️ Known Issues:**
- Yellow taxi 2017 May-Dec: ❌ Not available
- FHV 2017: ❌ Not available  
- **Solution**: Use years 2018-2023 for complete coverage

---

## 📝 **Research Foundation**

Based on **AAAI 2021 paper**:
*"Integrated Optimization for Bipartite Matching and Its Stochastic Behavior: New Formulation and Approximation Algorithm via Min-cost Flow Optimization"*

**Authors**: Yuya Hikima, Yasunori Akagi, Hideaki Kim, Masahiro Kohjima, Takeshi Kurashima, Hiroyuki Toda

---

**🎉 For complete parameter explanations, JSON field definitions, troubleshooting, and advanced usage examples, see [`COMPLETE_DOCUMENTATION.md`](./COMPLETE_DOCUMENTATION.md)**
