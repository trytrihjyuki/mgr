# 📁 Taxi Pricing Benchmark - Project Structure

## 🎯 **Refactored System Overview**

This project provides a **unified CLI interface** for systematic benchmarking of 4 taxi pricing methods. The system has been completely refactored to eliminate hardcoded values and provide an intuitive command-line experience.

## 🚀 **Quick Start Examples**

```bash
# Test all methods for full year 2019 with both acceptance functions
python cli.py --methods=-1 --days=-1 --months=-1 --year=2019 --func=PL,Sigmoid --start-hour=0 --end-hour=24 --window=5m

# Hikima replication setup (2 specific days in October)  
python cli.py --methods=-1 --days=1,6 --month=10 --year=2019 --func=PL,Sigmoid --start-hour=10 --end-hour=20 --window=5m --location=Manhattan

# Quick test with 2 methods
python cli.py --methods=hikima,maps --days=1 --month=10 --year=2019 --func=PL --requesters=50 --taxis=40

# Extended analysis for Q1 2019
python cli.py --methods=-1 --days=-1 --months=1,2,3 --year=2019 --func=Sigmoid --window=30m

# Brooklyn-specific analysis  
python cli.py --methods=maps,lp --days=1-7 --month=10 --year=2019 --func=PL --location=Brooklyn
```

## 📂 **Directory Structure**

```
📁 mgr/
├── 🎛️ cli.py                     # Unified CLI interface (MAIN ENTRY POINT)
├── 📋 requirements.txt            # Python dependencies
├── 📖 README.md                  # Main documentation
├── 📊 PROJECT_STRUCTURE.md       # This file
│
├── ⚙️ config/                    # Configuration system
│   └── experiment_config.py      # Dataclass-based config framework
│
├── 🧠 src/                       # Core implementation
│   ├── __init__.py
│   ├── orchestrator.py           # Benchmark orchestrator
│   └── pricing_methods/          # All 4 pricing methods
│       ├── __init__.py
│       ├── hikima_method.py      # Hikima MinMax Cost Flow
│       ├── maps_method.py        # Multi-Area Pricing Strategy
│       ├── linucb_method.py      # Linear Upper Confidence Bound
│       └── linear_program_method.py # Linear Programming (Gupta-Nagarajan)
│
├── 📊 results/                   # Generated experiment results
│   ├── *.json                   # Individual experiment results
│   └── SUMMARY_*.json           # Multi-experiment summaries
│
└── 📋 configs/                   # Example configurations (optional)
    ├── hikima_replication.json  # Hikima paper replication
    ├── extended_benchmark.json  # Extended benchmarking
    └── default.json             # Default configuration
```

## 🛠️ **CLI Interface**

### **Core Parameters**
- `--methods`: Methods to run (shortcuts: `hikima,maps,linucb,lp` or `-1` for all)
- `--days`: Days to run (`1,6` or `1-7` or `-1` for all days in month)
- `--month`: Month to run (1-12, default: 10)
- `--months`: Multiple months (`1,2,3` or `1-6` or `-1` for all, overrides `--month`)
- `--year`: Year to run (default: 2019)
- `--func`: Acceptance functions (`PL,Sigmoid`, default: PL)

### **Time Configuration**
- `--start-hour`: Start hour (0-23, default: 10)
- `--end-hour`: End hour (1-24, default: 20)
- `--window`: Time window (`5m`, `30s`, `1h`, default: 5m)

### **Data Parameters**
- `--requesters`: Number of requesters to simulate (default: 200)
- `--taxis`: Number of taxis to simulate (default: 150)
- `--location`: Geographic filter (`Manhattan`, `Brooklyn`, `Queens`, `Bronx`, `Staten_Island`)

### **Utility Commands**
- `--list-methods`: List available pricing methods
- `--validate CONFIG_FILE`: Validate configuration file  
- `--create-examples`: Create example configuration files

## 🔬 **4 Pricing Methods**

| Method | Shortcut | Description | Source |
|--------|----------|-------------|--------|
| **HikimaMinMaxCostFlow** | `hikima` | MinMax Cost Flow method | Hikima paper |
| **MAPS** | `maps` | Multi-Area Pricing Strategy | Extracted from sources |
| **LinUCB** | `linucb` | Linear Upper Confidence Bound | Extracted from sources |
| **LinearProgram** | `lp` | Gupta-Nagarajan LP formulation | Academic literature |

## 📈 **Results & Analysis**

### **Individual Results**
Each experiment generates a JSON file with:
- Experiment metadata (date, methods, parameters)
- Objective values for each method
- Computation times
- Configuration details

### **Summary Reports**
Multi-experiment runs generate summary files with:
- **Performance statistics** (mean, std, min/max by method)
- **Detailed results** for each individual experiment
- **Comparative analysis** across all methods

### **Sample Results** (20 requesters, 15 taxis, PL function):
```
Method               | Objective | Time
---------------------|-----------|-------
LinUCB              |    94.80  | 0.009s  ⭐ BEST
HikimaMinMaxCostFlow|   214.16  | 0.001s  ⚡ FASTEST  
LinearProgram       |   523.64  | 0.115s
MAPS                |   541.87  | 0.002s
```

## 🌐 **Geographic Filtering**

The system supports NYC borough-specific analysis:
- **Manhattan**: Taxi zones 1-68
- **Brooklyn**: Taxi zones 70-158  
- **Queens**: Taxi zones 160-228
- **Bronx**: Taxi zones 230-253
- **Staten Island**: Taxi zones 255-263

## ⚙️ **Configuration System**

- **Zero hardcoded values**: Everything is configurable
- **Dataclass-based**: Type-safe configuration with validation
- **JSON export/import**: Save and load configurations
- **Pre-built configs**: Hikima replication, extended benchmark, custom setups

## 🎯 **Key Features**

✅ **Unified CLI**: Single command interface with intelligent defaults  
✅ **Shorthand notation**: `-1` for "all", comma-separated lists, ranges  
✅ **Geographic filtering**: NYC borough-specific analysis  
✅ **Time flexibility**: From minutes to years of analysis  
✅ **Parallel execution**: All methods run simultaneously  
✅ **Rich reporting**: JSON results with comparative statistics  
✅ **Academic compliance**: Exact Hikima methodology replication  
✅ **Cloud-ready**: S3/Lambda integration (optional)  

## 📚 **Usage Patterns**

### **Research Scenarios**
1. **Hikima Replication**: `--methods=-1 --days=1,6 --month=10 --year=2019 --func=PL,Sigmoid --start-hour=10 --end-hour=20`
2. **Extended Benchmark**: `--methods=-1 --days=-1 --months=-1 --year=2019 --func=PL,Sigmoid`
3. **Geographic Analysis**: `--methods=-1 --location=Manhattan --days=1-7`
4. **Method Comparison**: `--methods=hikima,lp --days=1-30 --func=PL`
5. **Quick Testing**: `--methods=maps,linucb --days=1 --requesters=50 --taxis=40`

### **Time Ranges**
- **Hikima Setup**: 2 days, 10:00-20:00, 5-minute windows
- **Daily Analysis**: 24-hour coverage, configurable windows
- **Weekly Studies**: 7-day ranges with statistical analysis
- **Monthly Research**: Full month analysis with trend detection
- **Annual Studies**: Year-long comparative benchmarking

This system enables seamless scaling from Hikima's original 2-day setup to comprehensive 100+ day research studies with full configurability and academic rigor. 