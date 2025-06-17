#!/bin/bash
# Comprehensive Multi-Dataset Experiment Runner
# Supports various common scenarios for running experiments across multiple datasets

set -e  # Exit on error

echo "🚕 Multi-Dataset Experiment Runner"
echo "=================================="

# Default parameters
VEHICLE_TYPES="yellow green"
YEARS="2019"
MONTHS="10"
DAYS="6"
EXPERIMENT_TYPES="PL Sigmoid"
BOROUGHS="Manhattan"
MAX_WORKERS=4
SIMULATION_RANGE=5
TIME_INTERVAL=30
TIME_UNIT="s"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --vehicle-types)
            VEHICLE_TYPES="$2"
            shift 2
            ;;
        --years)
            YEARS="$2"
            shift 2
            ;;
        --months)
            MONTHS="$2"
            shift 2
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --experiment-types)
            EXPERIMENT_TYPES="$2"
            shift 2
            ;;
        --boroughs)
            BOROUGHS="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --simulation-range)
            SIMULATION_RANGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --scenario SCENARIO         Predefined scenario (quick, comprehensive, comparison)"
            echo "  --vehicle-types TYPES       Space-separated vehicle types (yellow green fhv fhvhv)"
            echo "  --years YEARS              Space-separated years (2018 2019 2020)"
            echo "  --months MONTHS            Space-separated months (10 11 12)"
            echo "  --days DAYS                Space-separated days (6 7 8)"
            echo "  --experiment-types TYPES   Space-separated experiment types (PL Sigmoid)"
            echo "  --boroughs BOROUGHS        Space-separated boroughs (Manhattan Queens Bronx Brooklyn)"
            echo "  --max-workers N            Number of parallel workers (default: 4)"
            echo "  --simulation-range N       Number of simulation iterations (default: 5)"
            echo "  --dry-run                  Show what would run without executing"
            echo "  --help                     Show this help"
            echo ""
            echo "Predefined scenarios:"
            echo "  quick         - Fast test across 2 vehicle types, 1 month, minimal iterations"
            echo "  comprehensive - Full comparison across all vehicle types and multiple time periods"
            echo "  comparison    - Compare different acceptance functions across vehicle types"
            echo ""
            echo "Examples:"
            echo "  $0 --scenario quick"
            echo "  $0 --scenario comprehensive --max-workers 8"
            echo "  $0 --vehicle-types \"yellow green\" --years \"2019 2020\" --months \"10 11\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Apply predefined scenarios
case "$SCENARIO" in
    quick)
        VEHICLE_TYPES="yellow green"
        YEARS="2019"
        MONTHS="10"
        DAYS="6"
        EXPERIMENT_TYPES="PL"
        BOROUGHS="Manhattan"
        SIMULATION_RANGE=3
        MAX_WORKERS=2
        echo "📋 Quick Test Scenario"
        ;;
    comprehensive)
        VEHICLE_TYPES="yellow green fhv"
        YEARS="2019 2020"
        MONTHS="10 11"
        DAYS="6 7"
        EXPERIMENT_TYPES="PL Sigmoid"
        BOROUGHS="Manhattan Queens"
        SIMULATION_RANGE=10
        MAX_WORKERS=6
        echo "📋 Comprehensive Analysis Scenario"
        ;;
    comparison)
        VEHICLE_TYPES="yellow green fhv fhvhv"
        YEARS="2019"
        MONTHS="10"
        DAYS="6"
        EXPERIMENT_TYPES="PL Sigmoid"
        BOROUGHS="Manhattan"
        SIMULATION_RANGE=8
        MAX_WORKERS=4
        echo "📋 Vehicle Type Comparison Scenario"
        ;;
    *)
        if [[ -n "$SCENARIO" ]]; then
            echo "❌ Unknown scenario: $SCENARIO"
            exit 1
        fi
        echo "📋 Custom Configuration"
        ;;
esac

echo "Configuration:"
echo "  Vehicle types: $VEHICLE_TYPES"
echo "  Years: $YEARS"
echo "  Months: $MONTHS"
echo "  Days: $DAYS"
echo "  Experiment types: $EXPERIMENT_TYPES"
echo "  Boroughs: $BOROUGHS"
echo "  Max workers: $MAX_WORKERS"
echo "  Simulation range: $SIMULATION_RANGE"
echo ""

# Check if data exists locally
echo "🔍 Checking local data files..."

# Simple check for basic files that should exist
if [[ -f "data/yellow_tripdata_2019-10.parquet" && -f "data/green_tripdata_2019-10.parquet" ]]; then
    echo "✅ Found existing data files for yellow and green taxi Oct 2019"
else
    echo "❌ Missing basic data files. Expected:"
    echo "   data/yellow_tripdata_2019-10.parquet"
    echo "   data/green_tripdata_2019-10.parquet"
    echo ""
    echo "💡 Download data first:"
    echo "   python3 setup_multi.py --vehicle-types yellow green --years 2019 --months 10"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ Data availability check failed"
    echo "💡 Try downloading data first:"
    echo "   python3 setup_multi.py --vehicle-types $VEHICLE_TYPES --years $YEARS --months $MONTHS"
    exit 1
fi

# Estimate experiment count
VEHICLE_COUNT=$(echo $VEHICLE_TYPES | wc -w)
YEAR_COUNT=$(echo $YEARS | wc -w)
MONTH_COUNT=$(echo $MONTHS | wc -w)
DAY_COUNT=$(echo $DAYS | wc -w)
EXP_TYPE_COUNT=$(echo $EXPERIMENT_TYPES | wc -w)
BOROUGH_COUNT=$(echo $BOROUGHS | wc -w)

TOTAL_EXPERIMENTS=$((VEHICLE_COUNT * YEAR_COUNT * MONTH_COUNT * DAY_COUNT * EXP_TYPE_COUNT * BOROUGH_COUNT))

echo ""
echo "📊 Experiment Estimate:"
echo "  Total configurations: $TOTAL_EXPERIMENTS"
echo "  Estimated time: $((TOTAL_EXPERIMENTS * SIMULATION_RANGE / MAX_WORKERS)) minutes (rough)"

if [[ $TOTAL_EXPERIMENTS -gt 100 ]]; then
    echo ""
    echo "⚠️  Large number of experiments detected!"
    echo "   Consider using --dry-run first to review configurations"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled by user"
        exit 0
    fi
fi

# Run experiments
echo ""
echo "🚀 Starting parallel experiments..."

python3 run_experiments_parallel.py \
    --vehicle-types $VEHICLE_TYPES \
    --years $YEARS \
    --months $MONTHS \
    --days $DAYS \
    --experiment-types $EXPERIMENT_TYPES \
    --boroughs $BOROUGHS \
    --max-workers $MAX_WORKERS \
    --simulation-range $SIMULATION_RANGE \
    --time-interval $TIME_INTERVAL \
    --time-unit $TIME_UNIT \
    $DRY_RUN

if [[ $? -eq 0 && -z "$DRY_RUN" ]]; then
    echo ""
    echo "🎉 Experiments completed successfully!"
    echo ""
    echo "📊 Next steps:"
    echo "  1. Aggregate results: python3 aggregate_results.py --create-plots"
    echo "  2. View summaries in: results/aggregated/"
    echo "  3. Check detailed results in: results/detailed/"
else
    echo ""
    if [[ -n "$DRY_RUN" ]]; then
        echo "✅ Dry run completed - configurations look good!"
        echo "💡 Remove --dry-run to execute experiments"
    else
        echo "❌ Experiments failed - check the logs for details"
    fi
fi 