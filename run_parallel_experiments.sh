#!/bin/bash

# Parallel Experiment Runner for Ride-Hailing Pricing
# Distributes days across 5 parallel processes using modulo arithmetic

set -e

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE PARAMETERS
# =============================================================================

# Basic experiment parameters
YEAR=2019
MONTH=10
TOTAL_DAYS=31  # Number of days to process (adjust for month)
HOURS="8,9,10,11,12,13,14,15,16,17,18,19,20,21,22"
WINDOW=5
ACCEPTANCE_FUNC="PL"
METHODS="MinMaxCostFlow,MAPS,LinUCB,LP"
VEHICLE_TYPE="green"
NUM_EVAL=1000

# Output configuration
OUTPUT_DIR="parallel_experiments_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_DIR/logs"

# =============================================================================
# SCRIPT LOGIC - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
# =============================================================================

# Function to display usage
usage() {
    echo "Usage: $0 [--days_modulo=DIVISOR,REMAINDER]"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run all 5 parallel experiments"
    echo "  $0 --days_modulo=5,1      # Run only process 1 (days 1,6,11,16,21,26,31)"
    echo "  $0 --days_modulo=5,2      # Run only process 2 (days 2,7,12,17,22,27)"
    echo "  $0 --days_modulo=5,3      # Run only process 3 (days 3,8,13,18,23,28)"
    echo "  $0 --days_modulo=5,4      # Run only process 4 (days 4,9,14,19,24,29)"
    echo "  $0 --days_modulo=5,0      # Run only process 5 (days 5,10,15,20,25,30)"
    echo ""
    echo "Configuration (edit at top of script):"
    echo "  Year: $YEAR"
    echo "  Month: $MONTH"
    echo "  Total Days: $TOTAL_DAYS"
    echo "  Hours: $HOURS"
    echo "  Window: $WINDOW minutes"
    echo "  Function: $ACCEPTANCE_FUNC"
    echo "  Methods: $METHODS"
    echo "  Vehicle Type: $VEHICLE_TYPE"
    echo "  Num Eval: $NUM_EVAL"
    echo "  Output Dir: $OUTPUT_DIR"
    exit 1
}

# Parse command line arguments
DAYS_MODULO=""
SINGLE_PROCESS=false

for arg in "$@"; do
    case $arg in
        --days_modulo=*)
            DAYS_MODULO="${arg#*=}"
            SINGLE_PROCESS=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            ;;
    esac
done

# Function to get days for a specific modulo
get_days_for_modulo() {
    local divisor=$1
    local remainder=$2
    local total_days=$3
    local days=()
    
    for ((day=1; day<=total_days; day++)); do
        if (( day % divisor == remainder )); then
            days+=($day)
        fi
    done
    
    echo "${days[@]}"
}

# Function to run a single experiment process
run_experiment_process() {
    local process_id=$1
    local remainder=$2
    local days_list=$3
    local log_file="$LOG_DIR/process_${process_id}.log"
    
    echo "🚀 Starting Process $process_id (remainder=$remainder) - Days: $days_list"
    echo "📝 Log file: $log_file"
    
    # Convert days list to comma-separated format
    local days_csv=$(echo $days_list | tr ' ' ',')
    
    # Run the experiment
    python run_pricing_experiment.py \
        --year=$YEAR \
        --month=$MONTH \
        --days=$days_csv \
        --hours=$HOURS \
        --window=$WINDOW \
        --func=$ACCEPTANCE_FUNC \
        --methods=$METHODS \
        --vehicle_type=$VEHICLE_TYPE \
        --num_eval=$NUM_EVAL \
        --output_dir="$OUTPUT_DIR/process_$process_id" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Process $process_id completed successfully"
    else
        echo "❌ Process $process_id failed with exit code $exit_code"
        echo "📋 Check log: $log_file"
    fi
    
    return $exit_code
}

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Save experiment configuration
cat > "$OUTPUT_DIR/experiment_config.txt" << EOF
Parallel Experiment Configuration
Generated: $(date)

Year: $YEAR
Month: $MONTH
Total Days: $TOTAL_DAYS
Hours: $HOURS
Window: $WINDOW minutes
Acceptance Function: $ACCEPTANCE_FUNC
Methods: $METHODS
Vehicle Type: $VEHICLE_TYPE
Num Eval: $NUM_EVAL

Output Directory: $OUTPUT_DIR
Log Directory: $LOG_DIR
EOF

echo "🎯 Parallel Ride-Hailing Pricing Experiments"
echo "📅 Period: $YEAR-$(printf "%02d" $MONTH) (Days 1-$TOTAL_DAYS)"
echo "⏰ Hours: $HOURS"
echo "🔧 Methods: $METHODS"
echo "📊 Evaluations: $NUM_EVAL per scenario"
echo "📁 Output: $OUTPUT_DIR"
echo ""

if [ "$SINGLE_PROCESS" = true ]; then
    # Run single process mode
    IFS=',' read -r divisor remainder <<< "$DAYS_MODULO"
    
    if [[ ! "$divisor" =~ ^[0-9]+$ ]] || [[ ! "$remainder" =~ ^[0-9]+$ ]]; then
        echo "❌ Invalid days_modulo format. Use: --days_modulo=DIVISOR,REMAINDER"
        echo "   Example: --days_modulo=5,1"
        exit 1
    fi
    
    if (( remainder >= divisor )); then
        echo "❌ Remainder ($remainder) must be less than divisor ($divisor)"
        exit 1
    fi
    
    days_list=$(get_days_for_modulo $divisor $remainder $TOTAL_DAYS)
    
    if [ -z "$days_list" ]; then
        echo "❌ No days found for modulo $divisor with remainder $remainder"
        exit 1
    fi
    
    echo "🎯 Single Process Mode: Divisor=$divisor, Remainder=$remainder"
    echo "📅 Processing days: $days_list"
    echo ""
    
    run_experiment_process 1 $remainder "$days_list"
    
else
    # Run all 5 processes in parallel
    echo "🔄 Starting 5 parallel processes..."
    echo ""
    
    # Start background processes
    pids=()
    
    for remainder in {1..5}; do
        # Adjust remainder for modulo 5 (0,1,2,3,4)
        actual_remainder=$((remainder % 5))
        days_list=$(get_days_for_modulo 5 $actual_remainder $TOTAL_DAYS)
        
        if [ -n "$days_list" ]; then
            run_experiment_process $remainder $actual_remainder "$days_list" &
            pids+=($!)
        else
            echo "⚠️ No days found for process $remainder (remainder=$actual_remainder)"
        fi
    done
    
    echo "⏳ Waiting for all processes to complete..."
    echo "📊 Progress can be monitored in: $LOG_DIR/"
    echo ""
    
    # Wait for all processes and collect results
    failed_processes=()
    
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        process_id=$((i + 1))
        
        if wait $pid; then
            echo "✅ Process $process_id (PID: $pid) completed successfully"
        else
            echo "❌ Process $process_id (PID: $pid) failed"
            failed_processes+=($process_id)
        fi
    done
    
    echo ""
    echo "🎉 All processes completed!"
    
    if [ ${#failed_processes[@]} -eq 0 ]; then
        echo "✅ All 5 processes succeeded"
    else
        echo "❌ Failed processes: ${failed_processes[*]}"
        echo "📋 Check logs in: $LOG_DIR/"
    fi
fi

# Summary
echo ""
echo "📋 Experiment Summary:"
echo "   Configuration: $OUTPUT_DIR/experiment_config.txt"
echo "   Logs: $LOG_DIR/"
echo "   Results: $OUTPUT_DIR/"
echo ""
echo "🔍 To monitor progress:"
echo "   tail -f $LOG_DIR/process_*.log"
echo ""
echo "📊 To check results:"
echo "   ls -la $OUTPUT_DIR/*/experiment_summary.json"

echo ""
echo "🎯 Example commands for manual parallel execution:"
echo "   $0 --days_modulo=5,1  # Terminal 1"
echo "   $0 --days_modulo=5,2  # Terminal 2" 
echo "   $0 --days_modulo=5,3  # Terminal 3"
echo "   $0 --days_modulo=5,4  # Terminal 4"
echo "   $0 --days_modulo=5,0  # Terminal 5" 