#!/bin/bash

# Enhanced Parallel Experiment Runner for Ride-Hailing Pricing
# Features:
# - Comprehensive tracking of parallel lambda executions
# - Daily save logic with check-if-saved-then-proceed
# - Detailed timestamps in all logs
# - Health monitoring and status tracking
# - Recovery from failures
# - Progress reporting

set -e

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Basic experiment parameters
YEAR=2019
MONTH=10
TOTAL_DAYS=31
WINDOW=5
ACCEPTANCE_FUNC="PL,Sigmoid"
METHODS="LP,MinMaxCostFlow,LinUCB,MAPS"
VEHICLE_TYPE="yellow"
BOROUGH="Manhattan"
NUM_EVAL=20
HOUR_START=0
HOUR_END=24
TIME_INTERVAL=5
TIME_UNIT="m"
# ==============================
# 🔧 CONFIGURABLE PARAMETERS
# ==============================
# Override these via environment variables for different experiment types:
# 
# Quick experiments:     PARALLEL_WORKERS=1 NUM_PROCESSES=2 MAX_PROCESS_TIMEOUT=1800
# Standard experiments:  PARALLEL_WORKERS=1 NUM_PROCESSES=2 MAX_PROCESS_TIMEOUT=0     # REDUCED from 3 to 2
# Long experiments:      PARALLEL_WORKERS=1 NUM_PROCESSES=1 MAX_PROCESS_TIMEOUT=0     # REDUCED to 1 for safety
# Stress tests:          PARALLEL_WORKERS=2 NUM_PROCESSES=3 MAX_PROCESS_TIMEOUT=3600  # Only for testing
#
# SMART TIMEOUT: Processes are killed ONLY if no batch progress for 20 minutes
# Slow processes (making <10 scenarios/30s) get extra 5 minutes (25 min total)
# They can run for hours/days as long as they're making progress
# Set MAX_PROCESS_TIMEOUT=0 to disable absolute timeout (recommended for production)
#
# ⚠️  CRITICAL: AWS Lambda Concurrency Limit = 400 total
#     Each process submits ~100+ batches simultaneously
#     3 processes = ~300+ concurrent lambdas = hits AWS limit and causes hangs
#     SOLUTION: Use NUM_PROCESSES=2 max, or PARALLEL_WORKERS=1 (safer)

PARALLEL_WORKERS=${PARALLEL_WORKERS:-1}          # AWS parallel workers per experiment (1=safe, 2=risky)
NUM_PROCESSES=${NUM_PROCESSES:-2}                # Total parallel experiment processes (REDUCED from 3 to 2)
MAX_PROCESS_TIMEOUT=${MAX_PROCESS_TIMEOUT:-0}    # Absolute max time in seconds (0=no absolute timeout, recommended)
PYTHON_TIMEOUT_ARG=${PYTHON_TIMEOUT_ARG:-0}     # Timeout passed to Python script (0=no timeout)
SKIP_TRAINING="--skip_training"

# Enhanced tracking configuration
EXPERIMENT_ID="parallel_exp_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="parallel_experiments_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_DIR/logs"
STATUS_DIR="$OUTPUT_DIR/status"
DAILY_CACHE_DIR="$OUTPUT_DIR/daily_cache"
HEALTH_CHECK_INTERVAL=300  # 5 minutes
PROGRESS_REPORT_INTERVAL=60  # 1 minute

# S3 configuration
S3_BUCKET="magisterka"
S3_PREFIX="parallel_experiments"

# =============================================================================
# ENHANCED LOGGING AND UTILITIES
# =============================================================================

# Enhanced logging with timestamps
log_with_timestamp() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Ensure log directory exists
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR" 2>/dev/null || true
    fi
    
    # Log to console and file (if directory exists)
    if [ -d "$LOG_DIR" ]; then
        echo "[$timestamp] [$level] $message" | tee -a "$LOG_DIR/master.log"
    else
        echo "[$timestamp] [$level] $message"
    fi
}

log_info() {
    log_with_timestamp "INFO" "$1"
}

log_warning() {
    log_with_timestamp "WARN" "$1"
}

log_error() {
    log_with_timestamp "ERROR" "$1"
}

log_success() {
    log_with_timestamp "SUCCESS" "$1"
}

# Status tracking functions
update_experiment_status() {
    local day=$1
    local status=$2
    local message=$3
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "{\"day\": $day, \"status\": \"$status\", \"message\": \"$message\", \"timestamp\": \"$timestamp\"}" > "$STATUS_DIR/day_${day}_status.json"
}

check_daily_save_status() {
    local day=$1
    local cache_file="$DAILY_CACHE_DIR/day_${day}_saved.json"
    
    if [ -f "$cache_file" ]; then
        local saved=$(cat "$cache_file" | jq -r '.saved_to_s3 // false')
        if [ "$saved" = "true" ]; then
            log_info "📅 Daily save for day $day already completed - skipping"
            return 0
        fi
    fi
    return 1
}

mark_daily_save_complete() {
    local day=$1
    local s3_path=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "{\"day\": $day, \"saved_to_s3\": true, \"s3_path\": \"$s3_path\", \"timestamp\": \"$timestamp\"}" > "$DAILY_CACHE_DIR/day_${day}_saved.json"
}

# Progress monitoring
start_progress_monitor() {
    local total_days=$1
    
    (
        while true; do
            sleep $PROGRESS_REPORT_INTERVAL
            
            # Check if shutdown was requested
            if [ "$SHUTDOWN_REQUESTED" = true ]; then
                break
            fi
            
            # Count completed/failed processes
            local completed=0
            local failed=0
            local running=0
            
            for ((day=1; day<=total_days; day++)); do
                if [ -f "$STATUS_DIR/day_${day}_status.json" ]; then
                    local status=$(cat "$STATUS_DIR/day_${day}_status.json" 2>/dev/null | jq -r '.status' 2>/dev/null || echo "unknown")
                    case $status in
                        "completed") completed=$((completed + 1)) ;;
                        "failed") failed=$((failed + 1)) ;;
                        "running") running=$((running + 1)) ;;
                    esac
                fi
            done
            
            local elapsed=$(( $(date +%s) - START_TIME ))
            local rate=0
            if [ $elapsed -gt 0 ]; then
                rate=$(( completed * 60 / elapsed ))
            fi
            
            log_info "📊 Progress: $completed/$total_days completed, $failed failed, $running running | Rate: ${rate}/min"
            
            # Show detailed progress for running processes
            if [ $running -gt 0 ]; then
                for process_num in $(seq 1 $NUM_PROCESSES); do
                    local process_log="$LOG_DIR/process_${process_num}.log"
                    if [ -f "$process_log" ]; then
                        local detailed_progress=$(extract_batch_progress "$process_log")
                        if [ -n "$detailed_progress" ]; then
                            log_info "📈 Process $process_num: $detailed_progress"
                        fi
                    fi
                done
            fi
            
            # Check if all done
            if [ $((completed + failed)) -eq $total_days ]; then
                break
            fi
        done
    ) &
    
    PROGRESS_PID=$!
}

# Health monitoring
start_health_monitor() {
    (
        while true; do
            sleep $HEALTH_CHECK_INTERVAL
            
            # Check if shutdown was requested
            if [ "$SHUTDOWN_REQUESTED" = true ]; then
                break
            fi
            
            # Check disk space
            local disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//' 2>/dev/null || echo "0")
            if [ "$disk_usage" -gt 90 ] 2>/dev/null; then
                log_warning "⚠️ Disk usage high: ${disk_usage}%"
            fi
            
            # Check memory usage (macOS compatible)
            local mem_usage=0
            if command -v free >/dev/null 2>&1; then
                # Linux
                mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}' 2>/dev/null || echo "0")
            elif command -v vm_stat >/dev/null 2>&1; then
                # macOS
                local vm_stats=$(vm_stat)
                local total_pages=$(echo "$vm_stats" | grep "Pages free:" | awk '{print $3}' | sed 's/\.//')
                local free_pages=$(echo "$vm_stats" | grep "Pages active:" | awk '{print $3}' | sed 's/\.//')
                if [ "$total_pages" -gt 0 ] 2>/dev/null; then
                    mem_usage=$(( (total_pages - free_pages) * 100 / total_pages ))
                fi
            fi
            
            if [ "$mem_usage" -gt 90 ] 2>/dev/null; then
                log_warning "⚠️ Memory usage high: ${mem_usage}%"
            fi
            
            # Check lambda concurrency
            local running_lambdas=$(ps aux | grep "run_pricing_experiment.py" | grep -v grep | wc -l 2>/dev/null || echo "0")
            if [ "$running_lambdas" -gt $((PARALLEL_WORKERS * 2)) ] 2>/dev/null; then
                log_warning "⚠️ Too many lambda processes running: $running_lambdas"
            fi
            
            log_info "🏥 Health check: Disk ${disk_usage}%, Memory ${mem_usage}%, Lambda processes: $running_lambdas"
        done
    ) &
    
    HEALTH_PID=$!
}

# Cleanup function
cleanup() {
    # Only cleanup if not already cleaning up
    if [ "$CLEANUP_STARTED" = true ]; then
        return
    fi
    CLEANUP_STARTED=true
    
    log_info "🧹 Cleaning up background processes..."
    
    # Kill background monitors first (they're less critical)
    [ ! -z "$PROGRESS_PID" ] && kill $PROGRESS_PID 2>/dev/null || true
    [ ! -z "$HEALTH_PID" ] && kill $HEALTH_PID 2>/dev/null || true
    
    # Kill any remaining python processes (but only our experiment ones)
    pkill -f "run_pricing_experiment.py.*--year=$YEAR.*--month=$MONTH" 2>/dev/null || true
    
    # Only kill process group if explicitly requested (not on normal exit)
    if [ "$SHUTDOWN_REQUESTED" = true ] && [ ! -z "$EXPERIMENT_PGRP" ]; then
        log_info "🛑 Killing experiment process group $EXPERIMENT_PGRP"
        kill -TERM -$EXPERIMENT_PGRP 2>/dev/null || true
        sleep 2
        kill -KILL -$EXPERIMENT_PGRP 2>/dev/null || true
    fi
    
    # Final status save
    if [ "$SAVE_FINAL_STATUS" = true ]; then
        save_final_status
    fi
    
    log_info "✅ Cleanup complete"
}

# Signal handlers for immediate termination
handle_interrupt() {
    log_warning "🛑 Received SIGINT (Ctrl+C), shutting down immediately..."
    SHUTDOWN_REQUESTED=true
    SAVE_FINAL_STATUS=true
    cleanup
    exit 130  # Standard exit code for SIGINT
}

handle_terminate() {
    log_warning "🛑 Received SIGTERM, shutting down..."
    SHUTDOWN_REQUESTED=true
    SAVE_FINAL_STATUS=true
    cleanup
    exit 143  # Standard exit code for SIGTERM
}

# Normal exit handler (less aggressive)
handle_exit() {
    # Only cleanup background processes on normal exit, not process group
    if [ "$SHUTDOWN_REQUESTED" != true ]; then
        log_info "🔚 Script ending normally, cleaning up background processes..."
        [ ! -z "$PROGRESS_PID" ] && kill $PROGRESS_PID 2>/dev/null || true
        [ ! -z "$HEALTH_PID" ] && kill $HEALTH_PID 2>/dev/null || true
        SAVE_FINAL_STATUS=true
        save_final_status
    fi
}

# Set up signal handlers
trap handle_interrupt INT
trap handle_terminate TERM
trap handle_exit EXIT

# Global flags
SHUTDOWN_REQUESTED=false
CLEANUP_STARTED=false
SAVE_FINAL_STATUS=false

# =============================================================================
# S3 OPERATIONS
# =============================================================================

check_s3_connection() {
    log_info "🔍 Checking S3 connection..."
    
    if aws s3 ls "s3://$S3_BUCKET" > /dev/null 2>&1; then
        log_success "✅ S3 connection verified"
        return 0
    else
        log_error "❌ S3 connection failed"
        return 1
    fi
}

save_daily_results_to_s3() {
    local day=$1
    local log_file=$2
    
    log_info "📅 Saving daily results for day $day to S3..."
    
    # Check if already saved
    if check_daily_save_status $day; then
        return 0
    fi
    
    # Create daily summary
    local day_dir="$OUTPUT_DIR/day_${day}"
    mkdir -p "$day_dir"
    
    # Copy relevant files
    if [ -f "$log_file" ]; then
        cp "$log_file" "$day_dir/"
    fi
    
    if [ -f "$STATUS_DIR/day_${day}_status.json" ]; then
        cp "$STATUS_DIR/day_${day}_status.json" "$day_dir/"
    fi
    
    # Create summary file
    local summary_file="$day_dir/daily_summary.json"
    cat > "$summary_file" << EOF
{
    "experiment_id": "$EXPERIMENT_ID",
    "day": $day,
    "year": $YEAR,
    "month": $MONTH,
    "vehicle_type": "$VEHICLE_TYPE",
    "borough": "$BOROUGH",
    "methods": "$METHODS",
    "acceptance_function": "$ACCEPTANCE_FUNC",
    "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    # Upload to S3
    local s3_path="$S3_PREFIX/$EXPERIMENT_ID/day_${day}"
    
    if aws s3 cp "$day_dir" "s3://$S3_BUCKET/$s3_path/" --recursive; then
        log_success "✅ Daily results uploaded to s3://$S3_BUCKET/$s3_path/"
        mark_daily_save_complete $day "s3://$S3_BUCKET/$s3_path/"
        return 0
    else
        log_error "❌ Failed to upload daily results for day $day"
        return 1
    fi
}

# =============================================================================
# PROGRESS MONITORING FUNCTIONS
# =============================================================================

extract_batch_progress() {
    local log_file=$1
    local for_comparison=${2:-false}  # If true, return simplified format for progress comparison
    
    if [ ! -f "$log_file" ]; then
        return
    fi
    
    # Get the last few lines to find recent progress
    local recent_lines=$(tail -50 "$log_file" 2>/dev/null || echo "")
    
    if [ -z "$recent_lines" ]; then
        return
    fi
    
    # Extract the most recent progress information
    local progress_line=$(echo "$recent_lines" | grep -E "Progress: [0-9]+/[0-9]+ \([0-9.]+%\)" | tail -1)
    local batch_line=$(echo "$recent_lines" | grep -E "✅[0-9]+ ❌[0-9]+ \| Rate: [0-9.]+/s" | tail -1)
    local eval_line=$(echo "$recent_lines" | grep -E "day[0-9]+_(PL|Sigmoid)" | tail -1)
    local eta_line=$(echo "$recent_lines" | grep -E "ETA: [0-9]+s" | tail -1)
    
    # Build progress summary
    local progress_summary=""
    
    # Extract progress percentage and completion
    if [ -n "$progress_line" ]; then
        local progress_match=$(echo "$progress_line" | sed -n 's/.*Progress: \([0-9]*\/[0-9]*\) (\([0-9.]*%\)).*/\1 (\2)/p')
        if [ -n "$progress_match" ]; then
            progress_summary="$progress_match"
        fi
    fi
    
    # Extract success/failure counts and rate
    if [ -n "$batch_line" ]; then
        local success_count=$(echo "$batch_line" | sed -n 's/.*✅\([0-9]*\).*/\1/p')
        local error_count=$(echo "$batch_line" | sed -n 's/.*❌\([0-9]*\).*/\1/p')
        local rate=$(echo "$batch_line" | sed -n 's/.*Rate: \([0-9.]*\/s\).*/\1/p')
        
        if [ -n "$success_count" ] && [ -n "$error_count" ] && [ -n "$rate" ]; then
            if [ -n "$progress_summary" ]; then
                progress_summary="$progress_summary, ✅$success_count ❌$error_count @ $rate"
            else
                progress_summary="✅$success_count ❌$error_count @ $rate"
            fi
        fi
    fi
    
    # Extract current evaluation function
    if [ -n "$eval_line" ]; then
        local current_eval=$(echo "$eval_line" | sed -n 's/.*day[0-9]*_\(PL\|Sigmoid\).*/\1/p' | tail -1)
        if [ -n "$current_eval" ]; then
            if [ -n "$progress_summary" ]; then
                progress_summary="$progress_summary [$current_eval]"
            else
                progress_summary="Working on $current_eval"
            fi
        fi
    fi
    
    # Extract ETA if available
    if [ -n "$eta_line" ]; then
        local eta_seconds=$(echo "$eta_line" | sed -n 's/.*ETA: \([0-9]*\)s.*/\1/p')
        if [ -n "$eta_seconds" ] && [ "$eta_seconds" -gt 0 ]; then
            local eta_minutes=$((eta_seconds / 60))
            if [ "$eta_minutes" -gt 0 ]; then
                if [ -n "$progress_summary" ]; then
                    progress_summary="$progress_summary, ETA: ${eta_minutes}m"
                else
                    progress_summary="ETA: ${eta_minutes}m"
                fi
            fi
        fi
    fi
    
    # If no detailed progress found, try to extract any meaningful status
    if [ -z "$progress_summary" ]; then
        # Look for completion status
        local completion_line=$(echo "$recent_lines" | grep -E "(SUCCESS RATE|Experiment completed|Failed|✅.*S3)" | tail -1)
        if [ -n "$completion_line" ]; then
            if echo "$completion_line" | grep -q "SUCCESS RATE"; then
                local success_rate=$(echo "$completion_line" | sed -n 's/.*SUCCESS RATE: \([0-9.]*%\).*/\1/p')
                if [ -n "$success_rate" ]; then
                    progress_summary="Completed ($success_rate success)"
                fi
            elif echo "$completion_line" | grep -q "✅.*S3"; then
                progress_summary="Saving to S3..."
            elif echo "$completion_line" | grep -q "Experiment completed"; then
                progress_summary="Completed"
            fi
        fi
    fi
    
    # Return format based on usage
    if [ "$for_comparison" = true ]; then
        # For progress comparison, return the HIGHEST number found (most recent progress)
        local completed_from_progress=0
        local completed_from_batch=0
        
        if [ -n "$progress_line" ]; then
            completed_from_progress=$(echo "$progress_line" | sed -n 's/.*Progress: \([0-9]*\)\/[0-9]* .*/\1/p')
        fi
        
        if [ -n "$batch_line" ]; then
            completed_from_batch=$(echo "$batch_line" | sed -n 's/.*✅\([0-9]*\).*/\1/p')
        fi
        
        # Return the highest number (most recent progress)
        local max_progress=$completed_from_progress
        if [ "$completed_from_batch" -gt "$max_progress" ] 2>/dev/null; then
            max_progress=$completed_from_batch
        fi
        
        echo "$max_progress"
    else
        # Return full detailed progress for display
        echo "$progress_summary"
    fi
}

check_process_completion_status() {
    local log_file=$1
    
    if [ ! -f "$log_file" ]; then
        echo "unknown"
        return
    fi
    
    # Check last few lines for completion indicators
    local recent_lines=$(tail -20 "$log_file" 2>/dev/null || echo "")
    
    if echo "$recent_lines" | grep -q "SUCCESS RATE:"; then
        echo "completed"
    elif echo "$recent_lines" | grep -q "❌.*Failed"; then
        echo "failed"
    elif echo "$recent_lines" | grep -q "🛑.*Shutdown"; then
        echo "interrupted"
    else
        echo "running"
    fi
}

# =============================================================================
# ENHANCED EXPERIMENT EXECUTION
# =============================================================================

run_enhanced_experiment_process() {
    local process_id=$1
    local remainder=$2
    local days_list=$3
    local log_file="$LOG_DIR/process_${process_id}.log"
    
    log_info "🚀 Starting Enhanced Process $process_id (remainder=$remainder)"
    log_info "📅 Days: $days_list"
    log_info "📝 Log file: $log_file"
    
    # Convert days list to array
    local days_array=($days_list)
    
    # Track process status
    local process_start_time=$(date +%s)
    local successful_days=0
    local failed_days=0
    
    for day in "${days_array[@]}"; do
        # Check shutdown flag before each day
        if [ "$SHUTDOWN_REQUESTED" = true ]; then
            log_warning "🛑 Shutdown requested, stopping process $process_id"
            break
        fi
        
        log_info "🔄 Processing day $day..."
        update_experiment_status $day "running" "Started processing"
        
        # Check if already completed
        if check_daily_save_status $day; then
            log_info "⏭️ Day $day already completed, skipping..."
            update_experiment_status $day "completed" "Previously completed"
            successful_days=$((successful_days + 1))
            continue
        fi
        
        # Run the experiment for this day
        local day_start_time=$(date +%s)
        
        # Check if Python script exists
        if [ ! -f "run_pricing_experiment.py" ]; then
            log_error "❌ run_pricing_experiment.py not found!"
            update_experiment_status $day "failed" "Python script not found"
            failed_days=$((failed_days + 1))
            continue
        fi
        
        # Start the python process in background to allow interruption
        log_info "📤 Starting Python experiment for day $day..."
        
        # Build timeout argument if specified
        local timeout_arg=""
        if [ "$PYTHON_TIMEOUT_ARG" -gt 0 ]; then
            timeout_arg="--timeout=$PYTHON_TIMEOUT_ARG"
        fi
        
        # Execute Python script with configurable timeout (no external timeout command needed)
        python run_pricing_experiment.py \
            --year=$YEAR \
            --month=$MONTH \
            --days=$day \
            --borough=$BOROUGH \
            --vehicle_type=$VEHICLE_TYPE \
            --eval=$ACCEPTANCE_FUNC \
            --methods=$METHODS \
            --parallel=$PARALLEL_WORKERS \
            $SKIP_TRAINING \
            --num_eval=$NUM_EVAL \
            --hour_start=$HOUR_START \
            --hour_end=$HOUR_END \
            --time_interval=$TIME_INTERVAL \
            --time_unit=$TIME_UNIT \
            $timeout_arg \
            >> "$log_file" 2>&1 &
        
        local python_pid=$!
        log_info "🔄 Python process started with PID: $python_pid for day $day"
        
        # Wait for the process while checking for shutdown and progress-based timeout
        local wait_count=0
        local last_progress_check=0  # Initialize to 0, not empty string
        local last_progress_time=$(date +%s)
        local no_progress_timeout=1200  # 20 minutes without progress = timeout (increased from 15)
        local grace_period=300  # 5 minute grace period after detecting slow progress
        local max_absolute_timeout=${MAX_PROCESS_TIMEOUT:-0}  # 0 = no absolute timeout
        local process_appears_slow=false
        
        # Get initial progress to set baseline
        sleep 5  # Give process time to start and create initial logs
        local initial_progress=$(extract_batch_progress "$log_file" true)
        if [ -n "$initial_progress" ] && [ "$initial_progress" -gt 0 ] 2>/dev/null; then
            last_progress_check="$initial_progress"
            log_info "📊 Initial progress for day $day: $initial_progress scenarios"
        fi
        
        while kill -0 $python_pid 2>/dev/null; do
            if [ "$SHUTDOWN_REQUESTED" = true ]; then
                log_warning "🛑 Killing python process $python_pid for day $day"
                kill -TERM $python_pid 2>/dev/null || true
                sleep 2
                kill -KILL $python_pid 2>/dev/null || true
                break
            fi
            
            local current_time=$(date +%s)
            
            # Check for progress every 30 seconds
            if [ $((wait_count % 30)) -eq 0 ]; then
                # Force log file sync before checking progress (fix race condition)
                sync 2>/dev/null || true
                sleep 1  # Give log file time to flush
                
                local current_progress_number=$(extract_batch_progress "$log_file" true)  # Get simplified number for comparison
                
                # Check for progress (with improved logging on important events only)
                
                # Check if progress has changed (more completed scenarios)
                if [ -n "$current_progress_number" ] && [ "$current_progress_number" -gt "$last_progress_check" ] 2>/dev/null; then
                    # Progress detected! Reset the no-progress timer
                    local old_progress=$last_progress_check
                    local progress_delta=$((current_progress_number - old_progress))
                    last_progress_time=$current_time
                    last_progress_check="$current_progress_number"
                    
                    # Check if process is slow (less than 10 scenarios in 30 seconds)
                    if [ $progress_delta -lt 10 ] && [ "$old_progress" -gt 0 ]; then
                        process_appears_slow=true
                        log_info "📈 Slow progress detected for day $day: $old_progress → $current_progress_number scenarios (+$progress_delta in 30s) ✅TIMER_RESET"
                    else
                        process_appears_slow=false
                        log_info "📈 Progress detected for day $day: $old_progress → $current_progress_number scenarios (+$progress_delta in 30s) ✅TIMER_RESET"
                    fi
                # If no progress detected, we'll just wait (reduced logging to avoid spam)
                fi
            fi
            
            # Check smart timeout: only if NO progress for configured time
            local time_since_progress=$((current_time - last_progress_time))
            local effective_timeout=$no_progress_timeout
            
            # If process appears slow but is making progress, give it extra time
            if [ "$process_appears_slow" = true ]; then
                effective_timeout=$((no_progress_timeout + grace_period))
            fi
            
            if [ $time_since_progress -gt $effective_timeout ]; then
                if [ "$process_appears_slow" = true ]; then
                    log_warning "⏰ Slow progress timeout for day $day - no batch progress for ${time_since_progress}s (slow process limit: ${effective_timeout}s)"
                else
                    log_warning "⏰ No progress timeout for day $day - no batch progress for ${time_since_progress}s (limit: ${effective_timeout}s)"
                fi
                log_warning "   Last progress: $last_progress_check scenarios completed"
                log_warning "   Log file sync and final check..."
                
                # Final check with log sync before killing
                sync 2>/dev/null || true
                sleep 2
                local final_progress=$(extract_batch_progress "$log_file" true)
                
                if [ -n "$final_progress" ] && [ "$final_progress" -gt "$last_progress_check" ] 2>/dev/null; then
                    log_warning "   📈 Late progress detected: $last_progress_check → $final_progress scenarios"
                    log_warning "   ⏳ Giving process another chance..."
                    last_progress_time=$current_time
                    last_progress_check="$final_progress"
                    continue
                fi
                
                log_warning "   🛑 Confirmed no progress, killing process"
                kill -TERM $python_pid 2>/dev/null || true
                sleep 2
                kill -KILL $python_pid 2>/dev/null || true
                break
            fi
            
            # Optional absolute timeout (only if configured)
            if [ $max_absolute_timeout -gt 0 ] && [ $wait_count -ge $max_absolute_timeout ]; then
                log_warning "⏰ Absolute timeout reached for day $day after ${max_absolute_timeout}s, killing process"
                log_warning "   Process was still making progress, but hit absolute limit"
                kill -TERM $python_pid 2>/dev/null || true
                sleep 2
                kill -KILL $python_pid 2>/dev/null || true
                break
            fi
            
            sleep 1
            wait_count=$((wait_count + 1))
            
            # Log enhanced progress every minute
            if [ $((wait_count % 60)) -eq 0 ]; then
                local progress_info=$(extract_batch_progress "$log_file")
                local progress_age=$((current_time - last_progress_time))
                
                if [ -n "$progress_info" ]; then
                    # Check if we've recently detected progress (within last 2 minutes)
                    if [ $progress_age -lt 120 ]; then
                        log_info "⏳ Day $day (${wait_count}s, PID: $python_pid): $progress_info [ACTIVE - last update ${progress_age}s ago]"
                    else
                        log_info "⏳ Day $day (${wait_count}s, PID: $python_pid): $progress_info [STALE ${progress_age}s] - checking..."
                        
                        # Do an immediate progress check to see if we missed an update
                        sync 2>/dev/null || true
                        local fresh_progress=$(extract_batch_progress "$log_file" true)
                        if [ -n "$fresh_progress" ] && [ "$fresh_progress" -gt "$last_progress_check" ] 2>/dev/null; then
                            log_info "   🔄 Found fresh progress: $last_progress_check → $fresh_progress, updating timer"
                            last_progress_time=$current_time
                            last_progress_check="$fresh_progress"
                        fi
                    fi
                else
                    log_info "⏳ Day $day (${wait_count}s, PID: $python_pid): Waiting for progress... [${progress_age}s since last]"
                fi
            fi
        done
        
        wait $python_pid 2>/dev/null
        local exit_code=$?
        log_info "🏁 Python process for day $day finished with exit code: $exit_code"
        
        if [ $exit_code -eq 0 ]; then
            local day_duration=$(($(date +%s) - day_start_time))
            log_success "✅ Day $day completed in ${day_duration}s"
            
            # Save daily results to S3
            if save_daily_results_to_s3 $day "$log_file"; then
                update_experiment_status $day "completed" "Successfully completed and saved to S3"
                successful_days=$((successful_days + 1))
            else
                update_experiment_status $day "failed" "Experiment succeeded but S3 save failed"
                failed_days=$((failed_days + 1))
            fi
        else
            local day_duration=$(($(date +%s) - day_start_time))
            if [ "$SHUTDOWN_REQUESTED" = true ]; then
                log_warning "⚠️ Day $day interrupted by shutdown after ${day_duration}s"
                update_experiment_status $day "interrupted" "Interrupted by shutdown"
            else
                log_error "❌ Day $day failed after ${day_duration}s"
                update_experiment_status $day "failed" "Experiment execution failed"
                failed_days=$((failed_days + 1))
            fi
        fi
        
        # Brief pause between days to prevent overwhelming (unless shutting down)
        if [ "$SHUTDOWN_REQUESTED" != true ]; then
            sleep 5
        fi
    done
    
    local process_duration=$(($(date +%s) - process_start_time))
    
    log_info "📊 Process $process_id completed in ${process_duration}s"
    log_info "✅ Successful days: $successful_days"
    log_info "❌ Failed days: $failed_days"
    
    # Return exit code based on success rate
    if [ $successful_days -gt 0 ] && [ $failed_days -eq 0 ]; then
        return 0
    elif [ $successful_days -gt $failed_days ]; then
        return 1
    else
        return 2
    fi
}

save_final_status() {
    log_info "💾 Saving final experiment status..."
    
    local final_status_file="$OUTPUT_DIR/final_status.json"
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    # Count final results
    local completed=0
    local failed=0
    local total=0
    
    for status_file in "$STATUS_DIR"/*.json; do
        if [ -f "$status_file" ]; then
            total=$((total + 1))
            # Use jq if available, otherwise use basic grep
            if command -v jq >/dev/null 2>&1; then
                local status=$(cat "$status_file" | jq -r '.status' 2>/dev/null || echo "unknown")
            else
                local status=$(grep '"status"' "$status_file" 2>/dev/null | cut -d'"' -f4 || echo "unknown")
            fi
            case $status in
                "completed") completed=$((completed + 1)) ;;
                "failed") failed=$((failed + 1)) ;;
            esac
        fi
    done
    
    # Calculate success rate (avoid bc dependency)
    local success_rate=0
    if [ $total -gt 0 ]; then
        success_rate=$(( completed * 100 / total ))
    fi
    
    cat > "$final_status_file" << EOF
{
    "experiment_id": "$EXPERIMENT_ID",
    "start_time": "$START_TIME",
    "end_time": "$end_time",
    "duration_seconds": $total_duration,
    "configuration": {
        "year": $YEAR,
        "month": $MONTH,
        "total_days": $TOTAL_DAYS,
        "vehicle_type": "$VEHICLE_TYPE",
        "borough": "$BOROUGH",
        "methods": "$METHODS",
        "acceptance_function": "$ACCEPTANCE_FUNC",
        "parallel_workers": $PARALLEL_WORKERS
    },
    "results": {
        "total_days": $total,
        "completed_days": $completed,
        "failed_days": $failed,
        "success_rate": $success_rate
    },
    "s3_bucket": "$S3_BUCKET",
    "s3_prefix": "$S3_PREFIX/$EXPERIMENT_ID"
}
EOF
    
    # Upload final status to S3
    aws s3 cp "$final_status_file" "s3://$S3_BUCKET/$S3_PREFIX/$EXPERIMENT_ID/" || true
    
    log_info "📄 Final status saved to $final_status_file"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Record start time
    START_TIME=$(date +%s)
    
    # Create directories FIRST (before any logging)
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$STATUS_DIR"
    mkdir -p "$DAILY_CACHE_DIR"
    
    # Set up process group for signal handling (but less aggressively on macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        set -m  # Enable job control on Linux
        EXPERIMENT_PGRP=$$
        log_info "🔧 Process Group: $EXPERIMENT_PGRP (Linux)"
    else
        # On macOS, be more conservative with process group handling
        EXPERIMENT_PGRP=$$
        log_info "🔧 Process ID: $EXPERIMENT_PGRP (macOS - limited process group)"
    fi
    
    log_info "🎯 Enhanced Parallel Ride-Hailing Pricing Experiments"
    log_info "🧪 Experiment ID: $EXPERIMENT_ID"
    log_info "📅 Period: $YEAR-$(printf "%02d" $MONTH) (Days 1-$TOTAL_DAYS)"
    log_info "🏙️ Location: $BOROUGH, $VEHICLE_TYPE taxis"
    log_info "⏰ Time Window: ${HOUR_START}:00-${HOUR_END}:00 (${TIME_INTERVAL}${TIME_UNIT} intervals)"
    log_info "🔧 Methods: $METHODS"
    log_info "📊 Evaluations: $NUM_EVAL per scenario"
    log_info "⚙️ Config: $PARALLEL_WORKERS workers"
    log_info "📁 Output: $OUTPUT_DIR"
    
    # Check S3 connection
    if ! check_s3_connection; then
        log_error "❌ S3 connection failed, aborting"
        exit 1
    fi
    
    # Start monitoring (but more robust)
    log_info "🔄 Starting monitoring processes..."
    start_progress_monitor $TOTAL_DAYS
    start_health_monitor
    log_info "📊 Monitoring started (Progress PID: $PROGRESS_PID, Health PID: $HEALTH_PID)"
    
    # Small delay to let monitoring settle
    sleep 2
    
    log_info "🔄 Starting parallel execution with $NUM_PROCESSES processes..."
    
    # Start background processes
    pids=()
    
    for remainder in $(seq 1 $NUM_PROCESSES); do
        # Check if shutdown was requested before starting new process
        if [ "$SHUTDOWN_REQUESTED" = true ]; then
            log_warning "🛑 Shutdown requested, not starting process $remainder"
            break
        fi
        
        # Adjust remainder for modulo 5 (0,1,2,3,4)
        actual_remainder=$((remainder % 5))
        days_list=$(get_days_for_modulo 5 $actual_remainder $TOTAL_DAYS)
        
        if [ -n "$days_list" ]; then
            log_info "📊 Process $remainder: Days $days_list"
            run_enhanced_experiment_process $remainder $actual_remainder "$days_list" &
            pids+=($!)
            
            # 🚀 COORDINATION: Stagger process starts to avoid simultaneous AWS rate limiting
            if [ $remainder -lt $NUM_PROCESSES ]; then
                log_info "⏳ Staggering start (process $remainder), waiting 30s..."
                sleep 30  # Increased from 10s to 30s to spread out Lambda usage
            fi
        else
            log_warning "⚠️ No days found for process $remainder (remainder=$actual_remainder)"
        fi
    done
    
    log_info "⏳ Waiting for all processes to complete..."
    
    # Wait for all processes with periodic shutdown checks
    failed_processes=()
    
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        process_id=$((i + 1))
        
        # Wait for process while checking for shutdown
        while kill -0 $pid 2>/dev/null; do
            if [ "$SHUTDOWN_REQUESTED" = true ]; then
                log_warning "🛑 Killing process $process_id (PID: $pid) due to shutdown"
                kill -TERM $pid 2>/dev/null || true
                sleep 2
                kill -KILL $pid 2>/dev/null || true
                break
            fi
            sleep 1
        done
        
        wait $pid 2>/dev/null
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            log_success "✅ Process $process_id (PID: $pid) completed successfully"
        elif [ "$SHUTDOWN_REQUESTED" = true ]; then
            log_warning "⚠️ Process $process_id (PID: $pid) interrupted by shutdown"
        else
            log_error "❌ Process $process_id (PID: $pid) failed with exit code $exit_code"
            failed_processes+=($process_id)
        fi
    done
    
    # Final reporting
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        log_warning "🛑 Experiment interrupted by user"
        log_info "📊 Partial results may be available in S3"
    else
        log_info "🎉 All processes completed!"
        
        if [ ${#failed_processes[@]} -eq 0 ]; then
            log_success "✅ All $NUM_PROCESSES processes succeeded"
        else
            log_warning "❌ Failed processes: ${failed_processes[*]}"
        fi
    fi
    
    # Generate final summary
    log_info "📋 Experiment Summary:"
    log_info "   🧪 Experiment ID: $EXPERIMENT_ID"
    log_info "   📁 Output Directory: $OUTPUT_DIR"
    log_info "   📝 Logs: $LOG_DIR"
    log_info "   📊 Status: $STATUS_DIR"
    log_info "   🪣 S3 Bucket: s3://$S3_BUCKET/$S3_PREFIX/$EXPERIMENT_ID/"
    
    # Show monitoring commands
    log_info "🔍 Monitoring commands:"
    log_info "   tail -f $LOG_DIR/process_*.log"
    log_info "   cat $STATUS_DIR/day_*_status.json"
    log_info "   aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/$EXPERIMENT_ID/"
    log_info "   # Quick progress check:"
    log_info "   grep -E '(Progress:|✅.*❌|ETA:)' $LOG_DIR/process_*.log | tail -10"
    
    # Exit with appropriate code
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        exit 130  # Standard exit code for SIGINT
    elif [ ${#failed_processes[@]} -gt 0 ]; then
        exit 1    # Exit with error if processes failed
    else
        exit 0    # Success
    fi
}

# Utility functions (unchanged from original)
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

# Run main function
main "$@" 