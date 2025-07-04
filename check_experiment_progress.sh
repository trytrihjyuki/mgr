#!/bin/bash

# Quick experiment progress checker
# Usage: ./check_experiment_progress.sh [experiment_dir]

set -euo pipefail

# Default to most recent experiment directory
EXPERIMENT_DIR="${1:-$(ls -dt parallel_experiments_* 2>/dev/null | head -1)}"

if [ -z "$EXPERIMENT_DIR" ] || [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "❌ No experiment directory found. Usage: $0 [experiment_dir]"
    echo "Available experiments:"
    ls -dt parallel_experiments_* 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "🔍 Checking progress for: $EXPERIMENT_DIR"
echo "=" * 60

LOG_DIR="$EXPERIMENT_DIR/logs"
STATUS_DIR="$EXPERIMENT_DIR/status"

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: $LOG_DIR"
    exit 1
fi

# Function to extract batch progress (simplified version)
extract_batch_progress() {
    local log_file=$1
    local process_name=$2
    
    if [ ! -f "$log_file" ]; then
        echo "$process_name: No log file"
        return
    fi
    
    # Get recent progress lines
    local recent_lines=$(tail -30 "$log_file" 2>/dev/null)
    
    # Extract the most recent progress
    local progress_line=$(echo "$recent_lines" | grep -E "Progress: [0-9]+/[0-9]+ \([0-9.]+%\)" | tail -1)
    local batch_line=$(echo "$recent_lines" | grep -E "✅[0-9]+ ❌[0-9]+ \| Rate: [0-9.]+/s" | tail -1)
    local eta_line=$(echo "$recent_lines" | grep -E "ETA: [0-9]+s" | tail -1)
    local completion_line=$(echo "$recent_lines" | grep -E "SUCCESS RATE: [0-9.]+%" | tail -1)
    
    local status=""
    
    if [ -n "$completion_line" ]; then
        local success_rate=$(echo "$completion_line" | sed -n 's/.*SUCCESS RATE: \([0-9.]*%\).*/\1/p')
        status="✅ COMPLETED ($success_rate)"
    elif [ -n "$progress_line" ]; then
        local progress_match=$(echo "$progress_line" | sed -n 's/.*Progress: \([0-9]*\/[0-9]*\) (\([0-9.]*%\)).*/\1 (\2)/p')
        status="🔄 $progress_match"
        
        if [ -n "$batch_line" ]; then
            local success_count=$(echo "$batch_line" | sed -n 's/.*✅\([0-9]*\).*/\1/p')
            local error_count=$(echo "$batch_line" | sed -n 's/.*❌\([0-9]*\).*/\1/p')
            local rate=$(echo "$batch_line" | sed -n 's/.*Rate: \([0-9.]*\/s\).*/\1/p')
            
            if [ -n "$success_count" ] && [ -n "$error_count" ]; then
                status="$status - ✅$success_count ❌$error_count @ $rate"
            fi
        fi
        
        if [ -n "$eta_line" ]; then
            local eta_seconds=$(echo "$eta_line" | sed -n 's/.*ETA: \([0-9]*\)s.*/\1/p')
            if [ -n "$eta_seconds" ] && [ "$eta_seconds" -gt 0 ]; then
                local eta_minutes=$((eta_seconds / 60))
                status="$status, ETA: ${eta_minutes}m"
            fi
        fi
    else
        # Check if process is starting or has issues
        if echo "$recent_lines" | grep -q "Starting.*experiment"; then
            status="🚀 Starting..."
        elif echo "$recent_lines" | grep -q "rate limit\|Rate limit"; then
            status="⚠️ Rate limited"
        elif echo "$recent_lines" | tail -10 | grep -q "✅\|❌"; then
            status="🔄 In progress..."
        else
            status="❓ Unknown status"
        fi
    fi
    
    echo "$process_name: $status"
}

# Check each process
echo "📈 PROCESS STATUS:"
process_count=0
for log_file in "$LOG_DIR"/process_*.log; do
    if [ -f "$log_file" ]; then
        process_num=$(basename "$log_file" | sed 's/process_\([0-9]*\)\.log/\1/')
        extract_batch_progress "$log_file" "Process $process_num"
        process_count=$((process_count + 1))
    fi
done

if [ $process_count -eq 0 ]; then
    echo "❌ No process logs found in $LOG_DIR"
fi

# Overall summary
echo ""
echo "📊 OVERALL SUMMARY:"

# Count completed/failed days
completed=0
failed=0
total=0

if [ -d "$STATUS_DIR" ]; then
    for status_file in "$STATUS_DIR"/*.json; do
        if [ -f "$status_file" ]; then
            total=$((total + 1))
            if grep -q '"status": "completed"' "$status_file" 2>/dev/null; then
                completed=$((completed + 1))
            elif grep -q '"status": "failed"' "$status_file" 2>/dev/null; then
                failed=$((failed + 1))
            fi
        fi
    done
fi

if [ $total -gt 0 ]; then
    echo "Completed: $completed/$total days"
    echo "Failed: $failed/$total days"
    echo "Running: $((total - completed - failed))/$total days"
else
    echo "No status files found yet"
fi

# Show recent errors/successes
echo ""
echo "🔍 RECENT ACTIVITY (last 5 events):"
grep -h -E "(✅|❌|SUCCESS RATE|Completed)" "$LOG_DIR"/process_*.log 2>/dev/null | tail -5 || echo "No recent activity found"

# Show quick monitoring commands
echo ""
echo "🛠️ MONITORING COMMANDS:"
echo "  # Watch live progress:"
echo "  tail -f $LOG_DIR/process_*.log"
echo ""
echo "  # Check specific process:"
echo "  tail -f $LOG_DIR/process_1.log"
echo ""
echo "  # Find rate limiting issues:"
echo "  grep -i 'rate limit' $LOG_DIR/process_*.log"
echo ""
echo "  # Check completion status:"
echo "  grep 'SUCCESS RATE' $LOG_DIR/process_*.log" 