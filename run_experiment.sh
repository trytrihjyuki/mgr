#!/bin/bash

# 🧪 MAIN EXPERIMENT RUNNER SCRIPT  
# Enhanced Rideshare Experiment Runner with Detailed Logging
# This is the PRIMARY script for running rideshare pricing experiments
set -e

REGION="eu-north-1"

# Colors for better terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${CYAN}ℹ️  INFO${NC}: $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS${NC}: $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  WARNING${NC}: $1"
}

log_error() {
    echo -e "${RED}❌ ERROR${NC}: $1"
}

log_progress() {
    echo -e "${BLUE}🔄 PROGRESS${NC}: $1"
}

log_result() {
    echo -e "${PURPLE}📊 RESULT${NC}: $1"
}

show_help() {
    echo -e "${CYAN}🧪 Enhanced Rideshare Experiment Runner${NC}"
    echo "======================================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo -e "${YELLOW}Data Commands:${NC}"
    echo "  download-single <vehicle_type> <year> <month> [limit]"
    echo "  download-bulk <year> <start_month> <end_month> [vehicle_types]"
    echo "  check-availability <vehicle_type> <year> <month>"
    echo "  list-data"
    echo ""
    echo -e "${YELLOW}Standard Experiments:${NC}"
    echo "  run-single <vehicle_type> <year> <month> <method> [acceptance_func] [scenarios]"
    echo "  run-comparative <vehicle_type> <year> <month> [acceptance_func] [scenarios]"
    echo "  run-benchmark <vehicle_type> <year> <month> [scenarios]"
    echo "  test-window-time <vehicle_type> <year> <month> <method> <window_seconds>"
    echo "  test-acceptance-functions <vehicle_type> <year> <month> <method>"
    echo "  test-meta-params <vehicle_type> <year> <month> <method>"
    echo ""
    echo -e "${YELLOW}Unified Experiments (experiment_PL.py format):${NC}"
    echo "  run-experiment <start_hour> <end_hour> <time_interval> <place> <time_step> <month> <day> <year> [vehicle_type] [methods] [acceptance_func]"
    echo "    Example: run-experiment 10 20 5m Manhattan 30s 10 6 2019 green \"hikima,maps,linucb,linear_program\" PL"
    echo "    Example: run-experiment 10 20 5m Bronx 300s 10 6 2019"
    echo "  run-experiment-24h <time_interval> <place> <time_step> <month> <day> <year> [vehicle_type] [methods] [acceptance_func]"
    echo "    Example: run-experiment-24h 30m Manhattan 30s 10 6 2019 green \"hikima,maps\" PL"
    echo "  run-multi-month <start_hour> <end_hour> <time_interval> <place> <time_step> <months> <days> <year> [vehicle_type] [methods] [acceptance_func]"
    echo "    Example: run-multi-month 10 20 5m Manhattan 30s \"3,4,5\" \"6,10\" 2019 green \"hikima,maps\" PL"
    echo ""
    echo -e "${YELLOW}Advanced Experiment Commands:${NC}"
    echo "  run-with-params <vehicle_type> <year> <month> <method> [params_json]"
    echo "  parameter-sweep <vehicle_type> <year> <month> <method>"
    echo ""
    echo -e "${YELLOW}Analysis Commands:${NC}"
    echo "  list-experiments [days]"
    echo "  show-experiment <experiment_id>"
    echo "  analyze <experiment_id>"
    echo "  compare-methods <experiment_id_1> <experiment_id_2>"
    echo ""
    echo -e "${YELLOW}Parameter Explanations:${NC}"
    echo "  📊 simulation_range: Number of scenarios per method (default: 3-5)"
    echo "  ⏰ window_time: Matching time window in seconds (default: 300s)"
    echo "  🔄 retry_count: Number of retry attempts (default: 3)"
    echo "  🎯 num_eval: Monte Carlo evaluations per scenario (default: 100)"
    echo "  ⏱️  time_interval: Time granularity in minutes (default: 5)"
    echo "  🚖 alpha: Algorithm parameter for w_ij calculation (default: 18)"
    echo "  🚗 s_taxi: Taxi speed parameter (default: 25)"
    echo "  💰 base_price: Base trip price (default: 5.875)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 check-availability yellow 2016 8    # Check if data is available (now works for 2016!)"
    echo "  $0 download-single green 2019 3"
    echo "  $0 download-bulk 2019 1 3 green,yellow"
    echo "  $0 run-comparative green 2019 3 PL 5"
    echo "  $0 run-experiment 10 20 5m Manhattan 30s 10 6 2019 green \"hikima,maps,linucb,linear_program\" PL"
    echo "  $0 run-experiment 10 20 5m Bronx 300s 10 6 2019    # Minimal format with defaults"
    echo "  $0 run-multi-month 10 20 5m Manhattan 30s \"3,4,5\" \"6,10\" 2019 green \"hikima,maps\" PL"
    echo "  $0 test-window-time green 2019 3 linear_program 600"
    echo "  $0 analyze unified_green_manhattan_2019_10_20250618_123456"
    echo "  $0 parameter-sweep green 2019 3 proposed"
    echo ""
    echo -e "${YELLOW}Available Methods:${NC}"
    echo "  Unified: hikima, maps, linucb, linear_program"
    echo "  Legacy: proposed (=hikima), maps, linucb, linear_program"
    echo -e "${YELLOW}Available Vehicle Types:${NC} green, yellow, fhv"
    echo -e "${YELLOW}Available Years:${NC} 2013-2023 (historical data now available)"
    echo -e "${YELLOW}Available Acceptance Functions:${NC} PL, Sigmoid"
    echo ""
    echo -e "${CYAN}✨ New Features:${NC}"
    echo "  📋 Original Format: Extended experiment_PL.py with all methods (hikima, maps, linucb, linear_program)"
    echo "  📅 Multi-temporal: Single days, multiple days, multiple months support"
    echo "  📊 Unified Results: Monthly summaries + daily summaries for comprehensive analysis"
    echo "  🚫 No Duplication: Clean JSON structure without repeated elements"
    echo "  🗓️  Historical Data: Now supports 2013-2023 data (fixed availability issues)"
    echo "  🎯 Scenarios vs num_eval: Clear distinction between time periods (scenarios) and Monte Carlo evaluations (num_eval)"
}

# Enhanced download single with detailed logging
download_single() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local limit=${4:-""}
    
    log_info "Starting single dataset download"
    log_progress "Target: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi data for $year-$(printf %02d $month)"
    
    if [[ -n "$limit" ]]; then
        log_progress "Record limit: $limit"
    else
        log_progress "Record limit: All records (no limit)"
    fi
    
    local payload="{\"action\":\"download_single\",\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month"
    if [[ -n "$limit" ]]; then
        payload="$payload,\"limit\":$limit"
    fi
    payload="$payload}"
    
    log_progress "Invoking data ingestion Lambda (waiting for completion)..."
    echo "⏱️  Please wait - this may take a few minutes..."
    echo ""
    
    # Remove output redirection to see real-time progress and errors
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    local status_code=$?
    echo ""
    
    # Check AWS CLI invocation status
    if [[ $status_code -ne 0 ]]; then
        log_error "AWS Lambda invocation failed with exit code: $status_code"
        log_error "This could indicate network issues, authentication problems, or Lambda service errors"
        return 1
    fi
    
    # Check if response file was created
    if [[ ! -f response.json ]]; then
        log_error "No response file generated - Lambda invocation may have failed"
        return 1
    fi
    
    log_success "Lambda invocation completed - processing results..."
    
    # Parse and validate response
    local response_body
    local lambda_status_code
    local lambda_error_type
    local lambda_error_message
    
    # Check for Lambda execution errors first
    lambda_error_type=$(cat response.json | jq -r '.errorType // ""' 2>/dev/null)
    lambda_error_message=$(cat response.json | jq -r '.errorMessage // ""' 2>/dev/null)
    
    if [[ -n "$lambda_error_type" && "$lambda_error_type" != "" ]]; then
        log_error "Lambda function execution failed!"
        log_error "Error Type: $lambda_error_type"
        log_error "Error Message: $lambda_error_message"
        echo ""
        log_info "Full error response:"
        cat response.json | jq . 2>/dev/null || cat response.json
        return 1
    fi
    
    # Check HTTP status code
    lambda_status_code=$(cat response.json | jq -r '.statusCode // ""' 2>/dev/null)
    if [[ "$lambda_status_code" != "200" ]]; then
        log_error "Lambda function returned non-200 status code: $lambda_status_code"
        echo ""
        log_info "Full response:"
        cat response.json | jq . 2>/dev/null || cat response.json
        return 1
    fi
    
    # Parse successful response body
    response_body=$(cat response.json | jq -r '.body // ""' 2>/dev/null)
    
    if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
        local status
        local s3_key
        local size_bytes
        local records_processed
        local data_source
        
        status=$(echo "$response_body" | jq -r '.status // ""' 2>/dev/null)
        s3_key=$(echo "$response_body" | jq -r '.s3_key // ""' 2>/dev/null)
        size_bytes=$(echo "$response_body" | jq -r '.size_bytes // 0' 2>/dev/null)
        records_processed=$(echo "$response_body" | jq -r '.records_processed // 0' 2>/dev/null)
        data_source=$(echo "$response_body" | jq -r '.data_source // ""' 2>/dev/null)
        
        if [[ "$status" == "success" ]]; then
            log_success "🎉 Single download completed successfully!"
            log_result "📊 DOWNLOAD SUMMARY:"
            log_result "  📁 S3 Location: s3://$BUCKET_NAME/$s3_key"
            log_result "  📦 File Size: $(( size_bytes / 1024 / 1024 )) MB ($(printf "%'d" $size_bytes) bytes)"
            log_result "  📊 Records: $(printf "%'d" $records_processed)"
            log_result "  🌐 Data Source: $data_source"
            
            # Verify upload to S3
            log_progress "Verifying S3 upload..."
            if /usr/local/bin/aws s3 ls "s3://$BUCKET_NAME/$s3_key" --region $REGION > /dev/null 2>&1; then
                log_success "✅ File verified in S3"
            else
                log_warning "⚠️  Could not verify file in S3 (may be access issue)"
            fi
        else
            log_error "Download failed with status: $status"
            
            # Extract error details if available
            local error_msg
            error_msg=$(echo "$response_body" | jq -r '.error // "Unknown error"' 2>/dev/null)
            log_error "Error details: $error_msg"
            return 1
        fi
        
        echo ""
        log_info "📄 Full Response:"
        echo "$response_body" | jq . 2>/dev/null || echo "$response_body"
        
    else
        log_error "Could not parse Lambda response body"
        log_info "Raw response:"
        cat response.json
        return 1
    fi
}

# Enhanced bulk download with progress tracking
download_bulk() {
    local year=$1
    local start_month=$2
    local end_month=$3
    local vehicle_types_str=${4:-"green,yellow,fhv"}
    
    log_info "Starting bulk dataset download"
    log_progress "Year: $year, Months: $start_month-$end_month"
    log_progress "Vehicle types: $(echo "$vehicle_types_str" | tr '[:lower:]' '[:upper:]')"
    
    # Convert comma-separated string to JSON array
    local vehicle_types_json
    vehicle_types_json=$(echo "$vehicle_types_str" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/')
    
    local total_datasets
    local num_types
    num_types=$(echo "$vehicle_types_str" | tr ',' '\n' | wc -l)
    total_datasets=$((num_types * (end_month - start_month + 1)))
    
    log_info "Total datasets to download: $total_datasets"
    log_warning "Large bulk downloads may take 10-15 minutes to complete"
    
    local payload="{\"action\":\"download_bulk\",\"vehicle_types\":$vehicle_types_json,\"year\":$year,\"start_month\":$start_month,\"end_month\":$end_month}"
    
    log_progress "Invoking bulk data ingestion Lambda (this will wait for completion)..."
    echo "⏱️  Please wait - downloading $total_datasets datasets..."
    echo ""
    
    # Remove output redirection to see real-time progress and errors
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    local status_code=$?
    echo ""
    
    # Check AWS CLI invocation status
    if [[ $status_code -ne 0 ]]; then
        log_error "AWS Lambda invocation failed with exit code: $status_code"
        log_error "This could indicate network issues, authentication problems, or Lambda service errors"
        return 1
    fi
    
    # Check if response file was created
    if [[ ! -f response.json ]]; then
        log_error "No response file generated - Lambda invocation may have failed"
        return 1
    fi
    
    log_success "Lambda invocation completed - processing results..."
    
    # Parse and validate response
    local response_body
    local lambda_status_code
    local lambda_error_type
    local lambda_error_message
    
    # Check for Lambda execution errors first
    lambda_error_type=$(cat response.json | jq -r '.errorType // ""' 2>/dev/null)
    lambda_error_message=$(cat response.json | jq -r '.errorMessage // ""' 2>/dev/null)
    
    if [[ -n "$lambda_error_type" && "$lambda_error_type" != "" ]]; then
        log_error "Lambda function execution failed!"
        log_error "Error Type: $lambda_error_type"
        log_error "Error Message: $lambda_error_message"
        echo ""
        log_info "Full error response:"
        cat response.json | jq . 2>/dev/null || cat response.json
        return 1
    fi
    
    # Check HTTP status code
    lambda_status_code=$(cat response.json | jq -r '.statusCode // ""' 2>/dev/null)
    if [[ "$lambda_status_code" != "200" ]]; then
        log_error "Lambda function returned non-200 status code: $lambda_status_code"
        echo ""
        log_info "Full response:"
        cat response.json | jq . 2>/dev/null || cat response.json
        return 1
    fi
    
    # Parse successful response body
    response_body=$(cat response.json | jq -r '.body // ""' 2>/dev/null)
    
    if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
        local successful
        local failed
        local total
        
        successful=$(echo "$response_body" | jq -r '.successful_downloads // 0' 2>/dev/null)
        failed=$(echo "$response_body" | jq -r '.failed_downloads // 0' 2>/dev/null)
        total=$(echo "$response_body" | jq -r '.total_downloads // 0' 2>/dev/null)
        
        log_success "Bulk download completed!"
        log_result "📊 BULK DOWNLOAD SUMMARY:"
        log_result "  📦 Total datasets: $total"
        log_result "  ✅ Successful: $successful"
        log_result "  ❌ Failed: $failed"
        
        # Check if any downloads failed
        if [[ $failed -gt 0 ]]; then
            log_warning "⚠️  $failed downloads failed - checking details..."
            
            # Extract and display failed download details
            echo "$response_body" | jq -r '.results[] | select(.status == "error") | "❌ \(.vehicle_type) \(.year)-\(.month): \(.error)"' 2>/dev/null
            
            if [[ $successful -eq 0 ]]; then
                log_error "❌ ALL downloads failed! This indicates a serious issue."
                return 1
            else
                log_warning "⚠️  Partial success: $successful/$total downloads completed"
                return 2  # Partial failure exit code
            fi
        else
            log_success "🎉 All $successful downloads completed successfully!"
        fi
        
        echo ""
        log_info "📄 Detailed Response (first 20 lines):"
        echo "$response_body" | jq . 2>/dev/null | head -20
        
        if [[ $(echo "$response_body" | jq . 2>/dev/null | wc -l) -gt 20 ]]; then
            echo "... (truncated - see response.json for full details)"
        fi
        
    else
        log_error "Could not parse Lambda response body"
        log_info "Raw response:"
        cat response.json
        return 1
    fi
}

# Enhanced experiment running with detailed logging
run_comparative() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local acceptance_function=${4:-"PL"}
    local simulation_range=${5:-5}
    
    log_info "Starting comparative experiment (all 4 methods)"
    log_progress "Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    log_progress "Period: $year-$(printf %02d $month)"
    log_progress "Acceptance function: $acceptance_function"
    log_progress "Simulation scenarios: $simulation_range"
    log_info "Methods: PROPOSED, MAPS, LINUCB, LINEAR_PROGRAM"
    
    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"proposed\",\"maps\",\"linucb\",\"linear_program\"],\"acceptance_function\":\"$acceptance_function\",\"simulation_range\":$simulation_range}"
    
    log_progress "Invoking experiment runner Lambda (waiting for completion)..."
    echo "⏱️  Please wait - running all 4 methods may take 5-10 minutes..."
    echo ""
    
    # Remove output redirection to see real-time progress and errors
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    local status_code=$?
    echo ""
    
    # Check AWS CLI invocation status
    if [[ $status_code -ne 0 ]]; then
        log_error "AWS Lambda invocation failed with exit code: $status_code"
        log_error "This could indicate network issues, authentication problems, or Lambda service errors"
        return 1
    fi
    
    # Check if response file was created
    if [[ ! -f response.json ]]; then
        log_error "No response file generated - Lambda invocation may have failed"
        return 1
    fi
    
    log_success "Lambda invocation completed - processing results..."
    
    # Parse and validate response
    local response_body
    local lambda_status_code
    local lambda_error_type
    local lambda_error_message
    
    # Check for Lambda execution errors first
    lambda_error_type=$(cat response.json | jq -r '.errorType // ""' 2>/dev/null)
    lambda_error_message=$(cat response.json | jq -r '.errorMessage // ""' 2>/dev/null)
    
    if [[ -n "$lambda_error_type" && "$lambda_error_type" != "" ]]; then
        log_error "Lambda function execution failed!"
        log_error "Error Type: $lambda_error_type"
        log_error "Error Message: $lambda_error_message"
        echo ""
        log_info "Full error response:"
        cat response.json | jq . 2>/dev/null || cat response.json
        return 1
    fi
    
    # Check HTTP status code
    lambda_status_code=$(cat response.json | jq -r '.statusCode // ""' 2>/dev/null)
    if [[ "$lambda_status_code" != "200" ]]; then
        log_error "Lambda function returned non-200 status code: $lambda_status_code"
        echo ""
        log_info "Full response:"
        cat response.json | jq . 2>/dev/null || cat response.json
        return 1
    fi
    
    # Parse successful response body
    response_body=$(cat response.json | jq -r '.body // ""' 2>/dev/null)
    
    if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
        local experiment_id
        local methods_tested
        local s3_key
        local best_method
        local best_objective
        
        experiment_id=$(echo "$response_body" | jq -r '.experiment_id // ""' 2>/dev/null)
        methods_tested=$(echo "$response_body" | jq -r '.methods_tested // []' 2>/dev/null)
        s3_key=$(echo "$response_body" | jq -r '.s3_key // ""' 2>/dev/null)
        best_method=$(echo "$response_body" | jq -r '.best_method // ""' 2>/dev/null)
        best_objective=$(echo "$response_body" | jq -r '.best_objective_value // 0' 2>/dev/null)
        
        log_success "🎉 Comparative experiment completed successfully!"
        log_result "🧪 EXPERIMENT SUMMARY:"
        log_result "  🆔 Experiment ID: $experiment_id"
        log_result "  📋 Methods Tested: $(echo "$methods_tested" | jq -r 'join(", ")' 2>/dev/null)"
        log_result "  🏆 Best Method: $(echo "$best_method" | tr '[:lower:]' '[:upper:]')"
        log_result "  📊 Best Objective: $(printf "%.2f" $best_objective)"
        log_result "  📁 S3 Location: s3://$BUCKET_NAME/$s3_key"
        
        echo ""
        log_info "📈 Method Performance Ranking:"
        echo "$response_body" | jq -r '.results[] | "  \(.rank). \(.method | ascii_upcase): \(.objective_value)"' 2>/dev/null | head -10
        
        echo ""
        log_info "📄 Analysis Command:"
        echo "  python local-manager/results_manager.py analyze $experiment_id"
        
        echo ""
        log_info "📄 Full Response (first 30 lines):"
        echo "$response_body" | jq . 2>/dev/null | head -30
        
        if [[ $(echo "$response_body" | jq . 2>/dev/null | wc -l) -gt 30 ]]; then
            echo "... (truncated - see response.json for full details)"
        fi
        
    else
        log_error "Could not parse Lambda response body"
        log_info "Raw response:"
        cat response.json
        return 1
    fi
}

# Enhanced single method experiment
run_single() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    local acceptance_function=${5:-"PL"}
    local simulation_range=${6:-5}
    
    log_info "Starting single method experiment"
    log_progress "Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    log_progress "Period: $year-$(printf %02d $month)"
    log_progress "Method: $(echo "$method" | tr '[:lower:]' '[:upper:]')"
    log_progress "Acceptance function: $acceptance_function"
    log_progress "Simulation scenarios: $simulation_range"
    
    local payload="{\\\"vehicle_type\\\":\\\"$vehicle_type\\\",\\\"year\\\":$year,\\\"month\\\":$month,\\\"methods\\\":[\\\"$method\\\"],\\\"acceptance_function\\\":\\\"$acceptance_function\\\",\\\"simulation_range\\\":$simulation_range}"
    
    log_progress "Invoking experiment runner Lambda..."
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json > /dev/null 2>&1

    local status_code=$?
    
    if [[ $status_code -eq 0 ]]; then
        log_success "Single method experiment completed"
        
        if [[ -f response.json ]]; then
            local response_body
            response_body=$(cat response.json | jq -r '.body' 2>/dev/null)
            
            if [[ "$response_body" != "null" && -n "$response_body" ]]; then
                local experiment_id
                experiment_id=$(echo "$response_body" | jq -r '.experiment_id // "unknown"' 2>/dev/null)
                
                log_result "Experiment ID: $experiment_id"
                log_info "Use './run_experiment.sh analyze $experiment_id' for detailed analysis"
            fi
            
            echo ""
            log_info "Detailed Response:"
            cat response.json | jq . 2>/dev/null || cat response.json
        fi
    else
        log_error "Single method experiment failed with status code: $status_code"
    fi
}

# Enhanced data listing
list_data() {
    log_info "Listing available datasets in S3..."
    log_progress "Scanning s3://magisterka/datasets/ ..."
    
    /usr/local/bin/aws s3 ls s3://magisterka/datasets/ --recursive --region $REGION | \
    grep -E '\.(parquet|csv)$' | \
    while read -r line; do
        # Parse the S3 listing line
        local date=$(echo "$line" | awk '{print $1}')
        local time=$(echo "$line" | awk '{print $2}')
        local size=$(echo "$line" | awk '{print $3}')
        local key=$(echo "$line" | awk '{print $4}')
        
        # Extract information from key path
        local vehicle_type=$(echo "$key" | sed 's|datasets/\([^/]*\)/.*|\1|')
        local filename=$(basename "$key")
        
        # Convert size to MB
        local size_mb
        if [[ $size -gt 0 ]]; then
            size_mb=$(echo "scale=1; $size/1024/1024" | bc -l 2>/dev/null || echo "0.0")
        else
            size_mb="0.0"
        fi
        
        log_result "$vehicle_type: $filename (${size_mb}MB) - $date $time"
    done
}

# Enhanced experiment listing
list_experiments() {
    local days=${1:-30}
    
    log_info "Listing experiments from last $days days..."
    log_progress "Scanning s3://magisterka/experiments/ ..."
    
    python local-manager/results_manager.py list $days
}

# Run unified experiment (following original experiment_PL.py structure)
run_unified_experiment() {
    local start_hour=$1
    local end_hour=$2
    local time_interval_str=$3  # e.g., "5m" or "30s"
    local place=$4
    local time_step_str=$5      # e.g., "30s" or "300s"
    local month=$6
    local day=$7
    local year=$8
    local vehicle_type=${9:-"green"}
    local methods_str=${10:-"hikima,maps,linucb,linear_program"}
    local acceptance_function=${11:-"PL"}
    
    # Parse time values
    local time_interval=$(echo "$time_interval_str" | sed 's/[ms]//g')
    local time_unit=$(echo "$time_interval_str" | sed 's/[0-9]//g')
    local time_step=$(echo "$time_step_str" | sed 's/s//g')
    
    # Calculate simulation_range (number of time periods)
    local total_minutes=$(( (end_hour - start_hour) * 60 ))
    local interval_minutes=$time_interval
    if [[ "$time_unit" == "s" ]]; then
        interval_minutes=$(( time_interval / 60 ))
    fi
    local simulation_range=$(( total_minutes / interval_minutes ))
    
    # Log experiment size for info
    local num_methods=$(echo "$methods_str" | tr ',' '\n' | wc -l | xargs)
    local total_evaluations=$(( simulation_range * 100 * num_methods ))
    log_info "💪 Running $total_evaluations total evaluations ($simulation_range scenarios × $num_methods methods × 100 evals)"
    
    log_info "🧪 Starting unified experiment (based on experiment_PL.py)"
    log_progress "📍 Place: $place"
    log_progress "📅 Date: $year-$(printf %02d $month)-$(printf %02d $day)"
    log_progress "🕐 Time: ${start_hour}:00-${end_hour}:00 (${time_interval}${time_unit} intervals)"
    log_progress "⏱️  Time step: ${time_step}s"
    log_progress "📊 Simulation range: $simulation_range scenarios"
    log_progress "🚗 Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    log_progress "🔬 Methods: $methods_str"
    log_progress "🎯 Acceptance: $acceptance_function"
    
    echo ""
    log_info "📋 Original experiment_PL.py Parameters:"
    log_progress "✓ place=$place"
    log_progress "✓ day=$day"
    log_progress "✓ time_interval=$time_interval"
    log_progress "✓ time_unit=$time_unit"
    log_progress "✓ simulation_range=$simulation_range"
    
    echo ""
    
    # Convert methods string to JSON array
    local methods_json=$(echo "[\"$(echo "$methods_str" | sed 's/,/", "/g')\"]")
    
    local payload="{\"place\":\"$place\",\"day\":$day,\"time_interval\":$time_interval,\"time_unit\":\"$time_unit\",\"simulation_range\":$simulation_range,\"year\":$year,\"month\":$month,\"vehicle_type\":\"$vehicle_type\",\"methods\":$methods_json,\"acceptance_function\":\"$acceptance_function\",\"num_eval\":100}"
    
    log_progress "Invoking unified experiment Lambda..."
    echo "⏱️  Please wait - running unified experiment..."
    echo ""
    
    # Add timeout handling for large experiments (macOS compatible)
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        --cli-read-timeout 900 \
        --cli-connect-timeout 60 \
        response.json
    
    local status_code=$?
    echo ""
    
    # Check both AWS CLI success AND Lambda function success
    local error_message=$(cat response.json | jq -r '.errorMessage // ""' 2>/dev/null || echo "")
    local error_type=$(cat response.json | jq -r '.errorType // ""' 2>/dev/null || echo "")
    
    if [[ $status_code -eq 0 && -z "$error_message" && -z "$error_type" ]]; then
        log_success "🎉 Unified experiment completed!"
        
        # Parse results
        local response_body=$(cat response.json | jq -r '.body // ""' 2>/dev/null)
        if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
            local experiment_id=$(echo "$response_body" | jq -r '.experiment_id // ""' 2>/dev/null)
            local best_method=$(echo "$response_body" | jq -r '.best_method // ""' 2>/dev/null)
            local s3_key=$(echo "$response_body" | jq -r '.s3_key // ""' 2>/dev/null)
            
            log_result "🆔 Experiment ID: $experiment_id"
            log_result "🏆 Best Method: $best_method"
            
            if [[ "$s3_key" != "" && "$s3_key" != "null" ]]; then
                echo ""
                log_result "📁 S3 Location: s3://$BUCKET_NAME/$s3_key"
                
                # Verify upload to S3
                log_progress "Verifying S3 upload..."
                if /usr/local/bin/aws s3 ls "s3://$BUCKET_NAME/$s3_key" --region $REGION > /dev/null 2>&1; then
                    log_success "✅ File verified in S3"
                else
                    log_warning "⚠️  Could not verify file in S3 (may be access issue)"
                fi
            fi
            
            echo ""
            log_info "🔍 Auto-analyzing experiment results..."
            python local-manager/results_manager.py analyze "$experiment_id" | cat
        fi
    else
        if [[ $status_code -ne 0 ]]; then
            log_error "AWS CLI command failed with status code: $status_code"
        else
            log_error "Lambda function failed: $error_type"
        fi
        
        # Show detailed error information
        if [[ -f response.json ]]; then
            echo ""
            log_info "Error details:"
            
            # Check for specific error types
            if [[ "$error_type" == "Runtime.ImportModuleError" ]]; then
                log_error "❌ Import Error: Lambda cannot import required modules"
                if [[ "$error_message" == *"numpy"* ]]; then
                    log_error "🔧 Numpy packaging issue detected - needs Lambda layer fix"
                fi
            elif [[ "$error_message" == *"timed out"* ]] || [[ "$error_message" == *"timeout"* ]] || [[ "$error_message" == *"Task timed out"* ]]; then
                log_error "⏰ Lambda function timed out - experiment too large for current setup"
            fi
            
            # Pretty print the error
            echo ""
            cat response.json | jq . 2>/dev/null || cat response.json
        else
            log_warning "No response file found - possible network timeout"
            log_info "💡 Try with smaller parameters or check network connection"
        fi
    fi
}

# Run multi-month unified experiments  
run_multi_month_unified() {
    local start_hour=$1
    local end_hour=$2
    local time_interval_str=$3
    local place=$4
    local time_step_str=$5
    local months_str=$6         # e.g., "3,4,5"
    local days_str=$7           # e.g., "6,10" 
    local year=$8
    local vehicle_type=${9:-"green"}
    local methods_str=${10:-"hikima,maps,linucb,linear_program"}
    local acceptance_function=${11:-"PL"}
    
    log_info "📅 Starting multi-month unified experiment"
    log_progress "📍 Place: $place"
    log_progress "📅 Year: $year"
    log_progress "📅 Months: $months_str"
    log_progress "📅 Days: $days_str"
    log_progress "🕐 Time: ${start_hour}:00-${end_hour}:00 (${time_interval_str} intervals)"
    log_progress "🚗 Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    
    echo ""
    log_info "📋 Multi-Month Features:"
    log_progress "✓ Monthly summaries for trend analysis"
    log_progress "✓ Daily summaries for plotting"
    log_progress "✓ Cross-month comparative statistics"
    
    echo ""
    log_warning "⚠️  Multi-month experiments can take 30-60 minutes"
    echo ""
    
    # Parse time values
    local time_interval=$(echo "$time_interval_str" | sed 's/[ms]//g')
    local time_unit=$(echo "$time_interval_str" | sed 's/[0-9]//g')
    local time_step=$(echo "$time_step_str" | sed 's/s//g')
    
    # Calculate simulation_range
    local total_minutes=$(( (end_hour - start_hour) * 60 ))
    local interval_minutes=$time_interval
    if [[ "$time_unit" == "s" ]]; then
        interval_minutes=$(( time_interval / 60 ))
    fi
    local simulation_range=$(( total_minutes / interval_minutes ))
    
    # Convert comma-separated strings to JSON arrays
    local months_json=$(echo "[$months_str]" | sed 's/,/, /g')
    local days_json=$(echo "[$days_str]" | sed 's/,/, /g')
    local methods_json=$(echo "[\"$(echo "$methods_str" | sed 's/,/", "/g')\"]")
    
    local payload="{\"place\":\"$place\",\"day\":$days_json,\"time_interval\":$time_interval,\"time_unit\":\"$time_unit\",\"simulation_range\":$simulation_range,\"year\":$year,\"month\":$months_json,\"vehicle_type\":\"$vehicle_type\",\"methods\":$methods_json,\"acceptance_function\":\"$acceptance_function\",\"num_eval\":100}"
    
    log_progress "Invoking multi-month unified Lambda..."
    echo "⏱️  Please wait - running multi-month experiment..."
    echo ""
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    local status_code=$?
    echo ""
    
    if [[ $status_code -eq 0 ]]; then
        log_success "🎉 Multi-month unified experiment completed!"
        
        # Parse results
        local response_body=$(cat response.json | jq -r '.body // ""' 2>/dev/null)
        if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
            local experiment_id=$(echo "$response_body" | jq -r '.experiment_id // ""' 2>/dev/null)
            local best_method=$(echo "$response_body" | jq -r '.best_method // ""' 2>/dev/null)
            local s3_key=$(echo "$response_body" | jq -r '.s3_key // ""' 2>/dev/null)
            
            log_result "🆔 Experiment ID: $experiment_id"
            log_result "🏆 Best Method: $best_method"
            
            if [[ "$s3_key" != "" && "$s3_key" != "null" ]]; then
                echo ""
                log_result "📁 S3 Location: s3://$BUCKET_NAME/$s3_key"
                
                # Verify upload to S3
                log_progress "Verifying S3 upload..."
                if /usr/local/bin/aws s3 ls "s3://$BUCKET_NAME/$s3_key" --region $REGION > /dev/null 2>&1; then
                    log_success "✅ File verified in S3"
                else
                    log_warning "⚠️  Could not verify file in S3 (may be access issue)"
                fi
            fi
            
            echo ""
            log_info "📄 Analysis Command:"
            echo "  python local-manager/results_manager.py analyze $experiment_id"
        fi
    else
        log_error "Multi-month unified experiment failed"
    fi
}

# 24-hour unified experiment function
run_unified_experiment_24h() {
    local time_interval=$1
    local place=$2
    local time_step=$3
    local month=$4
    local day=$5
    local year=$6
    local vehicle_type=${7:-green}
    local methods_str=${8:-"hikima,maps,linucb,linear_program"}
    local acceptance_func=${9:-PL}
    
    log_info "💪 Running 24-hour unified experiment (0:00-24:00)"
    log_progress "🌍 Location: $place"
    log_progress "📅 Date: $year-$(printf %02d $month)-$(printf %02d $day)"
    log_progress "🚖 Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]')"
    log_progress "🔬 Methods: $methods_str"
    log_progress "⏱️ Time Interval: $time_interval"
    log_progress "🎯 Acceptance Function: $acceptance_func"
    
    # Convert to start_hour=0, end_hour=24
    run_unified_experiment 0 24 "$time_interval" "$place" "$time_step" "$month" "$day" "$year" "$vehicle_type" "$methods_str" "$acceptance_func" 
}

# Enhanced analysis
analyze() {
    local experiment_id=$1
    
    if [[ -z "$experiment_id" ]]; then
        log_error "Experiment ID required"
        echo "Usage: $0 analyze <experiment_id>"
        return 1
    fi
    
    log_info "Analyzing experiment: $experiment_id"
    log_progress "Loading results and generating analysis..."
    
    python local-manager/results_manager.py analyze "$experiment_id"
}

# Enhanced main command dispatcher
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        return 0
    fi

    local command=$1
    
    echo ""
    log_info "Enhanced Rideshare Experiment Runner"
    log_info "Command: $command"
    echo ""
    
    case "$command" in
        "download-single")
            if [[ $# -lt 4 ]]; then
                log_error "Usage: $0 download-single <vehicle_type> <year> <month> [limit]"
                return 1
            fi
            download_single_enhanced "${@:2}"
            ;;
        "download-bulk")
            if [[ $# -lt 4 ]]; then
                log_error "Usage: $0 download-bulk <year> <start_month> <end_month> [vehicle_types]"
                return 1
            fi
            download_bulk_enhanced "${@:2}"
            ;;
        "check-availability")
            if [[ $# -lt 4 ]]; then
                log_error "Usage: $0 check-availability <vehicle_type> <year> <month>"
                return 1
            fi
            check_data_availability "${@:2}"
            ;;
        "list-data")
            list_data "${@:2}"
            ;;
        "run-single")
            if [[ $# -lt 5 ]]; then
                log_error "Usage: $0 run-single <vehicle_type> <year> <month> <method> [acceptance_func] [scenarios]"
                return 1
            fi
            run_single "${@:2}"
            ;;
        "run-comparative")
            if [[ $# -lt 4 ]]; then
                log_error "Usage: $0 run-comparative <vehicle_type> <year> <month> [acceptance_func] [scenarios]"
                return 1
            fi
            run_comparative "${@:2}"
            ;;
        "run-experiment")
            if [[ $# -lt 9 ]]; then
                log_error "Usage: $0 run-experiment <start_hour> <end_hour> <time_interval> <place> <time_step> <month> <day> <year> [vehicle_type] [methods] [acceptance_func]"
                log_error "Example: $0 run-experiment 10 20 5m Manhattan 30s 10 6 2019 green \"hikima,maps,linucb,linear_program\" PL"
                log_error "Example: $0 run-experiment 10 20 5m Bronx 300s 10 6 2019"
                return 1
            fi
            run_unified_experiment "${@:2}"
            ;;
        "run-experiment-24h")
            if [[ $# -lt 7 ]]; then
                log_error "Usage: $0 run-experiment-24h <time_interval> <place> <time_step> <month> <day> <year> [vehicle_type] [methods] [acceptance_func]"
                log_error "Example: $0 run-experiment-24h 30m Manhattan 30s 10 6 2019 green \"hikima,maps\" PL"
                return 1
            fi
            run_unified_experiment_24h "${@:2}"
            ;;
        "run-multi-month")
            if [[ $# -lt 9 ]]; then
                log_error "Usage: $0 run-multi-month <start_hour> <end_hour> <time_interval> <place> <time_step> <months> <days> <year> [vehicle_type] [methods] [acceptance_func]"
                log_error "Example: $0 run-multi-month 10 20 5m Manhattan 30s \"3,4,5\" \"6,10\" 2019 green \"hikima,maps\" PL"
                return 1
            fi
            run_multi_month_unified "${@:2}"
            ;;
        "run-benchmark")
            if [[ $# -lt 4 ]]; then
                log_error "Usage: $0 run-benchmark <vehicle_type> <year> <month> [scenarios]"
                return 1
            fi
            run_benchmark "${@:2}"
            ;;
        "test-window-time")
            if [[ $# -lt 6 ]]; then
                log_error "Usage: $0 test-window-time <vehicle_type> <year> <month> <method> <window_seconds>"
                return 1
            fi
            test_window_time "${@:2}"
            ;;
        "test-acceptance-functions")
            if [[ $# -lt 5 ]]; then
                log_error "Usage: $0 test-acceptance-functions <vehicle_type> <year> <month> <method>"
                return 1
            fi
            test_acceptance_functions "${@:2}"
            ;;
        "test-meta-params")
            if [[ $# -lt 5 ]]; then
                log_error "Usage: $0 test-meta-params <vehicle_type> <year> <month> <method>"
                return 1
            fi
            test_meta_params "${@:2}"
            ;;
        "run-with-params")
            if [[ $# -lt 5 ]]; then
                log_error "Usage: $0 run-with-params <vehicle_type> <year> <month> <method> [params_json]"
                return 1
            fi
            run_with_params "${@:2}"
            ;;
        "parameter-sweep")
            if [[ $# -lt 5 ]]; then
                log_error "Usage: $0 parameter-sweep <vehicle_type> <year> <month> <method>"
                return 1
            fi
            parameter_sweep "${@:2}"
            ;;
        "analyze")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 analyze <experiment_id>"
                return 1
            fi
            analyze "${@:2}"
            ;;
        "list-experiments")
            list_experiments "${@:2}"
            ;;
        "show-experiment")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 show-experiment <experiment_id>"
                return 1
            fi
            show_experiment "${@:2}"
            ;;
        "compare-methods")
            if [[ $# -lt 3 ]]; then
                log_error "Usage: $0 compare-methods <experiment_id_1> <experiment_id_2>"
                return 1
            fi
            compare_methods "${@:2}"
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
}

# Add new advanced command implementations

test_meta_params() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    
    log_info "Testing meta-parameters for $method method"
    log_progress "Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    log_progress "Period: $year-$(printf %02d $month)"
    
    echo ""
    log_info "🧪 Running meta-parameter sensitivity analysis..."
    
    # Test different num_eval values
    echo ""
    log_info "📊 Testing num_eval parameter (Monte Carlo evaluations):"
    for num_eval in 50 100 200; do
        log_progress "Testing num_eval=$num_eval..."
        
        local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"acceptance_function\":\"PL\",\"simulation_range\":2,\"num_eval\":$num_eval}"
        
        /usr/local/bin/aws lambda invoke \
            --function-name rideshare-experiment-runner \
            --payload "$payload" \
            --region $REGION \
            --cli-binary-format raw-in-base64-out \
            response_numeval_${num_eval}.json > /dev/null 2>&1
        
        if [[ $? -eq 0 ]]; then
            local obj_value=$(cat response_numeval_${num_eval}.json | jq -r '.body' | jq -r '.performance.objective_value // "0"' 2>/dev/null | sed 's/,//g')
            log_result "  num_eval=$num_eval → Objective: $(printf "%.2f" $obj_value)"
        else
            log_warning "  num_eval=$num_eval → Failed"
        fi
    done
    
    # Test different window_time values
    echo ""
    log_info "⏰ Testing window_time parameter (matching window):"
    for window_time in 180 300 600; do
        log_progress "Testing window_time=${window_time}s..."
        
        local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"acceptance_function\":\"PL\",\"simulation_range\":2,\"window_time\":$window_time}"
        
        /usr/local/bin/aws lambda invoke \
            --function-name rideshare-experiment-runner \
            --payload "$payload" \
            --region $REGION \
            --cli-binary-format raw-in-base64-out \
            response_window_${window_time}.json > /dev/null 2>&1
        
        if [[ $? -eq 0 ]]; then
            local obj_value=$(cat response_window_${window_time}.json | jq -r '.body' | jq -r '.performance.objective_value // "0"' 2>/dev/null | sed 's/,//g')
            log_result "  window_time=${window_time}s → Objective: $(printf "%.2f" $obj_value)"
        else
            log_warning "  window_time=${window_time}s → Failed"
        fi
    done
    
    # Test different alpha values
    echo ""
    log_info "🚖 Testing alpha parameter (algorithm parameter):"
    for alpha in 10 18 25; do
        log_progress "Testing alpha=$alpha..."
        
        local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"acceptance_function\":\"PL\",\"simulation_range\":2,\"alpha\":$alpha}"
        
        /usr/local/bin/aws lambda invoke \
            --function-name rideshare-experiment-runner \
            --payload "$payload" \
            --region $REGION \
            --cli-binary-format raw-in-base64-out \
            response_alpha_${alpha}.json > /dev/null 2>&1
        
        if [[ $? -eq 0 ]]; then
            local obj_value=$(cat response_alpha_${alpha}.json | jq -r '.body' | jq -r '.performance.objective_value // "0"' 2>/dev/null | sed 's/,//g')
            log_result "  alpha=$alpha → Objective: $(printf "%.2f" $obj_value)"
        else
            log_warning "  alpha=$alpha → Failed"
        fi
    done
    
    log_success "🎉 Meta-parameter testing completed!"
    log_info "📄 Results saved to response_*_*.json files"
}

run_with_params() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    local params_json=${5:-"{}"}
    
    log_info "Running experiment with custom parameters"
    log_progress "Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    log_progress "Period: $year-$(printf %02d $month)"
    log_progress "Method: $(echo "$method" | tr '[:lower:]' '[:upper:]')"
    log_progress "Custom params: $params_json"
    
    # Build base payload
    local base_payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"]"
    
    # Merge with custom parameters (simplified - could be more sophisticated)
    local payload="$base_payload"
    if [[ "$params_json" != "{}" ]]; then
        # Extract and add custom parameters (basic implementation)
        payload="${base_payload%?},$params_json}"
        payload="${payload#\{}"
        payload="{$payload"
    else
        payload="$base_payload}"
    fi
    
    log_progress "Invoking Lambda with custom parameters..."
    echo ""
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    local status_code=$?
    echo ""
    
    if [[ $status_code -eq 0 ]]; then
        log_success "🎉 Custom parameter experiment completed!"
        
        local response_body=$(cat response.json | jq -r '.body // ""' 2>/dev/null)
        if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
            local experiment_id=$(echo "$response_body" | jq -r '.experiment_id // ""' 2>/dev/null)
            local s3_key=$(echo "$response_body" | jq -r '.s3_key // ""' 2>/dev/null)
            
            log_result "  🆔 Experiment ID: $experiment_id"
            
            if [[ "$s3_key" != "" && "$s3_key" != "null" ]]; then
                log_result "  📁 S3 Location: s3://$BUCKET_NAME/$s3_key"
                
                # Verify upload to S3
                log_progress "Verifying S3 upload..."
                if /usr/local/bin/aws s3 ls "s3://$BUCKET_NAME/$s3_key" --region $REGION > /dev/null 2>&1; then
                    log_success "✅ File verified in S3"
                else
                    log_warning "⚠️  Could not verify file in S3 (may be access issue)"
                fi
            fi
            
            echo ""
            log_info "📄 Analysis Command:"
            echo "  python local-manager/results_manager.py analyze $experiment_id"
        fi
    else
        log_error "Custom parameter experiment failed"
    fi
}

parameter_sweep() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    
    log_info "Running comprehensive parameter sweep"
    log_progress "Vehicle: $(echo "$vehicle_type" | tr '[:lower:]' '[:upper:]') taxi"
    log_progress "Period: $year-$(printf %02d $month)"
    log_progress "Method: $(echo "$method" | tr '[:lower:]' '[:upper:]')"
    
    echo ""
    log_warning "⚠️  This will run multiple experiments and may take 15-20 minutes"
    echo ""
    
    # Parameter combinations to test
    local window_times=(180 300 600)
    local num_evals=(50 100 200)
    local alphas=(10 18 25)
    local acceptance_functions=("PL" "Sigmoid")
    
    local total_experiments=$((${#window_times[@]} * ${#num_evals[@]} * ${#alphas[@]} * ${#acceptance_functions[@]}))
    local current_experiment=0
    
    log_info "🔬 Running $total_experiments parameter combinations..."
    
    # Create results summary file
    local sweep_results="parameter_sweep_${vehicle_type}_${year}_${month}_${method}_$(date +%Y%m%d_%H%M%S).csv"
    echo "window_time,num_eval,alpha,acceptance_function,objective_value,match_rate,experiment_id" > $sweep_results
    
    for window_time in "${window_times[@]}"; do
        for num_eval in "${num_evals[@]}"; do
            for alpha in "${alphas[@]}"; do
                for acceptance_func in "${acceptance_functions[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    
                    log_progress "[$current_experiment/$total_experiments] Testing: window=${window_time}s, eval=$num_eval, alpha=$alpha, func=$acceptance_func"
                    
                    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"acceptance_function\":\"$acceptance_func\",\"simulation_range\":2,\"window_time\":$window_time,\"num_eval\":$num_eval,\"alpha\":$alpha}"
                    
                    /usr/local/bin/aws lambda invoke \
                        --function-name rideshare-experiment-runner \
                        --payload "$payload" \
                        --region $REGION \
                        --cli-binary-format raw-in-base64-out \
                        sweep_response_${current_experiment}.json > /dev/null 2>&1
                    
                    if [[ $? -eq 0 ]]; then
                        local response_body=$(cat sweep_response_${current_experiment}.json | jq -r '.body // ""' 2>/dev/null)
                        if [[ "$response_body" != "" && "$response_body" != "null" ]]; then
                            local experiment_id=$(echo "$response_body" | jq -r '.experiment_id // ""' 2>/dev/null)
                            local obj_value=$(echo "$response_body" | jq -r '.method_results[].summary.avg_objective_value // 0' 2>/dev/null || echo "0")
                            local match_rate=$(echo "$response_body" | jq -r '.method_results[].summary.avg_match_rate // 0' 2>/dev/null || echo "0")
                            
                            echo "$window_time,$num_eval,$alpha,$acceptance_func,$obj_value,$match_rate,$experiment_id" >> $sweep_results
                            log_result "    → Obj: $(printf "%.0f" $obj_value), Match: $(printf "%.2f%%" $(echo "$match_rate * 100" | bc -l 2>/dev/null || echo "0"))"
                        fi
                    else
                        log_warning "    → Failed"
                        echo "$window_time,$num_eval,$alpha,$acceptance_func,0,0,FAILED" >> $sweep_results
                    fi
                    
                    # Small delay to avoid overwhelming the Lambda
                    sleep 2
                done
            done
        done
    done
    
    log_success "🎉 Parameter sweep completed!"
    log_result "📊 Results saved to: $sweep_results"
    log_info "📈 Best performing combinations:"
    
    # Show top 3 results
    tail -n +2 $sweep_results | sort -t',' -k5 -nr | head -3 | while IFS=',' read -r wt ne alpha func obj match exp_id; do
        log_result "  Objective $obj: window=${wt}s, eval=$ne, alpha=$alpha, func=$func"
    done
}

# Add missing command implementations
show_experiment() {
    local experiment_id=$1
    
    log_info "Showing experiment details"
    log_progress "Experiment ID: $experiment_id"
    
    python local-manager/results_manager.py show "$experiment_id"
}

compare_methods() {
    local exp_id_1=$1
    local exp_id_2=$2
    
    log_info "Comparing experiment methods"
    log_progress "Experiment 1: $exp_id_1"
    log_progress "Experiment 2: $exp_id_2"
    
    python local-manager/results_manager.py compare "$exp_id_1" "$exp_id_2"
}

# Add new command function for checking data availability
check_data_availability() {
    local vehicle_type="$1"
    local year="$2"
    local month="$3"
    
    log_info "🔍 Checking data availability for $vehicle_type $year-$(printf "%02d" $month)"
    
    # Check current S3 data
    local s3_key="datasets/$vehicle_type/year=$year/month=$(printf "%02d" $month)/"
    
    if /usr/local/bin/aws s3 ls "s3://magisterka/$s3_key" --region eu-north-1 &>/dev/null; then
        log_success "✅ Data already available in S3"
        return 0
    fi
    
    # Check known availability patterns - Allow download attempts for all years
    # NYC Open Data has historical data going back to 2013-2016 depending on vehicle type
    # Let the actual download process determine availability rather than blocking preemptively
    
    log_warning "⚠️ Data not in S3, will attempt download"
    return 0
}

# Add availability check to download functions
download_single_enhanced() {
    local vehicle_type="$1"
    local year="$2" 
    local month="$3"
    
    # Check availability first
    if ! check_data_availability "$vehicle_type" "$year" "$month"; then
        log_error "Data availability check failed"
        return 1
    fi
    
    # Proceed with original download logic
    download_single "$vehicle_type" "$year" "$month"
}

# Enhanced bulk download with pre-checking
download_bulk_enhanced() {
    local start_year="$1"
    local start_month="$2"
    local end_month="$3"
    local vehicle_types="$4"
    
    log_info "🔍 Pre-checking data availability..."
    
    IFS=',' read -ra TYPES <<< "$vehicle_types"
    local total_requests=0
    local likely_failures=0
    
    for vehicle_type in "${TYPES[@]}"; do
        vehicle_type=$(echo "$vehicle_type" | tr '[:upper:]' '[:lower:]')
        
        for month in $(seq "$start_month" "$end_month"); do
            total_requests=$((total_requests + 1))
            
            if ! check_data_availability "$vehicle_type" "$start_year" "$month" 2>/dev/null; then
                likely_failures=$((likely_failures + 1))
            fi
        done
    done
    
    if [ "$likely_failures" -gt 0 ]; then
        log_warning "⚠️ Pre-check indicates $likely_failures/$total_requests requests may fail"
        log_warning "💡 Note: NYC Open Data has historical data available from 2013-2016 depending on vehicle type"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Operation cancelled"
            return 1
        fi
    fi
    
    # Proceed with original bulk download
    download_bulk "$start_year" "$start_month" "$end_month" "$vehicle_types"
}

# Run main function with all arguments
main "$@" 