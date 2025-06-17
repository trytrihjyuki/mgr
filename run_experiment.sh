#!/bin/bash

# Enhanced Rideshare Experiment Runner with Detailed Logging
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
    echo "  list-data"
    echo ""
    echo -e "${YELLOW}Experiment Commands:${NC}"
    echo "  run-single <vehicle_type> <year> <month> <method> [acceptance_func] [scenarios]"
    echo "  run-comparative <vehicle_type> <year> <month> [acceptance_func] [scenarios]"
    echo "  run-benchmark <vehicle_type> <year> <month> [scenarios]"
    echo "  test-window-time <vehicle_type> <year> <month> <method> <window_seconds>"
    echo "  test-acceptance-functions <vehicle_type> <year> <month> <method>"
    echo ""
    echo -e "${YELLOW}Analysis Commands:${NC}"
    echo "  list-experiments [days]"
    echo "  show-experiment <experiment_id>"
    echo "  analyze <experiment_id>"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 download-single green 2019 3"
    echo "  $0 download-bulk 2019 1 3 green,yellow"
    echo "  $0 run-comparative green 2019 3 PL 5"
    echo "  $0 test-window-time green 2019 3 linear_program 600"
    echo "  $0 analyze run_20241217_123456"
    echo ""
    echo -e "${YELLOW}Available Methods:${NC} proposed, maps, linucb, linear_program"
    echo -e "${YELLOW}Available Vehicle Types:${NC} green, yellow, fhv"
    echo -e "${YELLOW}Available Acceptance Functions:${NC} PL, Sigmoid"
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
        log_info "Record limit: $(printf "%'d" $limit)"
    else
        log_info "Record limit: No limit (full dataset)"
    fi
    
    local payload="{\\\"action\\\":\\\"download_single\\\",\\\"vehicle_type\\\":\\\"$vehicle_type\\\",\\\"year\\\":$year,\\\"month\\\":$month"
    if [[ -n "$limit" ]]; then
        payload="$payload,\\\"limit\\\":$limit"
    fi
    payload="$payload}"
    
    log_progress "Invoking data ingestion Lambda..."
    log_info "Payload: $payload"
    
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json > /dev/null 2>&1

    local status_code=$?
    
    if [[ $status_code -eq 0 ]]; then
        log_success "Lambda invocation completed"
        
        # Parse and display response
        if [[ -f response.json ]]; then
            log_progress "Parsing Lambda response..."
            
            # Extract status and body
            local response_body
            response_body=$(cat response.json | jq -r '.body' 2>/dev/null)
            
            if [[ "$response_body" != "null" && -n "$response_body" ]]; then
                # Parse the nested JSON body
                local status
                local size_mb
                local s3_key
                local error_msg
                
                status=$(echo "$response_body" | jq -r '.status // "unknown"' 2>/dev/null)
                size_mb=$(echo "$response_body" | jq -r '.size_mb // "unknown"' 2>/dev/null)
                s3_key=$(echo "$response_body" | jq -r '.s3_key // "unknown"' 2>/dev/null)
                error_msg=$(echo "$response_body" | jq -r '.error // ""' 2>/dev/null)
                
                if [[ "$status" == "success" ]]; then
                    log_success "Data download completed successfully!"
                    log_result "File size: ${size_mb}MB"
                    log_result "S3 location: s3://magisterka/$s3_key"
                else
                    log_error "Data download failed"
                    if [[ -n "$error_msg" && "$error_msg" != "null" ]]; then
                        log_error "Error: $error_msg"
                    fi
                fi
            else
                log_warning "Could not parse Lambda response body"
            fi
            
            # Show raw response for debugging
            echo ""
            log_info "Raw Lambda Response:"
            cat response.json | jq . 2>/dev/null || cat response.json
        else
            log_error "Response file not found"
        fi
    else
        log_error "Lambda invocation failed with status code: $status_code"
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
    
    local payload="{\\\"action\\\":\\\"download_bulk\\\",\\\"vehicle_types\\\":$vehicle_types_json,\\\"year\\\":$year,\\\"start_month\\\":$start_month,\\\"end_month\\\":$end_month}"
    
    log_progress "Invoking bulk data ingestion Lambda..."
    log_info "This may take several minutes for large date ranges..."
    
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json > /dev/null 2>&1

    local status_code=$?
    
    if [[ $status_code -eq 0 ]]; then
        log_success "Bulk Lambda invocation completed"
        
        if [[ -f response.json ]]; then
            log_progress "Parsing bulk download results..."
            
            local response_body
            response_body=$(cat response.json | jq -r '.body' 2>/dev/null)
            
            if [[ "$response_body" != "null" && -n "$response_body" ]]; then
                local successful
                local failed
                local total
                
                successful=$(echo "$response_body" | jq -r '.successful_downloads // 0' 2>/dev/null)
                failed=$(echo "$response_body" | jq -r '.failed_downloads // 0' 2>/dev/null)
                total=$(echo "$response_body" | jq -r '.total_downloads // 0' 2>/dev/null)
                
                log_result "Bulk Download Summary:"
                log_result "  Total datasets: $total"
                log_result "  Successful: $successful"
                log_result "  Failed: $failed"
                
                if [[ $failed -gt 0 ]]; then
                    log_warning "Some downloads failed. Check the detailed response for error information."
                fi
            fi
            
            echo ""
            log_info "Detailed Response:"
            cat response.json | jq . 2>/dev/null || cat response.json
        fi
    else
        log_error "Bulk Lambda invocation failed with status code: $status_code"
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
    
    local payload="{\\\"vehicle_type\\\":\\\"$vehicle_type\\\",\\\"year\\\":$year,\\\"month\\\":$month,\\\"methods\\\":[\\\"proposed\\\",\\\"maps\\\",\\\"linucb\\\",\\\"linear_program\\\"],\\\"acceptance_function\\\":\\\"$acceptance_function\\\",\\\"simulation_range\\\":$simulation_range}"
    
    log_progress "Invoking experiment runner Lambda..."
    log_info "This may take 2-5 minutes to complete all methods..."
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json > /dev/null 2>&1

    local status_code=$?
    
    if [[ $status_code -eq 0 ]]; then
        log_success "Experiment Lambda invocation completed"
        
        if [[ -f response.json ]]; then
            log_progress "Parsing experiment results..."
            
            local response_body
            response_body=$(cat response.json | jq -r '.body' 2>/dev/null)
            
            if [[ "$response_body" != "null" && -n "$response_body" ]]; then
                # Try to extract key metrics
                local experiment_id
                local s3_key
                local best_method
                local match_rate
                
                experiment_id=$(echo "$response_body" | jq -r '.experiment_id // "unknown"' 2>/dev/null)
                s3_key=$(echo "$response_body" | jq -r '.s3_key // ""' 2>/dev/null)
                
                log_success "Comparative experiment completed!"
                log_result "Experiment ID: $experiment_id"
                
                if [[ -n "$s3_key" && "$s3_key" != "null" ]]; then
                    log_result "Results stored at: s3://magisterka/$s3_key"
                    log_info "Use './run_experiment.sh analyze $experiment_id' to view detailed analysis"
                fi
            fi
            
            echo ""
            log_info "Detailed Response:"
            cat response.json | jq . 2>/dev/null || cat response.json
        fi
    else
        log_error "Experiment Lambda invocation failed with status code: $status_code"
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
    log_progress "Scanning s3://magisterka/experiments/results/ ..."
    
    python local-manager/results_manager.py list $days
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
    shift

    echo ""
    log_info "Enhanced Rideshare Experiment Runner"
    log_info "Command: $command"
    echo ""

    case $command in
        "download-single")
            if [[ $# -lt 3 ]]; then
                log_error "Usage: $0 download-single <vehicle_type> <year> <month> [limit]"
                return 1
            fi
            download_single "$@"
            ;;
        "download-bulk")
            if [[ $# -lt 3 ]]; then
                log_error "Usage: $0 download-bulk <year> <start_month> <end_month> [vehicle_types]"
                return 1
            fi
            download_bulk "$@"
            ;;
        "list-data")
            list_data
            ;;
        "run-single")
            if [[ $# -lt 4 ]]; then
                log_error "Usage: $0 run-single <vehicle_type> <year> <month> <method> [acceptance_func] [scenarios]"
                return 1
            fi
            run_single "$@"
            ;;
        "run-comparative")
            if [[ $# -lt 3 ]]; then
                log_error "Usage: $0 run-comparative <vehicle_type> <year> <month> [acceptance_func] [scenarios]"
                return 1
            fi
            run_comparative "$@"
            ;;
        "list-experiments")
            list_experiments "$@"
            ;;
        "analyze")
            analyze "$@"
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            return 1
            ;;
    esac
}

# Run main function
main "$@" 