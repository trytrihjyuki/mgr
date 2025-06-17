#!/bin/bash

# Enhanced Rideshare Experiment Runner
set -e

REGION="eu-north-1"

show_help() {
    echo "🧪 Enhanced Rideshare Experiment Runner"
    echo "======================================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Data Commands:"
    echo "  download-single <vehicle_type> <year> <month> [limit]"
    echo "  download-bulk <year> <start_month> <end_month> [vehicle_types]"
    echo "  list-data"
    echo ""
    echo "Experiment Commands:"
    echo "  run-single <vehicle_type> <year> <month> <method> [acceptance_func] [scenarios]"
    echo "  run-comparative <vehicle_type> <year> <month> [acceptance_func] [scenarios]"
    echo "  run-benchmark <vehicle_type> <year> <month> [scenarios]"
    echo ""
    echo "Meta-Parameter Testing:"
    echo "  test-window-time <vehicle_type> <year> <month> <method> <window_seconds>"
    echo "  test-retry-count <vehicle_type> <year> <month> <method> <retry_count>"
    echo "  test-acceptance-functions <vehicle_type> <year> <month> <method>"
    echo ""
    echo "Analysis Commands:"
    echo "  list-experiments [days]"
    echo "  show-experiment <experiment_id>"
    echo "  analyze-comparative <experiment_id>"
    echo ""
    echo "Methods: proposed, maps, linucb, linear_program"
    echo "Acceptance Functions: PL, Sigmoid"
    echo ""
    echo "Examples:"
    echo "  $0 run-single green 2019 3 proposed PL 5"
    echo "  $0 run-comparative green 2019 3 PL 3"
    echo "  $0 run-benchmark green 2019 3 5"
    echo "  $0 test-window-time green 2019 3 maps 600"
    echo "  $0 test-acceptance-functions green 2019 3 linucb"
    echo ""
}

download_single() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local limit=${4:-""}
    
    echo "🔄 Downloading $vehicle_type taxi data for $year-$(printf %02d $month)..."
    
    local payload="{\"action\":\"download_single\",\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month"
    if [[ -n "$limit" ]]; then
        payload="$payload,\"limit\":$limit"
    fi
    payload="$payload}"
    
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    echo "📄 Response:"
    cat response.json | python3 -m json.tool
    rm response.json
}

download_bulk() {
    local year=$1
    local start_month=$2
    local end_month=$3
    local vehicle_types=${4:-"green,yellow,fhv"}
    
    echo "🔄 Bulk downloading data for $year months $start_month-$end_month..."
    
    # Convert comma-separated vehicle types to JSON array
    local types_array="["
    IFS=',' read -ra TYPES <<< "$vehicle_types"
    for i in "${!TYPES[@]}"; do
        if [ $i -gt 0 ]; then
            types_array="$types_array,"
        fi
        types_array="$types_array\"${TYPES[i]}\""
    done
    types_array="$types_array]"
    
    local payload="{\"action\":\"download_bulk\",\"vehicle_types\":$types_array,\"year\":$year,\"start_month\":$start_month,\"end_month\":$end_month}"
    
    /usr/local/bin/aws lambda invoke \
        --function-name nyc-data-ingestion \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    echo "📄 Response:"
    cat response.json | python3 -m json.tool
    rm response.json
}

run_single() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    local acceptance_function=${5:-"PL"}
    local simulation_range=${6:-5}
    
    echo "🧪 Running single method experiment: $method on $vehicle_type taxi $year-$(printf %02d $month)..."
    
    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"acceptance_function\":\"$acceptance_function\",\"simulation_range\":$simulation_range}"
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    echo "📄 Response:"
    cat response.json | python3 -m json.tool
    rm response.json
}

run_comparative() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local acceptance_function=${4:-"PL"}
    local simulation_range=${5:-5}
    
    echo "🧪 Running comparative experiment (all 4 methods): $vehicle_type taxi $year-$(printf %02d $month)..."
    
    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"proposed\",\"maps\",\"linucb\",\"linear_program\"],\"acceptance_function\":\"$acceptance_function\",\"simulation_range\":$simulation_range}"
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    echo "📄 Response:"
    cat response.json | python3 -m json.tool
    rm response.json
}

run_benchmark() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local simulation_range=${4:-5}
    
    echo "🧪 Running benchmark (all methods, both acceptance functions)..."
    
    for acceptance_func in "PL" "Sigmoid"; do
        echo "  Testing $acceptance_func acceptance function..."
        local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"proposed\",\"maps\",\"linucb\",\"linear_program\"],\"acceptance_function\":\"$acceptance_func\",\"simulation_range\":$simulation_range}"
        
        /usr/local/bin/aws lambda invoke \
            --function-name rideshare-experiment-runner \
            --payload "$payload" \
            --region $REGION \
            --cli-binary-format raw-in-base64-out \
            response_${acceptance_func}.json
        
        echo "  ✅ $acceptance_func completed"
    done
    
    echo "📄 PL Results:"
    cat response_PL.json | python3 -m json.tool
    echo ""
    echo "📄 Sigmoid Results:"
    cat response_Sigmoid.json | python3 -m json.tool
    
    rm response_PL.json response_Sigmoid.json
}

test_window_time() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    local window_time=$5
    
    echo "🧪 Testing window time parameter: $window_time seconds for $method method..."
    
    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"window_time\":$window_time,\"simulation_range\":3}"
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    echo "📄 Response:"
    cat response.json | python3 -m json.tool
    rm response.json
}

test_retry_count() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    local retry_count=$5
    
    echo "🧪 Testing retry count parameter: $retry_count retries for $method method..."
    
    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"retry_count\":$retry_count,\"simulation_range\":3}"
    
    /usr/local/bin/aws lambda invoke \
        --function-name rideshare-experiment-runner \
        --payload "$payload" \
        --region $REGION \
        --cli-binary-format raw-in-base64-out \
        response.json
    
    echo "📄 Response:"
    cat response.json | python3 -m json.tool
    rm response.json
}

test_acceptance_functions() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local method=$4
    
    echo "🧪 Testing both acceptance functions for $method method..."
    
    for acceptance_func in "PL" "Sigmoid"; do
        echo "  Testing $acceptance_func..."
        local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"methods\":[\"$method\"],\"acceptance_function\":\"$acceptance_func\",\"simulation_range\":3}"
        
        /usr/local/bin/aws lambda invoke \
            --function-name rideshare-experiment-runner \
            --payload "$payload" \
            --region $REGION \
            --cli-binary-format raw-in-base64-out \
            response_${acceptance_func}.json
    done
    
    echo "📄 PL Results:"
    cat response_PL.json | python3 -m json.tool
    echo ""
    echo "📄 Sigmoid Results:"
    cat response_Sigmoid.json | python3 -m json.tool
    
    rm response_PL.json response_Sigmoid.json
}

list_data() {
    echo "📊 Listing datasets in S3..."
    /usr/local/bin/aws s3 ls s3://magisterka/datasets/ --recursive --region $REGION
}

list_experiments() {
    local days=${1:-7}
    echo "📊 Listing experiments from last $days days..."
    python local-manager/results_manager.py list --days $days
}

show_experiment() {
    local experiment_id=$1
    echo "📊 Showing experiment details..."
    python local-manager/results_manager.py show $experiment_id
}

analyze_comparative() {
    local experiment_id=$1
    echo "📊 Analyzing comparative experiment..."
    python local-manager/results_manager.py analyze $experiment_id
}

# Main execution
case "${1:-help}" in
    "download-single")
        if [[ $# -lt 4 ]]; then
            echo "❌ Usage: $0 download-single <vehicle_type> <year> <month> [limit]"
            exit 1
        fi
        download_single $2 $3 $4 $5
        ;;
    "download-bulk")
        if [[ $# -lt 4 ]]; then
            echo "❌ Usage: $0 download-bulk <year> <start_month> <end_month> [vehicle_types]"
            exit 1
        fi
        download_bulk $2 $3 $4 $5
        ;;
    "run-single")
        if [[ $# -lt 5 ]]; then
            echo "❌ Usage: $0 run-single <vehicle_type> <year> <month> <method> [acceptance_func] [scenarios]"
            exit 1
        fi
        run_single $2 $3 $4 $5 $6 $7
        ;;
    "run-comparative")
        if [[ $# -lt 4 ]]; then
            echo "❌ Usage: $0 run-comparative <vehicle_type> <year> <month> [acceptance_func] [scenarios]"
            exit 1
        fi
        run_comparative $2 $3 $4 $5 $6
        ;;
    "run-benchmark")
        if [[ $# -lt 4 ]]; then
            echo "❌ Usage: $0 run-benchmark <vehicle_type> <year> <month> [scenarios]"
            exit 1
        fi
        run_benchmark $2 $3 $4 $5
        ;;
    "test-window-time")
        if [[ $# -lt 6 ]]; then
            echo "❌ Usage: $0 test-window-time <vehicle_type> <year> <month> <method> <window_seconds>"
            exit 1
        fi
        test_window_time $2 $3 $4 $5 $6
        ;;
    "test-retry-count")
        if [[ $# -lt 6 ]]; then
            echo "❌ Usage: $0 test-retry-count <vehicle_type> <year> <month> <method> <retry_count>"
            exit 1
        fi
        test_retry_count $2 $3 $4 $5 $6
        ;;
    "test-acceptance-functions")
        if [[ $# -lt 5 ]]; then
            echo "❌ Usage: $0 test-acceptance-functions <vehicle_type> <year> <month> <method>"
            exit 1
        fi
        test_acceptance_functions $2 $3 $4 $5
        ;;
    "list-data")
        list_data
        ;;
    "list-experiments")
        list_experiments $2
        ;;
    "show-experiment")
        if [[ $# -lt 2 ]]; then
            echo "❌ Usage: $0 show-experiment <experiment_id>"
            exit 1
        fi
        show_experiment $2
        ;;
    "analyze-comparative")
        if [[ $# -lt 2 ]]; then
            echo "❌ Usage: $0 analyze-comparative <experiment_id>"
            exit 1
        fi
        analyze_comparative $2
        ;;
    "help"|*)
        show_help
        ;;
esac 