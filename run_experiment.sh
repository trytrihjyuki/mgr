#!/bin/bash

# Helper script to run experiments without JSON parsing issues
set -e

REGION="eu-north-1"

show_help() {
    echo "🧪 Rideshare Experiment Runner"
    echo "==============================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  download-single <vehicle_type> <year> <month> [limit]"
    echo "  download-bulk <year> <start_month> <end_month> [vehicle_types]"
    echo "  run-experiment <vehicle_type> <year> <month> [simulation_range] [acceptance_function]"
    echo "  list-data"
    echo "  list-experiments [days]"
    echo "  show-experiment <experiment_id>"
    echo ""
    echo "Examples:"
    echo "  $0 download-single green 2019 3"
    echo "  $0 download-bulk 2019 1 3 green,yellow"
    echo "  $0 run-experiment green 2019 3 5 PL"
    echo "  $0 list-experiments 7"
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

run_experiment() {
    local vehicle_type=$1
    local year=$2
    local month=$3
    local simulation_range=${4:-5}
    local acceptance_function=${5:-"PL"}
    
    echo "🧪 Running experiment: $vehicle_type taxi $year-$(printf %02d $month) with $simulation_range scenarios..."
    
    local payload="{\"vehicle_type\":\"$vehicle_type\",\"year\":$year,\"month\":$month,\"simulation_range\":$simulation_range,\"acceptance_function\":\"$acceptance_function\"}"
    
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
    "run-experiment")
        if [[ $# -lt 4 ]]; then
            echo "❌ Usage: $0 run-experiment <vehicle_type> <year> <month> [simulation_range] [acceptance_function]"
            exit 1
        fi
        run_experiment $2 $3 $4 $5 $6
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
    "help"|*)
        show_help
        ;;
esac 