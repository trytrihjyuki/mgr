#!/bin/bash

# Test script for progress extraction regex patterns

# Create a temporary log file with sample Python script output
TEST_LOG="test_progress.log"

cat > "$TEST_LOG" << 'EOF'
🚀 CLOUD EXECUTION: 576 scenarios, 1 parallel workers, BATCH mode (size=10)
📦 Created 58 batches of up to 10 scenarios each
🎯 Starting batch submission...
⚡ Progress: 155/576 (26.9%) | Rate: 0.12/s | ETA: 55m
⚡ [160/???] ✅160 ❌0 | Rate: 0.1/s
⚡ Batch batch_005: [70/???] ✅70 ❌0 | Rate: 0.2/s
day1_PL evaluation starting...
day1_Sigmoid evaluation starting...
📈 SUCCESS RATE: 98.5%
✅ Results saved to S3
EOF

# Source the extract_batch_progress function from the main script
source enhanced_parallel_experiments.sh 2>/dev/null || echo "Warning: Could not source main script"

# Test the function
echo "Testing progress extraction..."
echo "============================="

echo "Test 1: Basic progress extraction (for display)"
result=$(extract_batch_progress "$TEST_LOG" false)
echo "Result: '$result'"
echo ""

echo "Test 2: Progress extraction for comparison (numeric only)" 
result=$(extract_batch_progress "$TEST_LOG" true)
echo "Result: '$result'"
echo ""

echo "Test 3: Debug mode enabled"
result=$(extract_batch_progress "$TEST_LOG" true true)
echo "Result: '$result'"
echo ""

# Clean up
rm -f "$TEST_LOG"

echo "Test completed!" 