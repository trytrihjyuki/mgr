#!/bin/bash

# Quick test script for refactored LP-based experiments

echo "=== Quick Test of Refactored LP Experiments ==="

cd bin

# Test with small simulation range for quick validation
SIMULATION_RANGE=5

echo "Installing/checking dependencies..."
pip install -r ../requirements.txt

echo ""
echo "=== Testing PL Experiment ==="
python3 experiment_PL_refactored.py Manhattan 6 30 s $SIMULATION_RANGE

echo ""
echo "=== Testing Sigmoid Experiment ==="
python3 experiment_Sigmoid_refactored.py Manhattan 6 30 s $SIMULATION_RANGE

echo ""
echo "=== Test Completed ==="
echo "Check ../results/ directory for outputs" 