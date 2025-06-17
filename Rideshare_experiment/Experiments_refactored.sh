#!/bin/bash

# Refactored Ride-Hailing Pricing Experiments using Linear Programming
# Based on Gupta-Nagarajan reduction

echo "=== Starting Refactored LP-based Ride-Hailing Experiments ==="

cd bin

# Set parameters
SIMULATION_RANGE=120

echo "Installing/checking dependencies..."
pip install -r ../requirements.txt

echo ""
echo "=== Running PL (Piecewise Linear) Experiments ==="
echo ""

echo "Running Manhattan experiment..."
python3 experiment_PL_refactored.py Manhattan 6 30 s $SIMULATION_RANGE

echo "Running Queens experiment..."
python3 experiment_PL_refactored.py Queens 6 5 m $SIMULATION_RANGE

echo "Running Bronx experiment..."
python3 experiment_PL_refactored.py Bronx 6 5 m $SIMULATION_RANGE

echo "Running Brooklyn experiment..."
python3 experiment_PL_refactored.py Brooklyn 6 5 m $SIMULATION_RANGE

echo ""
echo "=== Running Sigmoid Experiments ==="
echo ""

echo "Running Manhattan experiment..."
python3 experiment_Sigmoid_refactored.py Manhattan 6 30 s $SIMULATION_RANGE

echo "Running Queens experiment..."
python3 experiment_Sigmoid_refactored.py Queens 6 5 m $SIMULATION_RANGE

echo "Running Bronx experiment..."
python3 experiment_Sigmoid_refactored.py Bronx 6 5 m $SIMULATION_RANGE

echo "Running Brooklyn experiment..."
python3 experiment_Sigmoid_refactored.py Brooklyn 6 5 m $SIMULATION_RANGE

echo ""
echo "=== Experiments Completed ==="
echo "Results are saved in ../results/ directory"
echo "  - Summary CSV files: ../results/summary/"
echo "  - Detailed JSON files: ../results/detailed/"
echo "  - Logs: ../results/logs/"
echo "  - Benchmarks: ../results/benchmarks/" 