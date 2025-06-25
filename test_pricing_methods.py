#!/usr/bin/env python3
"""
Test script to verify pricing methods work correctly.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all pricing methods can be imported."""
    print("🧪 Testing pricing method imports...")
    
    try:
        from pricing_methods import HikimaMinMaxCostFlow, MAPS, LinUCB, LinearProgram
        from pricing_methods.base_method import BasePricingMethod, PricingResult
        print("✅ All pricing methods imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_instantiation():
    """Test that all pricing methods can be instantiated."""
    print("🧪 Testing pricing method instantiation...")
    
    try:
        from pricing_methods import HikimaMinMaxCostFlow, MAPS, LinUCB, LinearProgram
        
        # Test instantiation with default parameters
        hikima = HikimaMinMaxCostFlow()
        maps = MAPS()
        linucb = LinUCB()
        lp = LinearProgram()
        
        print(f"✅ HikimaMinMaxCostFlow: {hikima.method_name}")
        print(f"✅ MAPS: {maps.method_name}")
        print(f"✅ LinUCB: {linucb.method_name}")
        print(f"✅ LinearProgram: {lp.method_name}")
        
        return True
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False

def test_basic_calculation():
    """Test basic price calculation with dummy data."""
    print("🧪 Testing basic price calculation...")
    
    try:
        from pricing_methods import HikimaMinMaxCostFlow, MAPS, LinUCB
        
        # Create dummy data
        requesters_data = pd.DataFrame({
            'trip_distance': [2.5, 3.1, 1.8],
            'total_amount': [15.0, 18.5, 12.0],
            'PULocationID': [161, 162, 161]
        })
        
        taxis_data = pd.DataFrame({
            'trip_distance': [2.0, 2.8],
            'total_amount': [14.0, 16.0],
            'PULocationID': [161, 162]
        })
        
        # Simple distance matrix (3 requesters x 2 taxis)
        distance_matrix = np.array([
            [0.5, 1.2],
            [0.8, 0.4],
            [0.3, 1.0]
        ])
        
        # Test LinUCB (fastest method)
        linucb = LinUCB()
        result = linucb.calculate_prices(requesters_data, taxis_data, distance_matrix)
        
        print(f"✅ LinUCB calculation successful:")
        print(f"   - Prices: {len(result.prices)} values")
        print(f"   - Objective value: {result.objective_value:.2f}")
        print(f"   - Computation time: {result.computation_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Basic calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔬 Testing Pricing Methods Benchmark System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_instantiation,
        test_basic_calculation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"✅ Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! System ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 