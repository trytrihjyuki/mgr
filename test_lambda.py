#!/usr/bin/env python3
"""
Test script for the updated Hikima pricing benchmark Lambda function.
Tests the new data loading and experiment structure.
"""

import json
from datetime import datetime

# Mock event for testing
def create_test_event():
    return {
        "test_mode": False,
        "execution_date": "20241201_120000",
        "training_id": "test_12345",
        "year": 2019,
        "month": 10,
        "day": 1,
        "borough": "Manhattan", 
        "time_window": {
            "hour_start": 10,
            "hour_end": 20,
            "minute_start": 0,
            "time_interval": 5
        },
        "scenario_index": 0,
        "vehicle_type": "yellow",
        "acceptance_function": "PL",
        "methods": ["MinMaxCostFlow", "MAPS", "LinUCB", "LP"]
    }

# Mock context
class MockContext:
    def get_remaining_time_in_millis(self):
        return 900000  # 15 minutes

def test_lambda_structure():
    """Test the Lambda function structure and data loading."""
    try:
        # Import the lambda function
        import sys
        sys.path.append('lambdas/pricing-benchmark')
        from lambda_function import lambda_handler, HikimaExperimentRunner
        
        print("✅ Lambda function imports successful")
        
        # Test experiment runner initialization
        runner = HikimaExperimentRunner()
        print("✅ HikimaExperimentRunner initialization successful")
        
        # Test area information loading (should handle missing file gracefully)
        area_info = runner.load_area_information()
        print(f"✅ Area information loaded: {len(area_info)} areas")
        
        # Create test event
        event = create_test_event()
        context = MockContext()
        
        print("\n🧪 Test Event:")
        print(json.dumps(event, indent=2))
        
        # Test with test mode first
        test_event = event.copy()
        test_event["test_mode"] = True
        
        response = lambda_handler(test_event, context)
        print(f"\n✅ Test mode response: {response['statusCode']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_parameters():
    """Test different experiment parameter combinations."""
    test_cases = [
        {
            "name": "Manhattan Yellow PL",
            "params": {
                "borough": "Manhattan",
                "vehicle_type": "yellow", 
                "acceptance_function": "PL",
                "year": 2019,
                "month": 10,
                "day": 1
            }
        },
        {
            "name": "Bronx Green Sigmoid",
            "params": {
                "borough": "Bronx",
                "vehicle_type": "green",
                "acceptance_function": "Sigmoid", 
                "year": 2019,
                "month": 10,
                "day": 6
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        event = create_test_event()
        event.update(test_case['params'])
        
        # Test parameter parsing
        try:
            year = event.get('year', 2019)
            month = event.get('month', 10)
            day = event.get('day', 1)
            borough = event.get('borough', 'Manhattan')
            vehicle_type = event.get('vehicle_type', 'green')
            acceptance_function = event.get('acceptance_function', 'PL')
            
            print(f"  📅 Date: {year}-{month:02d}-{day:02d}")
            print(f"  🏙️ Borough: {borough}")
            print(f"  🚕 Vehicle: {vehicle_type}")
            print(f"  📊 Acceptance: {acceptance_function}")
            print(f"  ✅ Parameters parsed successfully")
            
        except Exception as e:
            print(f"  ❌ Parameter parsing failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Hikima Lambda Function Structure")
    print("=" * 50)
    
    # Test basic structure
    if test_lambda_structure():
        print("\n✅ Basic structure tests passed")
    else:
        print("\n❌ Basic structure tests failed")
        exit(1)
    
    # Test experiment parameters
    test_experiment_parameters()
    
    print("\n🎉 All tests completed!")
    print("\nTo run a full experiment, use an event like:")
    
    example_event = create_test_event()
    print(json.dumps(example_event, indent=2)) 