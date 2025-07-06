#!/usr/bin/env python3
"""
Debug Lambda Invocation Script
Test lambda function directly to identify hanging issues
"""
import boto3
import json
import time
from datetime import datetime

def test_lambda_invocation():
    """Test the lambda function directly"""
    print("🔍 Testing Lambda Function Invocation...")
    
    # Create Lambda client
    lambda_client = boto3.client('lambda', region_name='eu-north-1')
    
    # Simple test payload
    test_payload = {
        "year": 2019,
        "month": 10,
        "day": 1,
        "borough": "Manhattan",
        "vehicle_type": "yellow",
        "acceptance_function": "PL",
        "methods": ["LP"],
        "scenario_index": 0,
        "time_window": {
            "hour_start": 10,
            "hour_end": 20,
            "time_interval": 5,
            "time_unit": "m"
        },
        "training_id": "debug_test",
        "num_eval": 5  # Very small number for quick test
    }
    
    print(f"📤 Invoking Lambda with payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = lambda_client.invoke(
            FunctionName='rideshare-pricing-benchmark',
            InvocationType='RequestResponse',
            Payload=json.dumps(test_payload)
        )
        
        execution_time = time.time() - start_time
        print(f"⏱️ Lambda execution time: {execution_time:.2f}s")
        
        # Read response
        response_payload = response['Payload'].read().decode('utf-8')
        print(f"📥 Response status: {response['StatusCode']}")
        
        if response['StatusCode'] == 200:
            print("✅ Lambda invocation successful!")
            result = json.loads(response_payload)
            print(f"🎯 Result keys: {list(result.keys())}")
            
            if 'success' in result:
                print(f"📊 Success: {result['success']}")
            if 'results' in result:
                print(f"📈 Results count: {len(result.get('results', []))}")
                
        else:
            print(f"❌ Lambda invocation failed with status: {response['StatusCode']}")
            print(f"📄 Response: {response_payload}")
            
    except Exception as e:
        print(f"❌ Lambda invocation error: {e}")
        import traceback
        traceback.print_exc()

def test_s3_access():
    """Test S3 data access"""
    print("\n🔍 Testing S3 Data Access...")
    
    s3_client = boto3.client('s3', region_name='eu-north-1')
    
    # Test data file access
    test_files = [
        'datasets/yellow/year=2019/month=10/yellow_tripdata_2019-10.parquet',
        'datasets/area_information.csv',
        'models/linucb/yellow_Manhattan_201907/trained_model.pkl'
    ]
    
    for file_path in test_files:
        try:
            response = s3_client.head_object(Bucket='magisterka', Key=file_path)
            size_mb = response['ContentLength'] / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ {file_path} - {e}")

def test_lambda_logs():
    """Check recent Lambda logs for errors"""
    print("\n🔍 Checking Lambda Logs...")
    
    logs_client = boto3.client('logs', region_name='eu-north-1')
    
    try:
        # Get recent log events
        end_time = int(time.time() * 1000)
        start_time = end_time - (60 * 60 * 1000)  # Last hour
        
        response = logs_client.filter_log_events(
            logGroupName='/aws/lambda/rideshare-pricing-benchmark',
            startTime=start_time,
            endTime=end_time,
            limit=50
        )
        
        events = response.get('events', [])
        print(f"📋 Found {len(events)} log events in last hour")
        
        # Show recent errors
        error_events = [e for e in events if 'ERROR' in e.get('message', '').upper()]
        if error_events:
            print(f"⚠️ Found {len(error_events)} error events:")
            for event in error_events[-5:]:  # Show last 5 errors
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                print(f"   {timestamp}: {event['message']}")
        else:
            print("✅ No errors found in recent logs")
            
        # Show recent successful events
        success_events = [e for e in events if '✅' in e.get('message', '')]
        if success_events:
            print(f"📈 Found {len(success_events)} successful events")
        else:
            print("⚠️ No successful events found")
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")

def main():
    """Main debug function"""
    print("🚀 LAMBDA DEBUG SCRIPT")
    print("=" * 50)
    
    # Test 1: S3 Access
    test_s3_access()
    
    # Test 2: Lambda Logs
    test_lambda_logs()
    
    # Test 3: Lambda Invocation
    test_lambda_invocation()
    
    print("\n" + "=" * 50)
    print("🎯 Debug complete!")

if __name__ == '__main__':
    main() 