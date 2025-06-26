#!/usr/bin/env python3
"""
Test script to validate AWS setup and Lambda connectivity
"""

import boto3
import json
import sys

def test_aws_connectivity():
    """Test basic AWS connectivity"""
    print("🔍 Testing AWS Connectivity...")
    
    try:
        # Test STS (credentials)
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Identity: {identity['Arn']}")
        
        # Test S3 access
        s3 = boto3.client('s3')
        try:
            s3.head_bucket(Bucket='magisterka')
            print("✅ S3 bucket 'magisterka' accessible")
        except Exception as e:
            print(f"❌ S3 bucket error: {e}")
            return False
        
        # Test Lambda function exists
        lambda_client = boto3.client('lambda')
        try:
            response = lambda_client.get_function(FunctionName='rideshare-pricing-benchmark')
            print(f"✅ Lambda function 'rideshare-pricing-benchmark' found")
            print(f"   Status: {response['Configuration']['State']}")
            print(f"   Runtime: {response['Configuration']['Runtime']}")
            print(f"   Memory: {response['Configuration']['MemorySize']} MB")
            print(f"   Timeout: {response['Configuration']['Timeout']} seconds")
        except Exception as e:
            print(f"❌ Lambda function error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AWS setup error: {e}")
        return False

def test_lambda_invocation():
    """Test Lambda function with a simple test payload"""
    print("\n🧪 Testing Lambda Function Invocation...")
    
    lambda_client = boto3.client('lambda')
    
    # Create a simple test event
    test_event = {
        'test_mode': True,
        'year': 2019,
        'month': 10,
        'day': 6,
        'borough': 'Manhattan',
        'vehicle_type': 'yellow'
    }
    
    try:
        print("📤 Invoking Lambda function with test payload...")
        response = lambda_client.invoke(
            FunctionName='rideshare-pricing-benchmark',
            Payload=json.dumps(test_event)
        )
        
        # Read response
        result = json.loads(response['Payload'].read())
        
        if result.get('statusCode') == 200:
            print("✅ Lambda function invocation successful")
            
            # Parse the body if it's a string
            body = result.get('body', '')
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except:
                    pass
            
            print(f"📊 Response: {body}")
            return True
        else:
            print(f"❌ Lambda function returned error: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Lambda invocation error: {e}")
        return False

def test_cli_configuration():
    """Test CLI configuration matches reality"""
    print("\n⚙️ Testing CLI Configuration...")
    
    # Import our CLI to check function name
    try:
        from run_pricing_experiment import PricingExperimentCLI
        cli = PricingExperimentCLI()
        
        print(f"✅ CLI configured for Lambda function: {cli.function_name}")
        print(f"✅ CLI configured for S3 bucket: {cli.bucket_name}")
        
        # Test a simple dry run
        print("🔍 Testing dry run functionality...")
        import subprocess
        result = subprocess.run([
            'python', 'run_pricing_experiment.py',
            '--year=2019', '--month=10', '--day=6',
            '--borough=Manhattan', '--vehicle_type=yellow',
            '--eval=PL', '--methods=MAPS',
            '--dry_run'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI dry run works correctly")
            return True
        else:
            print(f"❌ CLI dry run failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI configuration error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 RIDE-HAILING PRICING SYSTEM - SETUP VALIDATION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: AWS Connectivity
    if not test_aws_connectivity():
        all_tests_passed = False
    
    # Test 2: Lambda Invocation
    if not test_lambda_invocation():
        all_tests_passed = False
    
    # Test 3: CLI Configuration
    if not test_cli_configuration():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! System is ready for experiments.")
        print("\n💡 Next steps:")
        print("   1. Ensure you have taxi data in S3:")
        print("      aws s3 ls s3://magisterka/datasets/yellow/year=2019/month=10/")
        print("   2. Run a small test experiment:")
        print("      python run_pricing_experiment.py --year=2019 --month=10 --day=6 --borough=Manhattan --vehicle_type=yellow --eval=PL --methods=MAPS")
        
        return 0
    else:
        print("❌ SOME TESTS FAILED! Please fix the issues above before running experiments.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 