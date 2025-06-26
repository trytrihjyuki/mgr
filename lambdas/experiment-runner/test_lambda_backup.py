"""
Simple test Lambda function to debug numpy import issues.
"""

import json
import sys
import os

def lambda_handler(event, context):
    """Test Lambda handler for debugging imports."""
    
    try:
        # Print system info
        info = {
            "python_path": sys.path,
            "environment_vars": dict(os.environ),
            "working_directory": os.getcwd()
        }
        
        # Test basic imports first
        import pandas as pd
        info["pandas_version"] = pd.__version__
        info["pandas_location"] = pd.__file__
        
        # Test numpy import
        import numpy as np
        info["numpy_version"] = np.__version__
        info["numpy_location"] = np.__file__
        
        return {
            'statusCode': 200,
            'body': json.dumps(info, indent=2),
            'headers': {'Content-Type': 'application/json'}
        }
        
    except Exception as e:
        import traceback
        return {
            'statusCode': 500,
            'body': json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "python_path": sys.path,
                "working_directory": os.getcwd(),
                "environment_vars": dict(os.environ)
            }, indent=2),
            'headers': {'Content-Type': 'application/json'}
        } 