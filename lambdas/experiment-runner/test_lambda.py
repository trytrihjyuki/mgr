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
        
        # List contents of /opt/python to see what's actually there
        try:
            opt_python_contents = os.listdir('/opt/python') if os.path.exists('/opt/python') else []
            info["opt_python_contents"] = opt_python_contents
        except Exception as e:
            info["opt_python_error"] = str(e)
        
        # Test numpy import directly
        try:
            import numpy as np
            info["numpy_success"] = True
            info["numpy_version"] = np.__version__
            info["numpy_location"] = np.__file__
        except ImportError as e:
            info["numpy_import_error"] = str(e)
        except Exception as e:
            info["numpy_other_error"] = str(e)
        
        # Now try pandas if numpy worked
        if "numpy_success" in info:
            try:
                import pandas as pd
                info["pandas_success"] = True
                info["pandas_version"] = pd.__version__
                info["pandas_location"] = pd.__file__
            except Exception as e:
                info["pandas_error"] = str(e)
        
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