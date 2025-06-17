#!/usr/bin/env python3
"""
Quick setup verification script.
Checks if all required files and dependencies are available.
"""

import os
import sys
from pathlib import Path

def check_file(filepath: str, description: str, required: bool = True) -> bool:
    """Check if a file exists and show its size."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) // (1024*1024)
        print(f"✅ {description}: {os.path.basename(filepath)} ({size_mb} MB)")
        return True
    else:
        status = "❌" if required else "⚠️ "
        print(f"{status} {description}: {os.path.basename(filepath)} - NOT FOUND")
        return not required

def check_dependency(package: str) -> bool:
    """Check if a Python package is importable."""
    try:
        __import__(package)
        print(f"✅ Python package: {package}")
        return True
    except ImportError:
        print(f"❌ Python package: {package} - NOT FOUND")
        return False

def main():
    """Check setup status."""
    print("🔍 Setup Verification")
    print("=" * 50)
    
    # Check data files
    print("\n📁 Data Files:")
    data_dir = "data"
    
    required_files = [
        (f"{data_dir}/area_information.csv", "NYC area information"),
        (f"{data_dir}/green_tripdata_2019-10.csv", "Green taxi data (CSV)"),
        (f"{data_dir}/yellow_tripdata_2019-10.csv", "Yellow taxi data (CSV)"),
    ]
    
    optional_files = [
        (f"{data_dir}/green_tripdata_2019-10.parquet", "Green taxi data (Parquet)"),
        (f"{data_dir}/yellow_tripdata_2019-10.parquet", "Yellow taxi data (Parquet)"),
    ]
    
    data_ok = True
    for filepath, desc in required_files:
        if not check_file(filepath, desc, required=True):
            data_ok = False
    
    for filepath, desc in optional_files:
        check_file(filepath, desc, required=False)
    
    # Check Python dependencies
    print("\n🐍 Python Dependencies:")
    
    required_packages = [
        "pandas", "numpy", "pulp", "networkx", 
        "pyproj", "requests", "matplotlib"
    ]
    
    optional_packages = ["pyarrow", "scipy", "shapely"]
    
    deps_ok = True
    for package in required_packages:
        if not check_dependency(package):
            deps_ok = False
    
    for package in optional_packages:
        check_dependency(package)
    
    # Check scripts
    print("\n📜 Scripts:")
    
    scripts = [
        ("bin/lp_pricing.py", "LP pricing module"),
        ("bin/benchmark_utils.py", "Benchmarking utilities"),
        ("bin/experiment_PL_refactored.py", "PL experiment"),
        ("bin/experiment_Sigmoid_refactored.py", "Sigmoid experiment"),
        ("bin/analyze_results.py", "Analysis script"),
    ]
    
    scripts_ok = True
    for filepath, desc in scripts:
        if not check_file(filepath, desc, required=True):
            scripts_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP STATUS")
    print("=" * 50)
    
    if data_ok and deps_ok and scripts_ok:
        print("🎉 Setup is COMPLETE!")
        print("\n✅ Ready to run experiments:")
        print("   bash Experiments_test_refactored.sh")
    else:
        print("⚠️  Setup is INCOMPLETE!")
        
        if not data_ok:
            print("\n🔧 To fix data issues:")
            print("   bash setup.sh")
            
        if not deps_ok:
            print("\n🔧 To fix dependency issues:")
            print("   pip install -r requirements.txt")
            
        if not scripts_ok:
            print("\n🔧 Scripts missing - please check file permissions")
    
    print()

if __name__ == "__main__":
    main() 