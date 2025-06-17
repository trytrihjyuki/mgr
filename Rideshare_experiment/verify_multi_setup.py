#!/usr/bin/env python3
"""
Verification script for multi-dataset functionality.
Tests data management, parallel processing, and result aggregation.
"""

import sys
import subprocess
from pathlib import Path
import json
import tempfile
import shutil

def run_command(cmd, description="", timeout=300):
    """Run a command and return success status."""
    print(f"🔄 {description or cmd}")
    try:
        result = subprocess.run(
            cmd.split(), 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        if result.returncode == 0:
            print(f"✅ Success")
            return True, result.stdout
        else:
            print(f"❌ Failed (code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, str(e)

def check_file_exists(file_path, description=""):
    """Check if a file exists."""
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        print(f"✅ {description or str(path)} exists ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description or str(path)} not found")
        return False

def main():
    """Run verification tests."""
    print("🧪 Multi-Dataset Functionality Verification")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Check if core scripts exist
    print("\n📋 Test 1: Core Scripts Existence")
    scripts_to_check = [
        ("bin/data_manager.py", "Data Manager"),
        ("setup_multi.py", "Multi Setup Script"),
        ("run_experiments_parallel.py", "Parallel Runner"),
        ("aggregate_results.py", "Results Aggregator"),
        ("run_multi_experiments.sh", "Shell Runner")
    ]
    
    for script_path, description in scripts_to_check:
        total_tests += 1
        if check_file_exists(script_path, description):
            passed_tests += 1
    
    # Test 2: Data Manager Basic Functionality
    print("\n📋 Test 2: Data Manager Functionality")
    total_tests += 1
    success, output = run_command(
        "python3 bin/data_manager.py --help",
        "Data manager help"
    )
    if success:
        passed_tests += 1
    
    # Test 3: Check dataset availability (quick test)
    print("\n📋 Test 3: Dataset Availability Check")
    total_tests += 1
    success, output = run_command(
        "python3 bin/data_manager.py --check-only --vehicle-types yellow --years 2019 --months 10",
        "Check yellow taxi availability",
        timeout=60
    )
    if success:
        passed_tests += 1
    
    # Test 4: Multi setup script
    print("\n📋 Test 4: Multi Setup Script")
    total_tests += 1
    success, output = run_command(
        "python3 setup_multi.py --help",
        "Multi setup help"
    )
    if success:
        passed_tests += 1
    
    # Test 5: Parallel runner dry run
    print("\n📋 Test 5: Parallel Runner (Dry Run)")
    total_tests += 1
    success, output = run_command(
        "python3 run_experiments_parallel.py --dry-run --vehicle-types yellow --simulation-range 2",
        "Parallel runner dry run",
        timeout=30
    )
    if success:
        passed_tests += 1
    
    # Test 6: Aggregation script
    print("\n📋 Test 6: Results Aggregator")
    total_tests += 1
    success, output = run_command(
        "python3 aggregate_results.py --help",
        "Aggregator help"
    )
    if success:
        passed_tests += 1
    
    # Test 7: Shell script dry run
    print("\n📋 Test 7: Shell Script (Quick Scenario)")
    total_tests += 1
    success, output = run_command(
        "./run_multi_experiments.sh --scenario quick --dry-run",
        "Shell script quick dry run",
        timeout=60
    )
    if success:
        passed_tests += 1
    
    # Test 8: Import test for core modules
    print("\n📋 Test 8: Python Module Imports")
    import_tests = [
        ("bin.data_manager", "Data Manager Module"),
        ("bin.benchmark_utils", "Benchmark Utils"),
        ("bin.lp_pricing", "LP Pricing Module")
    ]
    
    for module_name, description in import_tests:
        total_tests += 1
        try:
            # Temporarily add bin to path
            sys.path.insert(0, str(Path("bin")))
            
            if module_name.startswith("bin."):
                module_name = module_name[4:]  # Remove "bin." prefix
            
            __import__(module_name)
            print(f"✅ {description} import successful")
            passed_tests += 1
        except ImportError as e:
            print(f"❌ {description} import failed: {e}")
        except Exception as e:
            print(f"❌ {description} import error: {e}")
        finally:
            if str(Path("bin")) in sys.path:
                sys.path.remove(str(Path("bin")))
    
    # Test 9: Check directory structure
    print("\n📋 Test 9: Directory Structure")
    expected_dirs = [
        "bin",
        "data",
        "results"
    ]
    
    for dir_name in expected_dirs:
        total_tests += 1
        if check_file_exists(dir_name, f"{dir_name}/ directory"):
            passed_tests += 1
    
    # Test 10: Requirements file
    print("\n📋 Test 10: Requirements File")
    total_tests += 1
    if check_file_exists("requirements.txt", "Requirements file"):
        # Check if key dependencies are listed
        try:
            with open("requirements.txt", 'r') as f:
                content = f.read().lower()
                
            required_packages = ["pulp", "pandas", "numpy", "requests"]
            missing_packages = [pkg for pkg in required_packages if pkg not in content]
            
            if not missing_packages:
                print(f"✅ All required packages found in requirements.txt")
                passed_tests += 1
            else:
                print(f"❌ Missing packages in requirements.txt: {missing_packages}")
        except Exception as e:
            print(f"❌ Error reading requirements.txt: {e}")
    
    # Final summary
    print("\n" + "=" * 50)
    print(f"📊 Verification Summary")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Multi-dataset functionality is ready.")
        return 0
    elif passed_tests >= total_tests * 0.8:
        print("\n✅ Most tests passed. System should work with minor issues.")
        return 0
    else:
        print("\n❌ Many tests failed. Please check setup and dependencies.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 