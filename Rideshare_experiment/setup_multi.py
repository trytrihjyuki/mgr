#!/usr/bin/env python3
"""
Multi-dataset setup script for NYC taxi data.
Downloads and prepares data for multiple years, months, and vehicle types.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add bin directory to path for imports
sys.path.append(str(Path(__file__).parent / "bin"))

from data_manager import NYCTaxiDataManager

def main():
    """Main setup function with command line interface."""
    parser = argparse.ArgumentParser(description='Multi-Dataset NYC Taxi Data Setup')
    
    parser.add_argument('--vehicle-types', nargs='+',
                       choices=['yellow', 'green', 'fhv', 'fhvhv'],
                       default=['yellow', 'green'],
                       help='Vehicle types to download (default: yellow green)')
    
    parser.add_argument('--years', nargs='+', type=int,
                       default=[2019],
                       help='Years to download (default: 2019)')
    
    parser.add_argument('--months', nargs='+', type=int,
                       default=[10],
                       help='Months to download (default: 10)')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers (default: 4)')
    
    parser.add_argument('--check-only', action='store_true',
                       help='Only check availability, do not download')
    
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV conversion')
    
    args = parser.parse_args()
    
    print("🚕 Multi-Dataset NYC Taxi Data Setup")
    print("=" * 50)
    print(f"Vehicle types: {args.vehicle_types}")
    print(f"Years: {args.years}")
    print(f"Months: {args.months}")
    print(f"Data directory: {args.data_dir}")
    print(f"Max workers: {args.max_workers}")
    print()
    
    # Initialize data manager (compatible with existing structure)
    manager = NYCTaxiDataManager(data_dir=args.data_dir, max_workers=args.max_workers, use_subdirs=False)
    
    # Generate dataset list
    datasets = manager.generate_dataset_info(
        vehicle_types=args.vehicle_types,
        years=args.years,
        months=args.months
    )
    
    if not datasets:
        print("❌ No datasets to process")
        return 1
    
    print(f"📋 Generated {len(datasets)} dataset configurations")
    
    # Check availability
    print("\n🔍 Checking dataset availability...")
    report = manager.verify_dataset_availability(datasets)
    
    available_count = report['available_count']
    total_count = report['total_checked']
    
    print(f"\n📊 Availability Summary:")
    print(f"   Available: {available_count}/{total_count} ({available_count/total_count*100:.1f}%)")
    
    if args.check_only:
        print(f"\n✅ Availability check complete")
        print(f"   Report saved to: {args.data_dir}/metadata/availability_report.json")
        return 0
    
    if available_count == 0:
        print("\n❌ No datasets available for download")
        return 1
    
    # Download available datasets
    available_datasets = [
        d for d in datasets 
        if any(a['vehicle_type'] == d.vehicle_type and 
               a['year'] == d.year and 
               a['month'] == d.month 
               for a in report['available_datasets'])
    ]
    
    print(f"\n📥 Starting download of {len(available_datasets)} available datasets...")
    
    download_summary = manager.download_datasets(
        datasets=available_datasets,
        convert_to_csv=not args.no_csv
    )
    
    # Final summary
    print(f"\n🎉 Setup Complete!")
    print(f"   Downloaded: {download_summary['downloaded_count']}/{len(available_datasets)}")
    if not args.no_csv:
        print(f"   Converted to CSV: {download_summary['converted_count']}/{download_summary['downloaded_count']}")
    
    print(f"\n📊 Summary saved to: {args.data_dir}/metadata/download_summary.json")
    print(f"\n✅ Ready to run experiments!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 