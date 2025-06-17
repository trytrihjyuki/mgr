#!/usr/bin/env python3
"""
Comprehensive data manager for NYC taxi data.
Handles multiple years, months, and vehicle types with parallel downloads.
"""

import requests
import pandas as pd
import os
import sys
import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    vehicle_type: str
    year: int
    month: int
    url: str
    local_parquet: Path
    local_csv: Path
    expected_size_mb: Optional[int] = None

class NYCTaxiDataManager:
    """
    Manages NYC taxi data downloads and conversions.
    Supports Yellow Taxi, Green Taxi, and For-Hire Vehicle data.
    """
    
    # URL patterns for different data types
    URL_PATTERNS = {
        'yellow': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet',
        'green': 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet',
        'fhv': 'https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_{year}-{month:02d}.parquet',
        'fhvhv': 'https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{year}-{month:02d}.parquet'
    }
    
    # Data availability ranges (approximate)
    DATA_RANGES = {
        'yellow': {'start_year': 2009, 'end_year': 2024},
        'green': {'start_year': 2013, 'end_year': 2024},
        'fhv': {'start_year': 2015, 'end_year': 2024},
        'fhvhv': {'start_year': 2019, 'end_year': 2024}
    }
    
    def __init__(self, data_dir: Union[str, Path] = "../data", max_workers: int = 4, use_subdirs: bool = False):
        """
        Initialize data manager.
        
        Args:
            data_dir: Directory to store data files
            max_workers: Maximum number of parallel download threads
            use_subdirs: Whether to use subdirectories (False for compatibility with existing setup)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.use_subdirs = use_subdirs
        
        # Create subdirectories only if requested
        if use_subdirs:
            (self.data_dir / "parquet").mkdir(exist_ok=True)
            (self.data_dir / "csv").mkdir(exist_ok=True)
        
        # Always create metadata directory
        (self.data_dir / "metadata").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_dataset_info(self, vehicle_types: List[str], years: List[int], 
                            months: List[int]) -> List[DatasetInfo]:
        """
        Generate list of datasets to download.
        
        Args:
            vehicle_types: List of vehicle types ('yellow', 'green', 'fhv', 'fhvhv')
            years: List of years
            months: List of months (1-12)
            
        Returns:
            List of DatasetInfo objects
        """
        datasets = []
        
        for vehicle_type in vehicle_types:
            if vehicle_type not in self.URL_PATTERNS:
                self.logger.warning(f"Unknown vehicle type: {vehicle_type}")
                continue
                
            for year in years:
                # Check if year is in valid range
                data_range = self.DATA_RANGES.get(vehicle_type, {})
                start_year = data_range.get('start_year', 2009)
                end_year = data_range.get('end_year', 2024)
                
                if year < start_year or year > end_year:
                    self.logger.warning(f"Year {year} may not be available for {vehicle_type}")
                
                for month in months:
                    if not (1 <= month <= 12):
                        self.logger.warning(f"Invalid month: {month}")
                        continue
                    
                    url = self.URL_PATTERNS[vehicle_type].format(year=year, month=month)
                    
                    # Generate file paths (with or without subdirectories)
                    filename_base = f"{vehicle_type}_tripdata_{year}-{month:02d}"
                    if self.use_subdirs:
                        parquet_path = self.data_dir / "parquet" / f"{filename_base}.parquet"
                        csv_path = self.data_dir / "csv" / f"{filename_base}.csv"
                    else:
                        # Compatible with existing setup - files in root data directory
                        parquet_path = self.data_dir / f"{filename_base}.parquet"
                        csv_path = self.data_dir / f"{filename_base}.csv"
                    
                    datasets.append(DatasetInfo(
                        vehicle_type=vehicle_type,
                        year=year,
                        month=month,
                        url=url,
                        local_parquet=parquet_path,
                        local_csv=csv_path
                    ))
        
        return datasets
    
    def check_url_availability(self, url: str, timeout: int = 10) -> Tuple[bool, Optional[int]]:
        """
        Check if a URL is available and get file size.
        
        Args:
            url: URL to check
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (is_available, file_size_bytes)
        """
        try:
            response = requests.head(url, timeout=timeout)
            if response.status_code == 200:
                content_length = response.headers.get('content-length')
                file_size = int(content_length) if content_length else None
                return True, file_size
            else:
                return False, None
        except Exception as e:
            self.logger.debug(f"URL check failed for {url}: {e}")
            return False, None
    
    def verify_dataset_availability(self, datasets: List[DatasetInfo], 
                                  save_report: bool = True) -> Dict:
        """
        Verify which datasets are available online.
        
        Args:
            datasets: List of datasets to check
            save_report: Whether to save availability report
            
        Returns:
            Dictionary with availability information
        """
        self.logger.info(f"🔍 Checking availability of {len(datasets)} datasets...")
        
        available = []
        unavailable = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all URL checks
            future_to_dataset = {
                executor.submit(self.check_url_availability, dataset.url): dataset
                for dataset in datasets
            }
            
            # Process results
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    is_available, file_size = future.result()
                    
                    if is_available:
                        dataset.expected_size_mb = file_size // (1024*1024) if file_size else None
                        available.append(dataset)
                        size_str = f" ({dataset.expected_size_mb} MB)" if dataset.expected_size_mb else ""
                        self.logger.info(f"✅ {dataset.vehicle_type} {dataset.year}-{dataset.month:02d}{size_str}")
                    else:
                        unavailable.append(dataset)
                        self.logger.warning(f"❌ {dataset.vehicle_type} {dataset.year}-{dataset.month:02d}")
                        
                except Exception as e:
                    unavailable.append(dataset)
                    self.logger.error(f"❌ {dataset.vehicle_type} {dataset.year}-{dataset.month:02d}: {e}")
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checked': len(datasets),
            'available_count': len(available),
            'unavailable_count': len(unavailable),
            'available_datasets': [
                {
                    'vehicle_type': d.vehicle_type,
                    'year': d.year,
                    'month': d.month,
                    'url': d.url,
                    'size_mb': d.expected_size_mb
                }
                for d in available
            ],
            'unavailable_datasets': [
                {
                    'vehicle_type': d.vehicle_type,
                    'year': d.year,
                    'month': d.month,
                    'url': d.url
                }
                for d in unavailable
            ]
        }
        
        # Save report
        if save_report:
            report_path = self.data_dir / "metadata" / "availability_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"📊 Availability report saved to {report_path}")
        
        self.logger.info(f"📊 Summary: {len(available)}/{len(datasets)} datasets available")
        
        return report
    
    def download_with_progress(self, dataset: DatasetInfo, chunk_size: int = 8192) -> bool:
        """Download a single dataset with progress tracking."""
        try:
            # Check if file already exists
            if dataset.local_parquet.exists():
                file_size = dataset.local_parquet.stat().st_size
                self.logger.info(f"✅ {dataset.local_parquet.name} already exists ({file_size // (1024*1024)} MB)")
                return True
            
            self.logger.info(f"🔄 Downloading {dataset.local_parquet.name}...")
            
            # Start download
            response = requests.get(dataset.url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                self.logger.info(f"   📦 File size: {total_size // (1024*1024)} MB")
            
            downloaded = 0
            start_time = time.time()
            
            with open(dataset.local_parquet, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            elapsed = time.time() - start_time
            final_size = dataset.local_parquet.stat().st_size
            avg_speed = final_size / elapsed / (1024*1024)
            
            self.logger.info(f"   ✅ Complete! {final_size // (1024*1024)} MB in {elapsed:.1f}s ({avg_speed:.1f} MB/s)")
            return True
            
        except Exception as e:
            self.logger.error(f"   ❌ Download failed: {e}")
            # Clean up partial file
            if dataset.local_parquet.exists():
                dataset.local_parquet.unlink()
            return False
    
    def convert_parquet_to_csv(self, dataset: DatasetInfo) -> bool:
        """Convert parquet file to CSV."""
        try:
            # Check if CSV already exists
            if dataset.local_csv.exists():
                csv_size = dataset.local_csv.stat().st_size
                self.logger.info(f"✅ {dataset.local_csv.name} already exists ({csv_size // (1024*1024)} MB)")
                return True
            
            if not dataset.local_parquet.exists():
                self.logger.error(f"❌ Parquet file not found: {dataset.local_parquet}")
                return False
            
            self.logger.info(f"🔄 Converting {dataset.local_parquet.name} to CSV...")
            
            start_time = time.time()
            
            # Read and convert
            df = pd.read_parquet(dataset.local_parquet)
            df.to_csv(dataset.local_csv, index=True)
            
            elapsed = time.time() - start_time
            csv_size = dataset.local_csv.stat().st_size
            
            self.logger.info(f"   ✅ Conversion complete! {len(df):,} rows, {csv_size // (1024*1024)} MB in {elapsed:.1f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"   ❌ Conversion failed: {e}")
            # Clean up partial file
            if dataset.local_csv.exists():
                dataset.local_csv.unlink()
            return False
    
    def download_datasets(self, datasets: List[DatasetInfo], 
                         convert_to_csv: bool = True) -> Dict:
        """
        Download multiple datasets in parallel.
        
        Args:
            datasets: List of datasets to download
            convert_to_csv: Whether to convert parquet files to CSV
            
        Returns:
            Dictionary with download results
        """
        self.logger.info(f"📥 Starting download of {len(datasets)} datasets...")
        
        downloaded = []
        failed = []
        converted = []
        conversion_failed = []
        
        # Download phase
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.download_with_progress, dataset): dataset
                for dataset in datasets
            }
            
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    if future.result():
                        downloaded.append(dataset)
                    else:
                        failed.append(dataset)
                except Exception as e:
                    self.logger.error(f"Download error for {dataset.local_parquet.name}: {e}")
                    failed.append(dataset)
        
        # Conversion phase
        if convert_to_csv and downloaded:
            self.logger.info(f"🔄 Converting {len(downloaded)} files to CSV...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_dataset = {
                    executor.submit(self.convert_parquet_to_csv, dataset): dataset
                    for dataset in downloaded
                }
                
                for future in as_completed(future_to_dataset):
                    dataset = future_to_dataset[future]
                    try:
                        if future.result():
                            converted.append(dataset)
                        else:
                            conversion_failed.append(dataset)
                    except Exception as e:
                        self.logger.error(f"Conversion error for {dataset.local_parquet.name}: {e}")
                        conversion_failed.append(dataset)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_requested': len(datasets),
            'downloaded_count': len(downloaded),
            'download_failed_count': len(failed),
            'converted_count': len(converted),
            'conversion_failed_count': len(conversion_failed),
            'downloaded_datasets': [
                {
                    'vehicle_type': d.vehicle_type,
                    'year': d.year,
                    'month': d.month,
                    'parquet_path': str(d.local_parquet),
                    'csv_path': str(d.local_csv) if convert_to_csv else None
                }
                for d in downloaded
            ]
        }
        
        # Save summary
        summary_path = self.data_dir / "metadata" / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Download Summary:")
        self.logger.info(f"   Downloaded: {len(downloaded)}/{len(datasets)}")
        if convert_to_csv:
            self.logger.info(f"   Converted to CSV: {len(converted)}/{len(downloaded)}")
        self.logger.info(f"   Summary saved to: {summary_path}")
        
        return summary

def main():
    """Command line interface for data manager."""
    parser = argparse.ArgumentParser(description='NYC Taxi Data Manager')
    
    parser.add_argument('--vehicle-types', nargs='+', 
                       choices=['yellow', 'green', 'fhv', 'fhvhv'],
                       default=['yellow', 'green'],
                       help='Vehicle types to process')
    
    parser.add_argument('--years', nargs='+', type=int,
                       default=[2019],
                       help='Years to process')
    
    parser.add_argument('--months', nargs='+', type=int,
                       default=list(range(1, 13)),
                       help='Months to process (1-12, default: all months)')
    
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Data directory')
    
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers')
    
    parser.add_argument('--check-only', action='store_true',
                       help='Only check availability, do not download')
    
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV conversion')
    
    args = parser.parse_args()
    
    # Initialize data manager
    manager = NYCTaxiDataManager(data_dir=args.data_dir, max_workers=args.max_workers)
    
    # Generate dataset list
    datasets = manager.generate_dataset_info(
        vehicle_types=args.vehicle_types,
        years=args.years,
        months=args.months
    )
    
    if not datasets:
        print("❌ No datasets to process")
        return
    
    print(f"🚕 NYC Taxi Data Manager")
    print(f"   Vehicle types: {args.vehicle_types}")
    print(f"   Years: {args.years}")
    print(f"   Months: {args.months}")
    print(f"   Total datasets: {len(datasets)}")
    print()
    
    # Check availability
    report = manager.verify_dataset_availability(datasets)
    
    if args.check_only:
        print(f"✅ Availability check complete. Report saved to data/metadata/")
        return
    
    # Download available datasets
    available_datasets = [
        d for d in datasets 
        if any(a['vehicle_type'] == d.vehicle_type and 
               a['year'] == d.year and 
               a['month'] == d.month 
               for a in report['available_datasets'])
    ]
    
    if available_datasets:
        manager.download_datasets(
            datasets=available_datasets,
            convert_to_csv=not args.no_csv
        )
    else:
        print("❌ No available datasets to download")

if __name__ == "__main__":
    main() 