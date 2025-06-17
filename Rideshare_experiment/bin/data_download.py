#!/usr/bin/env python3
"""
Data download script for NYC taxi data with progress tracking.
Downloads green and yellow taxi trip data for October 2019.
"""

import requests
import pandas as pd
import os
import sys
from pathlib import Path
import time
from typing import Optional

def download_with_progress(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        chunk_size: Size of chunks to download at a time
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🔄 Downloading {os.path.basename(filename)}...")
        print(f"   URL: {url}")
        
        # Check if file already exists
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"   ✅ File already exists ({file_size // (1024*1024)} MB)")
            return True
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Start download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            print(f"   📦 File size: {total_size // (1024*1024)} MB")
        
        downloaded = 0
        start_time = time.time()
        last_progress_time = start_time
        
        with open(filename, 'wb') as f:
            for chunk_num, chunk in enumerate(response.iter_content(chunk_size=chunk_size)):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress every 2 seconds or every 1000 chunks
                    current_time = time.time()
                    if (current_time - last_progress_time > 2.0) or (chunk_num % 1000 == 0):
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            speed = downloaded / (current_time - start_time) / (1024*1024)  # MB/s
                            print(f"   📥 Progress: {percent:.1f}% ({downloaded // (1024*1024)} MB) - {speed:.1f} MB/s")
                        else:
                            speed = downloaded / (current_time - start_time) / (1024*1024)  # MB/s
                            print(f"   📥 Downloaded: {downloaded // (1024*1024)} MB - {speed:.1f} MB/s")
                        last_progress_time = current_time
        
        elapsed = time.time() - start_time
        final_size = os.path.getsize(filename)
        avg_speed = final_size / elapsed / (1024*1024)
        
        print(f"   ✅ Complete! {final_size // (1024*1024)} MB in {elapsed:.1f}s ({avg_speed:.1f} MB/s)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def convert_parquet_to_csv(parquet_file: str, csv_file: str) -> bool:
    """
    Convert parquet file to CSV with progress indication.
    
    Args:
        parquet_file: Input parquet file path
        csv_file: Output CSV file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🔄 Converting {os.path.basename(parquet_file)} to CSV...")
        
        # Check if CSV already exists
        if os.path.exists(csv_file):
            csv_size = os.path.getsize(csv_file)
            print(f"   ✅ CSV already exists ({csv_size // (1024*1024)} MB)")
            return True
        
        if not os.path.exists(parquet_file):
            print(f"   ❌ Parquet file not found: {parquet_file}")
            return False
        
        start_time = time.time()
        
        # Read parquet file
        print(f"   📖 Reading parquet file...")
        df = pd.read_parquet(parquet_file)
        
        # Write CSV file
        print(f"   💾 Writing CSV file ({len(df):,} rows)...")
        df.to_csv(csv_file, index=True)
        
        elapsed = time.time() - start_time
        csv_size = os.path.getsize(csv_file)
        
        print(f"   ✅ Conversion complete! {csv_size // (1024*1024)} MB in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        return False

def main():
    """Main function to download and process NYC taxi data."""
    print("=" * 60)
    print("🚕 NYC Taxi Data Download & Preparation")
    print("=" * 60)
    print()
    
    # Define data files
    data_dir = "../data"
    files_to_download = [
        {
            "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2019-10.parquet",
            "parquet": f"{data_dir}/green_tripdata_2019-10.parquet",
            "csv": f"{data_dir}/green_tripdata_2019-10.csv",
            "name": "Green Taxi Data"
        },
        {
            "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-10.parquet", 
            "parquet": f"{data_dir}/yellow_tripdata_2019-10.parquet",
            "csv": f"{data_dir}/yellow_tripdata_2019-10.csv",
            "name": "Yellow Taxi Data"
        }
    ]
    
    # Check if area information file exists
    area_file = f"{data_dir}/area_information.csv"
    if not os.path.exists(area_file):
        print(f"⚠️  Warning: {area_file} not found!")
        print(f"   This file should contain NYC taxi zone information.")
        print(f"   Please ensure it exists before running experiments.")
        print()
    else:
        print(f"✅ Area information file found: {area_file}")
        print()
    
    success_count = 0
    total_operations = len(files_to_download) * 2  # Download + convert for each file
    
    for file_info in files_to_download:
        print(f"📁 Processing {file_info['name']}")
        print("-" * 40)
        
        # Download parquet file
        if download_with_progress(file_info["url"], file_info["parquet"]):
            success_count += 1
        else:
            print(f"❌ Failed to download {file_info['name']}")
            continue
            
        # Convert to CSV
        if convert_parquet_to_csv(file_info["parquet"], file_info["csv"]):
            success_count += 1
        else:
            print(f"❌ Failed to convert {file_info['name']} to CSV")
            
        print()
    
    # Summary
    print("=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if success_count == total_operations:
        print("🎉 All files downloaded and converted successfully!")
        
        # Show file sizes
        print("\n📁 Final file sizes:")
        for file_info in files_to_download:
            for file_type, filepath in [("Parquet", file_info["parquet"]), ("CSV", file_info["csv"])]:
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) // (1024*1024)
                    print(f"   {os.path.basename(filepath)}: {size_mb} MB")
        
        print(f"\n✅ Ready to run experiments!")
        print(f"   Next step: bash Experiments_test_refactored.sh")
        
    else:
        print(f"⚠️  Completed {success_count}/{total_operations} operations")
        print("   Some files may be missing. Check the errors above.")
        
        if success_count > 0:
            print("   You may still be able to run experiments with partial data.")
    
    print()

if __name__ == "__main__":
    main()
