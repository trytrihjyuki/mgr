#!/usr/bin/env python3
"""
Enhanced NYC Taxi Data Ingestion Lambda Function
Supports multiple data sources with fallback and detailed logging.
"""

import json
import boto3
import requests
import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 and HTTP configuration
s3_client = boto3.client('s3')
BUCKET_NAME = 'magisterka'
TIMEOUT = 300  # 5 minutes timeout
CHUNK_SIZE = 8192  # 8KB chunks for streaming

# Data source configurations
DATA_SOURCES = {
    'cloudfront': {
        'name': 'NYC TLC CloudFront CDN',
        'base_url': 'https://d37ci6vzurychx.cloudfront.net/trip-data',
        'priority': 1,
        'timeout': 120
    },
    'open_data_api': {
        'name': 'NYC Open Data API', 
        'base_url': 'https://data.cityofnewyork.us/resource',
        'priority': 2,
        'timeout': 180,
        'endpoints': {
            'yellow_2023': 'ajxm-kzmj.json',  # 2023 Yellow Taxi endpoint
            'green_2023': 'peyi-gg4n.json',   # 2023 Green Taxi endpoint
            'fhv_2023': 'gkne-dk5s.json'      # 2023 FHV endpoint (if available)
        }
    }
}

def lambda_handler(event, context):
    """
    Enhanced data ingestion with multiple source support and detailed logging.
    """
    start_time = time.time()
    
    try:
        logger.info("🚀 STARTING NYC TAXI DATA INGESTION")
        logger.info(f"📥 Event: {json.dumps(event, default=str)}")
        
        # Parse event
        action = event.get('action', 'download_single')
        
        if action == 'download_single':
            result = download_single_dataset(event)
        elif action == 'download_bulk':
            result = download_bulk_datasets(event)
        else:
            raise ValueError(f"Unknown action: {action}")
        
        total_time = time.time() - start_time
        logger.info(f"✅ INGESTION COMPLETED in {total_time:.2f}s")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result, default=str)
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ INGESTION FAILED after {error_time:.2f}s: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'error',
                'error': str(e),
                'execution_time': error_time
            }, default=str)
        }

def download_single_dataset(event: Dict[str, Any]) -> Dict[str, Any]:
    """Download a single dataset with multi-source fallback."""
    vehicle_type = event.get('vehicle_type', 'green')
    year = event.get('year', 2019)
    month = event.get('month', 3)
    limit = event.get('limit')
    
    logger.info(f"🎯 TARGET: {vehicle_type.upper()} taxi data for {year}-{month:02d}")
    if limit:
        logger.info(f"📊 LIMIT: {limit:,} records")
    
    # Try multiple data sources
    sources_tried = []
    last_error = None
    
    for source_key, source_config in sorted(DATA_SOURCES.items(), key=lambda x: x[1]['priority']):
        logger.info(f"🔄 TRYING: {source_config['name']}")
        sources_tried.append(source_config['name'])
        
        try:
            result = download_from_source(
                source_key, source_config, vehicle_type, year, month, limit
            )
            if result['status'] == 'success':
                result['sources_tried'] = sources_tried
                return result
                
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️  FAILED: {source_config['name']} - {str(e)}")
            continue
    
    # All sources failed
    return {
        'status': 'error',
        'error': f"All data sources failed. Last error: {last_error}",
        'sources_tried': sources_tried,
        'vehicle_type': vehicle_type,
        'year': year,
        'month': month
    }

def download_bulk_datasets(event: Dict[str, Any]) -> Dict[str, Any]:
    """Download multiple datasets with progress tracking."""
    vehicle_types = event.get('vehicle_types', ['green', 'yellow', 'fhv'])
    year = event.get('year', 2019)
    start_month = event.get('start_month', 1)
    end_month = event.get('end_month', 12)
    limit = event.get('limit')
    
    logger.info(f"🎯 BULK DOWNLOAD: {', '.join([v.upper() for v in vehicle_types])}")
    logger.info(f"📅 PERIOD: {year}-{start_month:02d} to {year}-{end_month:02d}")
    if limit:
        logger.info(f"📊 LIMIT: {limit:,} records per dataset")
    
    results = []
    successful_downloads = 0
    failed_downloads = 0
    
    total_datasets = len(vehicle_types) * (end_month - start_month + 1)
    current_dataset = 0
    
    for vehicle_type in vehicle_types:
        for month in range(start_month, end_month + 1):
            current_dataset += 1
            logger.info(f"📦 DATASET {current_dataset}/{total_datasets}: {vehicle_type.upper()} {year}-{month:02d}")
            
            single_event = {
                'action': 'download_single',
                'vehicle_type': vehicle_type,
                'year': year,
                'month': month,
                'limit': limit
            }
            
            result = download_single_dataset(single_event)
            results.append(result)
            
            if result['status'] == 'success':
                successful_downloads += 1
                logger.info(f"✅ SUCCESS: {result.get('size_mb', 'Unknown')}MB uploaded")
            else:
                failed_downloads += 1
                logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    logger.info(f"📈 BULK SUMMARY: {successful_downloads} success, {failed_downloads} failed")
    
    return {
        'action': 'download_bulk',
        'total_downloads': total_datasets,
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'results': results
    }

def download_from_source(source_key: str, source_config: Dict[str, Any], 
                        vehicle_type: str, year: int, month: int, 
                        limit: Optional[int] = None) -> Dict[str, Any]:
    """Download from a specific data source."""
    
    if source_key == 'cloudfront':
        return download_from_cloudfront(source_config, vehicle_type, year, month, limit)
    elif source_key == 'open_data_api':
        return download_from_open_data_api(source_config, vehicle_type, year, month, limit)
    else:
        raise ValueError(f"Unknown source: {source_key}")

def download_from_cloudfront(source_config: Dict[str, Any], vehicle_type: str, 
                           year: int, month: int, limit: Optional[int] = None) -> Dict[str, Any]:
    """Download from NYC TLC CloudFront CDN."""
    
    # Build URL
    filename = f"{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
    url = f"{source_config['base_url']}/{filename}"
    
    logger.info(f"🌐 URL: {url}")
    
    # Make request with detailed logging
    logger.info(f"🔍 CHECKING: File availability...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    # First, check if file exists
    try:
        head_response = session.head(url, timeout=30)
        logger.info(f"📊 HEAD Response: {head_response.status_code}")
        
        if head_response.status_code == 403:
            logger.warning(f"🚫 403 Forbidden - CloudFront may be blocking requests")
            raise requests.exceptions.RequestException("403 Forbidden from CloudFront")
        elif head_response.status_code == 404:
            logger.warning(f"🚫 404 Not Found - File may not exist for {year}-{month:02d}")
            raise requests.exceptions.RequestException("404 Not Found")
        elif head_response.status_code != 200:
            logger.warning(f"🚫 HTTP {head_response.status_code} - Unexpected response")
            raise requests.exceptions.RequestException(f"HTTP {head_response.status_code}")
            
        file_size = int(head_response.headers.get('Content-Length', 0))
        logger.info(f"📏 FILE SIZE: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
    except Exception as e:
        logger.error(f"❌ HEAD request failed: {str(e)}")
        raise
    
    # Download file
    logger.info(f"⬇️  DOWNLOADING: Starting stream...")
    
    try:
        with session.get(url, stream=True, timeout=source_config['timeout']) as response:
            response.raise_for_status()
            
            # Build S3 key
            s3_key = f"datasets/{vehicle_type}/year={year}/month={month:02d}/{filename}"
            
            # Upload to S3 with progress tracking
            logger.info(f"☁️  S3 KEY: {s3_key}")
            logger.info(f"📤 UPLOADING: Starting multipart upload...")
            
            bytes_downloaded = 0
            start_time = time.time()
            
            # Stream directly to S3
            s3_client.upload_fileobj(
                response.raw,
                BUCKET_NAME, 
                s3_key,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )
            
            upload_time = time.time() - start_time
            size_mb = file_size / 1024 / 1024
            speed_mbps = size_mb / upload_time if upload_time > 0 else 0
            
            logger.info(f"✅ UPLOAD COMPLETE: {size_mb:.1f}MB in {upload_time:.1f}s ({speed_mbps:.1f} MB/s)")
            
            return {
                'status': 'success',
                'source': source_config['name'],
                'vehicle_type': vehicle_type,
                'year': year,
                'month': month,
                's3_key': s3_key,
                'size_bytes': file_size,
                'size_mb': round(size_mb, 2),
                'upload_time_seconds': round(upload_time, 2),
                'download_speed_mbps': round(speed_mbps, 2),
                'url': url
            }
            
    except Exception as e:
        logger.error(f"❌ DOWNLOAD failed: {str(e)}")
        raise

def download_from_open_data_api(source_config: Dict[str, Any], vehicle_type: str,
                               year: int, month: int, limit: Optional[int] = None) -> Dict[str, Any]:
    """Download from NYC Open Data API (fallback)."""
    
    # For now, return an informative message since API integration needs more setup
    logger.info(f"📊 NYC Open Data API integration not yet implemented")
    logger.info(f"🔄 This would query the API for {vehicle_type} data {year}-{month:02d}")
    
    # Would implement API querying here with proper Socrata API calls
    raise NotImplementedError("NYC Open Data API integration coming soon")

if __name__ == "__main__":
    # Test event
    test_event = {
        'action': 'download_single',
        'vehicle_type': 'green',
        'year': 2019,
        'month': 3,
        'limit': 1000
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, default=str)) 