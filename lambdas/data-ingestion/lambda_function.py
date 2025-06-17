#!/usr/bin/env python3
"""
Data Ingestion Lambda Function
Downloads NYC Taxi & FHV data directly to S3 without local storage.
"""

import json
import boto3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from urllib.parse import urlparse
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class NYCDataIngester:
    """
    Downloads NYC Taxi and FHV data directly to S3.
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.environ.get('S3_BUCKET', 'magisterka')
        
        # NYC TLC direct data URLs (PARQUET format)
        self.data_sources = {
            'yellow': {
                'base_url': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet',
                'description': 'Yellow Taxi Trip Data'
            },
            'green': {
                'base_url': 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet', 
                'description': 'Green Taxi Trip Data'
            },
            'fhv': {
                'base_url': 'https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_{year}-{month:02d}.parquet',
                'description': 'For-Hire Vehicle Trip Data'
            }
        }
    
    def generate_s3_key(self, vehicle_type: str, year: int, month: int) -> str:
        """Generate S3 key for the dataset."""
        return f"datasets/{vehicle_type}/year={year}/month={month:02d}/{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
    
    def download_to_s3(self, vehicle_type: str, year: int, month: int, 
                      limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Download NYC taxi data directly to S3.
        
        Args:
            vehicle_type: Type of vehicle ('yellow', 'green', 'fhv')
            year: Year of data
            month: Month of data
            limit: Optional limit (NOTE: ignored for direct PARQUET downloads)
            
        Returns:
            Dictionary with download results
        """
        if vehicle_type not in self.data_sources:
            raise ValueError(f"Unsupported vehicle type: {vehicle_type}")
        
        # Build direct file URL
        file_url = self.data_sources[vehicle_type]['base_url'].format(year=year, month=month)
        
        logger.info(f"Downloading {vehicle_type} data for {year}-{month:02d} from {file_url}")
        
        try:
            # Stream download directly to S3
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            # Generate S3 key
            s3_key = self.generate_s3_key(vehicle_type, year, month)
            
            # Upload to S3 using streaming
            self.s3_client.upload_fileobj(
                response.raw,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'application/parquet'}
            )
            
            # Get object size
            obj_info = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            result = {
                'status': 'success',
                'vehicle_type': vehicle_type,
                'year': year,
                'month': month,
                's3_key': s3_key,
                's3_url': f"s3://{self.bucket_name}/{s3_key}",
                'size_bytes': obj_info['ContentLength'],
                'download_time': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Successfully uploaded {vehicle_type} data to {s3_key}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to download {vehicle_type} data: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'vehicle_type': vehicle_type,
                'year': year,
                'month': month
            }
        except Exception as e:
            logger.error(f"❌ Failed to upload to S3: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'vehicle_type': vehicle_type,
                'year': year,
                'month': month
            }
    

    
    def bulk_download(self, vehicle_types: List[str], year: int, 
                     start_month: int = 1, end_month: int = 12,
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Download multiple datasets in bulk.
        
        Args:
            vehicle_types: List of vehicle types to download
            year: Year of data
            start_month: Starting month
            end_month: Ending month
            limit: Optional limit on records per dataset
            
        Returns:
            List of download results
        """
        results = []
        
        for vehicle_type in vehicle_types:
            for month in range(start_month, end_month + 1):
                result = self.download_to_s3(vehicle_type, year, month, limit)
                results.append(result)
        
        return results

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "action": "download_single" | "download_bulk",
        "vehicle_type": "yellow|green|fhv",
        "year": 2020,
        "month": 3,
        "limit": 10000,  // optional
        
        // For bulk downloads:
        "vehicle_types": ["yellow", "green", "fhv"],
        "start_month": 1,
        "end_month": 12
    }
    """
    
    try:
        action = event.get('action', 'download_single')
        ingester = NYCDataIngester()
        
        if action == 'download_single':
            vehicle_type = event['vehicle_type']
            year = event['year']
            month = event['month']
            limit = event.get('limit')
            
            result = ingester.download_to_s3(vehicle_type, year, month, limit)
            
            return {
                'statusCode': 200 if result['status'] == 'success' else 500,
                'body': json.dumps(result)
            }
            
        elif action == 'download_bulk':
            vehicle_types = event['vehicle_types']
            year = event['year']
            start_month = event.get('start_month', 1)
            end_month = event.get('end_month', 12)
            limit = event.get('limit')
            
            results = ingester.bulk_download(vehicle_types, year, start_month, end_month, limit)
            
            success_count = len([r for r in results if r['status'] == 'success'])
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'action': 'download_bulk',
                    'total_downloads': len(results),
                    'successful_downloads': success_count,
                    'failed_downloads': len(results) - success_count,
                    'results': results
                })
            }
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f"Unknown action: {action}"
                })
            }
            
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

if __name__ == "__main__":
    # Test the function locally
    test_event = {
        "action": "download_single",
        "vehicle_type": "green",
        "year": 2019,
        "month": 3,
        "limit": 1000
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2)) 