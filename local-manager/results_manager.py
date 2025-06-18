#!/usr/bin/env python3
"""
Enhanced Local Results Manager
Loads experiment results and analysis from S3 with support for comparative multi-method experiments.
"""

import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import io
import sys
import os

# Add parent directory to path to import aws_config
sys.path.append(str(Path(__file__).parent.parent))
from aws_config import AWSConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentResultsManager:
    """
    Enhanced manager for loading, analysis, and reporting of experiment results from S3.
    Supports both legacy single-method and new comparative multi-method experiments.
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = AWSConfig.BUCKET_NAME
        self.local_cache_dir = Path("local-cache")
        self.local_cache_dir.mkdir(exist_ok=True)
        
        # Setup matplotlib for better plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def list_experiments(self, experiment_type: str = 'rideshare',
                        days_back: int = 30) -> List[Dict[str, Any]]:
        """
        List available experiments from S3, supporting partitioned structure.
        
        Args:
            experiment_type: Type of experiments to list
            days_back: Number of days back to search
            
        Returns:
            List of experiment metadata
        """
        # New partitioned structure: experiments/rideshare/type=*/eval=*/year=*/month=*/
        prefixes = []
        
        # Build prefixes for partitioned structure
        base_prefix = f"experiments/{experiment_type}/"
        for vehicle_type in ['green', 'yellow', 'fhv']:
            for eval_type in ['pl', 'sigmoid']:
                prefixes.append(f"{base_prefix}type={vehicle_type}/eval={eval_type}/")
        
        # Also check legacy structure
        prefixes.extend([
            f"experiments/results/{experiment_type}/",
            f"experiments/results/{experiment_type}/pl/",
            f"experiments/results/{experiment_type}/sigmoid/"
        ])
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            experiments = []
            
            for prefix in prefixes:
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=prefix
                    )
                    
                    for obj in response.get('Contents', []):
                        if obj['Key'].endswith('.json'):
                            # Filter by date
                            if obj['LastModified'].replace(tzinfo=None) >= cutoff_date:
                                
                                # Extract metadata from path
                                experiment_id = self._extract_experiment_id_from_path(obj['Key'])
                                vehicle_type = self._extract_vehicle_type_from_path(obj['Key'])
                                acceptance_function = self._extract_acceptance_function_from_path(obj['Key'])
                                
                                experiments.append({
                                    'key': obj['Key'],
                                    'filename': Path(obj['Key']).name,
                                    'size': obj['Size'],
                                    'last_modified': obj['LastModified'],
                                    'experiment_id': experiment_id,
                                    'vehicle_type': vehicle_type,
                                    'acceptance_function': acceptance_function,
                                    's3_url': f"s3://{self.bucket_name}/{obj['Key']}"
                                })
                except Exception as e:
                    logger.debug(f"No objects found in prefix {prefix}: {e}")
                    continue
            
            # Remove duplicates and sort by date (newest first)
            seen_ids = set()
            unique_experiments = []
            for exp in experiments:
                exp_key = f"{exp['experiment_id']}_{exp['acceptance_function']}"
                if exp_key not in seen_ids:
                    unique_experiments.append(exp)
                    seen_ids.add(exp_key)
            
            unique_experiments.sort(key=lambda x: x['last_modified'], reverse=True)
            
            logger.info(f"📊 Found {len(unique_experiments)} experiments from last {days_back} days")
            return unique_experiments
            
        except Exception as e:
            logger.error(f"❌ Failed to list experiments: {e}")
            return []
    
    def _extract_experiment_id_from_path(self, s3_key: str) -> str:
        """Extract experiment ID from S3 path"""
        filename = Path(s3_key).stem
        if filename.startswith('run_'):
            # New format: run_20250617_220136.json
            return filename
        else:
            # Legacy format: experiment_id_results.json
            return filename.replace('_results', '')
    
    def _extract_vehicle_type_from_path(self, s3_key: str) -> str:
        """Extract vehicle type from S3 path"""
        if '/type=' in s3_key:
            # New partitioned format
            parts = s3_key.split('/type=')
            if len(parts) > 1:
                return parts[1].split('/')[0]
        
        # Try to extract from experiment ID or path
        if 'green' in s3_key:
            return 'green'
        elif 'yellow' in s3_key:
            return 'yellow'
        elif 'fhv' in s3_key:
            return 'fhv'
        
        return 'unknown'
    
    def _extract_acceptance_function_from_path(self, s3_key: str) -> str:
        """Extract acceptance function from S3 path"""
        if '/eval=' in s3_key:
            # New partitioned format
            parts = s3_key.split('/eval=')
            if len(parts) > 1:
                eval_type = parts[1].split('/')[0]
                return eval_type.upper()
        
        # Legacy format
        if '/pl/' in s3_key:
            return 'PL'
        elif '/sigmoid/' in s3_key:
            return 'Sigmoid'
        
        return 'Unknown'
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load specific experiment results from S3, trying multiple path patterns.
        
        Args:
            experiment_id: ID of experiment to load
            
        Returns:
            Experiment results data
        """
        # Build possible S3 key patterns for both old and new structures
        possible_keys = []
        
        # New partitioned structure patterns
        for vehicle_type in ['green', 'yellow', 'fhv']:
            for eval_type in ['pl', 'sigmoid']:
                for year in range(2019, 2025):
                    for month in range(1, 13):
                        # Pattern: experiments/rideshare/type=green/eval=pl/year=2019/month=03/run_20250617_220136.json
                        possible_keys.append(f"experiments/rideshare/type={vehicle_type}/eval={eval_type}/year={year}/month={month:02d}/{experiment_id}.json")
        
        # Legacy structure patterns
        legacy_patterns = [
            f"experiments/results/rideshare/{experiment_id}_results.json",
            f"experiments/results/rideshare/pl/{experiment_id}_results.json",
            f"experiments/results/rideshare/sigmoid/{experiment_id}_results.json"
        ]
        possible_keys.extend(legacy_patterns)
        
        cache_file = self.local_cache_dir / f"{experiment_id}_results.json"
        
        try:
            # Check if cached version exists and is recent
            if cache_file.exists():
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=1):
                    logger.info(f"📁 Loading cached results for {experiment_id}")
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            
            # Try to find the file using S3 list with prefix matching
            # This is more efficient than trying every possible key
            found_key = self._find_experiment_file(experiment_id)
            
            if found_key:
                logger.info(f"📥 Downloading results from {found_key}")
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=found_key)
                results = json.loads(response['Body'].read().decode('utf-8'))
                
                # Cache locally
                with open(cache_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"✅ Loaded results for {experiment_id}")
                return results
            
            logger.error(f"❌ No results found for {experiment_id} in any location")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load results for {experiment_id}: {e}")
            return None
    
    def _find_experiment_file(self, experiment_id: str) -> Optional[str]:
        """Find experiment file using S3 list operations"""
        
        # Search in partitioned structure (new unified format)
        try:
            # Extract info from unified experiment ID format
            # Format: unified_green_manhattan_2019_10_20250618_204257 or 
            # Legacy: rideshare_green_2019_10_hikima_pl_20250618_204257
            
            # Try to extract vehicle type and timestamp from experiment ID
            vehicle_types = ['green', 'yellow', 'fhv']
            eval_types = ['pl', 'sigmoid']
            years = list(range(2015, 2025))
            
            # New unified format prefixes 
            prefixes_to_search = []
            
            # Add unified experiment prefixes (new format)
            for vehicle_type in vehicle_types:
                for eval_type in eval_types:
                    for year in years:
                        prefixes_to_search.append(f"experiments/rideshare/type={vehicle_type}/eval={eval_type}/year={year}/")
            
            # Add legacy prefixes
            prefixes_to_search.extend([
                "experiments/rideshare/type=green/",
                "experiments/rideshare/type=yellow/", 
                "experiments/rideshare/type=fhv/",
                "experiments/results/rideshare/pl/",
                "experiments/results/rideshare/sigmoid/",
                "experiments/results/rideshare/"
            ])
            
            for prefix in prefixes_to_search:
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=prefix
                    )
                    
                    for obj in response.get('Contents', []):
                        key = obj['Key']
                        filename = Path(key).stem
                        
                        # Check if this file matches our experiment ID
                        # New unified format: unified_green_manhattan_2019_10_20250618_204257
                        # File: unified_20250618_204257.json -> match on timestamp part
                        if experiment_id.startswith('unified_'):
                            # Extract timestamp from experiment ID
                            parts = experiment_id.split('_')
                            if len(parts) >= 6:  # unified_green_manhattan_2019_10_timestamp
                                timestamp = '_'.join(parts[-2:])  # Last two parts = date_time
                                if timestamp in filename:
                                    logger.info(f"Found unified experiment: {key}")
                                    return key
                        
                        # Legacy format and exact matches
                        if (experiment_id in filename or 
                            filename.replace('_results', '') == experiment_id or
                            filename == experiment_id):
                            logger.info(f"Found legacy experiment: {key}")
                            return key
                            
                except Exception as e:
                    logger.debug(f"Error searching prefix {prefix}: {e}")
                    continue
            
        except Exception as e:
            logger.debug(f"Error searching for experiment file: {e}")
        
        return None
    
    def analyze_comparative_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Analyze a comparative experiment with multiple methods."""
        exp_id = experiment_data.get('experiment_id', 'Unknown')
        exp_type = experiment_data.get('experiment_type', 'rideshare')
        
        if exp_type != 'rideshare_comparative':
            return "This is not a comparative experiment."
        
        params = experiment_data.get('parameters', {})
        method_results = experiment_data.get('method_results', {})
        comparative_stats = experiment_data.get('comparative_stats', {})
        execution_times = experiment_data.get('execution_times', {})
        
        report = f"""
🔬 Comparative Experiment Analysis: {exp_id}
{'='*80}
Status: ✅ COMPLETED
Methods Tested: {', '.join(method_results.keys())}
Acceptance Function: {params.get('acceptance_function', 'Unknown')}
Vehicle Type: {params.get('vehicle_type', 'Unknown')}
Data Period: {params.get('year', 'Unknown')}/{params.get('month', 'Unknown'):02d}

📊 Method Performance Summary:
"""
        
        # Method comparison table
        if method_results:
            comparison_data = []
            for method, results in method_results.items():
                summary = results.get('summary', {})
                comparison_data.append({
                    'Method': method.upper(),
                    'Algorithm': results.get('algorithm', 'Unknown'),
                    'Avg Objective Value': f"{summary.get('avg_objective_value', 0):,.2f}",
                    'Avg Match Rate': f"{summary.get('avg_match_rate', 0):.2%}",
                    'Avg Revenue': f"{summary.get('avg_revenue', 0):,.2f}",
                    'Execution Time': f"{execution_times.get(method, 0):.3f}s"
                })
            
            df = pd.DataFrame(comparison_data)
            report += df.to_string(index=False) + "\n\n"
        
        # Best performers
        best_performing = comparative_stats.get('best_performing', {})
        if best_performing:
            report += "🏆 Best Performing Methods:\n"
            for metric, winner in best_performing.items():
                report += f"  • {metric.replace('_', ' ').title()}: {winner['method'].upper()} ({winner['value']:.3f})\n"
            report += "\n"
        
        # Performance ranking
        ranking = comparative_stats.get('performance_ranking', {})
        if ranking:
            report += "📈 Performance Ranking (by objective value):\n"
            for rank, data in ranking.items():
                report += f"  {rank}. {data['method'].upper()}: {data['score']:.2f}\n"
            report += "\n"
        
        # Detailed method analysis
        report += "🔍 Detailed Method Analysis:\n"
        for method, results in method_results.items():
            scenarios = results.get('scenarios', [])
            if scenarios:
                total_requests = sum(s['total_requests'] for s in scenarios)
                total_matches = sum(s['successful_matches'] for s in scenarios)
                avg_supply_demand = sum(s['supply_demand_ratio'] for s in scenarios) / len(scenarios)
                
                report += f"""
  {method.upper()} Method:
    • Total Scenarios: {len(scenarios)}
    • Total Requests Processed: {total_requests:,}
    • Total Successful Matches: {total_matches:,}
    • Average Supply/Demand Ratio: {avg_supply_demand:.3f}
    • Standard Deviation (Match Rate): {pd.Series([s['match_rate'] for s in scenarios]).std():.3f}
"""
        
        return report
    
    def analyze_unified_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Analyze a unified rideshare experiment with multiple methods."""
        exp_id = experiment_data.get('experiment_id', 'Unknown')
        status = experiment_data.get('status', 'Unknown')
        
        if status != 'completed':
            return f"❌ Experiment {exp_id} - Status: {status}"
        
        # Extract parameters
        original_setup = experiment_data.get('original_setup', {})
        experiment_params = experiment_data.get('experiment_parameters', {})
        method_results = experiment_data.get('method_results', {})
        performance_ranking = experiment_data.get('performance_ranking', [])
        
        place = original_setup.get('place', 'Unknown')
        days = original_setup.get('days', [])
        time_interval = original_setup.get('time_interval', 0)
        time_unit = original_setup.get('time_unit', 's')
        simulation_range = original_setup.get('simulation_range', 0)
        
        vehicle_type = experiment_params.get('vehicle_type', 'Unknown')
        year = experiment_params.get('year', 'Unknown')
        months = experiment_params.get('months', [])
        methods = experiment_params.get('methods', [])
        acceptance_function = experiment_params.get('acceptance_function', 'Unknown')
        num_eval = experiment_params.get('num_eval', 100)
        
        report = f"""
🧪 Unified Rideshare Experiment Analysis: {exp_id}
{'='*80}
Status: ✅ COMPLETED
Execution Time: {experiment_data.get('execution_time_seconds', 0):.2f} seconds
Timestamp: {experiment_data.get('timestamp', 'Unknown')}

📋 Original Setup (experiment_PL.py format):
  • Place: {place}
  • Days: {', '.join(map(str, days))}
  • Time Interval: {time_interval}{time_unit}
  • Simulation Range: {simulation_range} scenarios

📊 Extended Parameters:
  • Vehicle Type: {vehicle_type.upper()}
  • Year: {year}
  • Months: {', '.join(map(str, months))}
  • Methods: {', '.join(methods)}
  • Acceptance Function: {acceptance_function}
  • Monte Carlo Evaluations: {num_eval}

🏆 Performance Ranking:
"""
        
        # Performance ranking
        if performance_ranking:
            for i, method_data in enumerate(performance_ranking, 1):
                method = method_data.get('method', 'Unknown')
                score = method_data.get('avg_objective_value', 0)
                report += f"  {i}. {method.upper()}: {score:.2f}\n"
        else:
            report += "  No ranking available\n"
        
        report += "\n📈 Method Performance Summary:\n"
        
        # Method comparison table
        if method_results:
            comparison_data = []
            for method, results in method_results.items():
                summary = results.get('overall_summary', {})
                exec_time = results.get('method_execution_time', 0)
                
                comparison_data.append({
                    'Method': method.upper(),
                    'Avg Objective': f"{summary.get('avg_objective_value', 0):,.2f}",
                    'Avg Revenue': f"{summary.get('avg_revenue', 0):,.2f}",
                    'Total Matches': f"{summary.get('total_matches', 0):,.0f}",
                    'Success Rate': f"{summary.get('success_rate', 0):.2%}",
                    'Exec Time': f"{exec_time:.3f}s"
                })
            
            df = pd.DataFrame(comparison_data)
            report += df.to_string(index=False) + "\n\n"
        
        # Detailed method analysis
        report += "🔍 Detailed Method Analysis:\n"
        for method, results in method_results.items():
            summary = results.get('overall_summary', {})
            monthly_data = results.get('monthly_aggregates', {})
            
            total_simulations = summary.get('total_simulations', 0)
            avg_computation_time = summary.get('avg_computation_time', 0)
            total_computation_time = summary.get('total_computation_time', 0)
            method_execution_time = results.get('method_execution_time', 0)
            
            report += f"""
  {method.upper()} Method:
    • Total Simulations: {total_simulations}
    • Average Objective Value: {summary.get('avg_objective_value', 0):,.2f}
    • Standard Deviation: {summary.get('std_objective_value', 0):.3f}
    • Range: {summary.get('min_objective_value', 0):.2f} - {summary.get('max_objective_value', 0):.2f}
    • Average Computation Time: {avg_computation_time:.4f}s per scenario
    • Total Computation Time: {total_computation_time:.3f}s
    • Method Execution Time: {method_execution_time:.3f}s
    • Months Analyzed: {len(monthly_data)}
"""
        
        # Monthly breakdown for multi-month experiments
        monthly_summaries = experiment_data.get('monthly_summaries', {})
        if monthly_summaries:
            report += "\n📅 Monthly Performance Summary:\n"
            for month_key, month_data in monthly_summaries.items():
                report += f"\n  {month_key}:\n"
                for method, month_summary in month_data.items():
                    avg_obj = month_summary.get('avg_objective_value', 0)
                    total_sims = month_summary.get('total_simulations', 0)
                    report += f"    • {method.upper()}: {avg_obj:.2f} (from {total_sims} simulations)\n"
        
        return report
    
    def create_comparative_visualizations(self, experiment_data: Dict[str, Any], 
                                        output_dir: Path = None) -> List[str]:
        """Create visualizations specifically for comparative experiments."""
        if output_dir is None:
            output_dir = self.local_cache_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        
        created_plots = []
        method_results = experiment_data.get('method_results', {})
        
        if not method_results:
            logger.warning("No method results to visualize")
            return created_plots
        
        exp_id = experiment_data.get('experiment_id', 'unknown')
        
        # 1. Method Performance Comparison
        try:
            plt.figure(figsize=(14, 8))
            
            methods = list(method_results.keys())
            metrics = ['avg_objective_value', 'avg_match_rate', 'avg_revenue']
            
            x = range(len(methods))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [method_results[method]['summary'][metric] for method in methods]
                offset = (i - 1) * width
                plt.bar([pos + offset for pos in x], values, width, 
                       label=metric.replace('_', ' ').title())
            
            plt.xlabel('Methods')
            plt.ylabel('Performance Metrics')
            plt.title('Method Performance Comparison')
            plt.xticks(x, [m.upper() for m in methods])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / f"{exp_id}_method_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_file))
            
        except Exception as e:
            logger.error(f"Failed to create method comparison plot: {e}")
        
        # 2. Match Rate Distribution by Method
        try:
            plt.figure(figsize=(12, 8))
            
            for method, results in method_results.items():
                scenarios = results.get('scenarios', [])
                if scenarios:
                    match_rates = [s['match_rate'] for s in scenarios]
                    plt.hist(match_rates, alpha=0.6, label=method.upper(), bins=20)
            
            plt.xlabel('Match Rate')
            plt.ylabel('Frequency')
            plt.title('Match Rate Distribution by Method')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / f"{exp_id}_match_rate_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_file))
            
        except Exception as e:
            logger.error(f"Failed to create distribution plot: {e}")
        
        # 3. Revenue vs Match Rate Scatter by Method
        try:
            plt.figure(figsize=(12, 8))
            
            for method, results in method_results.items():
                scenarios = results.get('scenarios', [])
                if scenarios:
                    match_rates = [s['match_rate'] for s in scenarios]
                    revenues = [s.get('total_revenue', 0) for s in scenarios]
                    plt.scatter(match_rates, revenues, label=method.upper(), alpha=0.7, s=50)
            
            plt.xlabel('Match Rate')
            plt.ylabel('Total Revenue')
            plt.title('Revenue vs Match Rate by Method')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / f"{exp_id}_revenue_vs_match.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_file))
            
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
        
        logger.info(f"📊 Created {len(created_plots)} comparative visualizations")
        return created_plots
    
    def generate_experiment_summary(self, experiment_data: Dict[str, Any]) -> str:
        """Generate a text summary of an experiment (enhanced for comparative experiments)."""
        exp_id = experiment_data.get('experiment_id', 'Unknown')
        exp_type = experiment_data.get('experiment_type', 'rideshare')
        status = experiment_data.get('status', 'Unknown')
        
        if status == 'failed':
            return f"""
🧪 Experiment Summary: {exp_id}
Status: ❌ FAILED
Error: {experiment_data.get('error', 'Unknown error')}
Timestamp: {experiment_data.get('timestamp', 'Unknown')}
"""
        
        # Handle unified rideshare experiments (new format)
        if exp_type == 'unified_rideshare':
            return self.analyze_unified_experiment(experiment_data)
        
        # Handle comparative experiments
        if exp_type == 'rideshare_comparative':
            return self.analyze_comparative_experiment(experiment_data)
        
        # Legacy single-method experiments
        params = experiment_data.get('parameters', {})
        results = experiment_data.get('results', {})
        data_info = experiment_data.get('data_info', {})
        
        return f"""
🧪 Experiment Summary: {exp_id}
{'='*60}
Status: ✅ COMPLETED
Execution Time: {experiment_data.get('execution_time_seconds', 0):.2f} seconds
Timestamp: {experiment_data.get('timestamp', 'Unknown')}

📋 Parameters:
  • Vehicle Type: {params.get('vehicle_type', 'Unknown')}
  • Year/Month: {params.get('year', 'Unknown')}/{params.get('month', 'Unknown'):02d}
  • Location: {params.get('place', 'Unknown')}
  • Simulation Range: {params.get('simulation_range', 'Unknown')}
  • Acceptance Function: {params.get('acceptance_function', 'Unknown')}

📊 Data Information:
  • Data Exists: {data_info.get('data_exists', False)}
  • Data Size: {data_info.get('data_size_bytes', 0):,} bytes

📈 Results:
  • Total Scenarios: {results.get('total_scenarios', 0)}
  • Total Requests: {results.get('total_requests', 0):,}
  • Successful Matches: {results.get('total_successful_matches', 0):,}
  • Average Match Rate: {results.get('average_match_rate', 0):.2%}
  • Average Acceptance Probability: {results.get('average_acceptance_probability', 0):.3f}

🔗 S3 Location: s3://{self.bucket_name}/{experiment_data.get('s3_key', 'unknown')}
"""
    
    def create_visualizations(self, experiments: List[Dict[str, Any]], 
                            output_dir: Path = None) -> List[str]:
        """
        Create visualizations for experiment results (enhanced for comparative experiments).
        
        Args:
            experiments: List of experiment data
            output_dir: Directory to save plots (default: local-cache/plots)
            
        Returns:
            List of created plot filenames
        """
        if output_dir is None:
            output_dir = self.local_cache_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        
        created_plots = []
        
        # Separate comparative and legacy experiments
        comparative_experiments = [exp for exp in experiments 
                                 if exp.get('experiment_type') == 'rideshare_comparative']
        legacy_experiments = [exp for exp in experiments 
                            if exp.get('experiment_type') == 'rideshare' and exp.get('status') == 'completed']
        
        # Create visualizations for comparative experiments
        for exp in comparative_experiments:
            plots = self.create_comparative_visualizations(exp, output_dir)
            created_plots.extend(plots)
        
        # Create legacy visualizations if we have legacy experiments
        if legacy_experiments:
            plots = self._create_legacy_visualizations(legacy_experiments, output_dir)
            created_plots.extend(plots)
        
        return created_plots
    
    def _create_legacy_visualizations(self, experiments: List[Dict[str, Any]], 
                                    output_dir: Path) -> List[str]:
        """Create visualizations for legacy single-method experiments."""
        created_plots = []
        
        # 1. Match Rate Comparison
        try:
            plt.figure(figsize=(12, 6))
            
            exp_names = [exp['experiment_id'][-15:] for exp in experiments]
            match_rates = [exp['results']['average_match_rate'] for exp in experiments]
            vehicle_types = [exp['parameters']['vehicle_type'] for exp in experiments]
            
            colors = ['green' if vt == 'green' else 'yellow' if vt == 'yellow' else 'blue' 
                     for vt in vehicle_types]
            
            bars = plt.bar(range(len(exp_names)), match_rates, color=colors, alpha=0.7)
            plt.xlabel('Experiment')
            plt.ylabel('Match Rate')
            plt.title('Match Rate Comparison Across Legacy Experiments')
            plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, match_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_file = output_dir / "legacy_match_rate_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_file))
            
        except Exception as e:
            logger.error(f"Failed to create legacy match rate plot: {e}")
        
        return created_plots

def main():
    """Enhanced command-line interface for the results manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Experiment Results Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List experiments
    list_cmd = subparsers.add_parser('list', help='List available experiments')
    list_cmd.add_argument('--days', type=int, default=30, help='Days back to search')
    
    # Show specific experiment
    show_cmd = subparsers.add_parser('show', help='Show specific experiment')
    show_cmd.add_argument('experiment_id', help='Experiment ID to show')
    
    # Analyze comparative experiment
    analyze_cmd = subparsers.add_parser('analyze', help='Analyze comparative experiment')
    analyze_cmd.add_argument('experiment_id', help='Comparative experiment ID to analyze')
    
    # Compare experiments
    compare_cmd = subparsers.add_parser('compare', help='Compare multiple experiments')
    compare_cmd.add_argument('experiment_ids', nargs='+', help='Experiment IDs to compare')
    
    # Generate full report
    report_cmd = subparsers.add_parser('report', help='Generate full report')
    report_cmd.add_argument('--days', type=int, default=7, help='Days back to include')
    report_cmd.add_argument('--output', help='Output file (default: print to console)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ExperimentResultsManager()
    
    if args.command == 'list':
        experiments = manager.list_experiments(days_back=args.days)
        print(f"\n📊 Found {len(experiments)} experiments:")
        for exp in experiments:
            exp_type = "COMPARATIVE" if "comparative" in exp['experiment_id'] else "SINGLE"
            print(f"  • {exp['experiment_id']} ({exp_type}) - {exp['last_modified'].strftime('%Y-%m-%d %H:%M')}")
    
    elif args.command == 'show':
        result = manager.load_experiment_results(args.experiment_id)
        if result:
            print(manager.generate_experiment_summary(result))
        else:
            print(f"❌ Could not load experiment: {args.experiment_id}")
    
    elif args.command == 'analyze':
        result = manager.load_experiment_results(args.experiment_id)
        if result:
            if result.get('experiment_type') == 'rideshare_comparative':
                analysis = manager.analyze_comparative_experiment(result)
                print(analysis)
                # Create visualizations
                plots = manager.create_comparative_visualizations(result)
                if plots:
                    print(f"\n📊 Visualizations created:")
                    for plot in plots:
                        print(f"  • {plot}")
            else:
                print("❌ This is not a comparative experiment. Use 'show' command instead.")
        else:
            print(f"❌ Could not load experiment: {args.experiment_id}")
    
    elif args.command == 'compare':
        experiments = manager.load_multiple_experiments(args.experiment_ids)
        if experiments:
            print(manager.create_comparison_report(experiments))
            manager.create_visualizations(experiments)
        else:
            print("❌ Could not load any experiments for comparison")
    
    elif args.command == 'report':
        report = manager.generate_full_report(days_back=args.days)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {args.output}")
        else:
            print(report)

if __name__ == "__main__":
    main() 