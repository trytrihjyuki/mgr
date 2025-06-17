#!/usr/bin/env python3
"""
Local Results Manager
Loads experiment results and analysis from S3 and displays them as reports.
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
    Manages loading, analysis, and reporting of experiment results from S3.
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
        List available experiments from S3.
        
        Args:
            experiment_type: Type of experiments to list
            days_back: Number of days back to search
            
        Returns:
            List of experiment metadata
        """
        prefix = f"experiments/results/{experiment_type}/"
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            experiments = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    # Filter by date
                    if obj['LastModified'].replace(tzinfo=None) >= cutoff_date:
                        experiments.append({
                            'key': obj['Key'],
                            'filename': Path(obj['Key']).name,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'experiment_id': Path(obj['Key']).stem.replace('_results', ''),
                            's3_url': f"s3://{self.bucket_name}/{obj['Key']}"
                        })
            
            # Sort by date (newest first)
            experiments.sort(key=lambda x: x['last_modified'], reverse=True)
            
            logger.info(f"📊 Found {len(experiments)} experiments from last {days_back} days")
            return experiments
            
        except Exception as e:
            logger.error(f"❌ Failed to list experiments: {e}")
            return []
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load specific experiment results from S3.
        
        Args:
            experiment_id: ID of experiment to load
            
        Returns:
            Experiment results data
        """
        s3_key = f"experiments/results/rideshare/{experiment_id}_results.json"
        cache_file = self.local_cache_dir / f"{experiment_id}_results.json"
        
        try:
            # Check if cached version exists and is recent
            if cache_file.exists():
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=1):
                    logger.info(f"📁 Loading cached results for {experiment_id}")
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            
            # Download from S3
            logger.info(f"📥 Downloading results for {experiment_id} from S3")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            results = json.loads(response['Body'].read().decode('utf-8'))
            
            # Cache locally
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Loaded results for {experiment_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to load results for {experiment_id}: {e}")
            return None
    
    def load_multiple_experiments(self, experiment_ids: List[str]) -> List[Dict[str, Any]]:
        """Load multiple experiments for comparison."""
        results = []
        for exp_id in experiment_ids:
            result = self.load_experiment_results(exp_id)
            if result:
                results.append(result)
        return results
    
    def generate_experiment_summary(self, experiment_data: Dict[str, Any]) -> str:
        """Generate a text summary of an experiment."""
        exp_id = experiment_data.get('experiment_id', 'Unknown')
        status = experiment_data.get('status', 'Unknown')
        
        if status == 'failed':
            return f"""
🧪 Experiment Summary: {exp_id}
Status: ❌ FAILED
Error: {experiment_data.get('error', 'Unknown error')}
Timestamp: {experiment_data.get('timestamp', 'Unknown')}
"""
        
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
  • Original Data Size: {data_info.get('original_data_size', 0):,} records
  • Processed Data Size: {data_info.get('processed_data_size', 0):,} records

📈 Results:
  • Total Scenarios: {results.get('total_scenarios', 0)}
  • Total Requests: {results.get('total_requests', 0):,}
  • Successful Matches: {results.get('total_successful_matches', 0):,}
  • Average Match Rate: {results.get('average_match_rate', 0):.2%}
  • Average Acceptance Probability: {results.get('average_acceptance_probability', 0):.3f}

🔗 S3 Location: s3://{self.bucket_name}/experiments/results/rideshare/{exp_id}_results.json
"""
    
    def create_comparison_report(self, experiments: List[Dict[str, Any]]) -> str:
        """Create a comparison report for multiple experiments."""
        if not experiments:
            return "No experiments to compare."
        
        report = f"""
📊 Experiment Comparison Report
{'='*60}
Comparing {len(experiments)} experiments
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Create comparison table
        comparison_data = []
        for exp in experiments:
            if exp.get('status') == 'completed':
                params = exp.get('parameters', {})
                results = exp.get('results', {})
                comparison_data.append({
                    'Experiment ID': exp.get('experiment_id', 'Unknown')[:20] + '...',
                    'Vehicle Type': params.get('vehicle_type', 'Unknown'),
                    'Year/Month': f"{params.get('year', 'Unknown')}/{params.get('month', 'Unknown'):02d}",
                    'Acceptance Fn': params.get('acceptance_function', 'Unknown'),
                    'Match Rate': f"{results.get('average_match_rate', 0):.2%}",
                    'Acceptance Prob': f"{results.get('average_acceptance_probability', 0):.3f}",
                    'Total Requests': f"{results.get('total_requests', 0):,}",
                    'Successful Matches': f"{results.get('total_successful_matches', 0):,}"
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report += "📋 Comparison Table:\n"
            report += df.to_string(index=False) + "\n\n"
            
            # Statistical summary
            match_rates = [exp['results']['average_match_rate'] for exp in experiments 
                          if exp.get('status') == 'completed' and 'results' in exp]
            if match_rates:
                report += f"""
📈 Statistical Summary:
  • Average Match Rate: {sum(match_rates)/len(match_rates):.2%}
  • Best Match Rate: {max(match_rates):.2%}
  • Worst Match Rate: {min(match_rates):.2%}
  • Standard Deviation: {pd.Series(match_rates).std():.3f}
"""
        
        return report
    
    def create_visualizations(self, experiments: List[Dict[str, Any]], 
                            output_dir: Path = None) -> List[str]:
        """
        Create visualizations for experiment results.
        
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
        
        # Filter successful experiments
        successful_experiments = [exp for exp in experiments 
                                 if exp.get('status') == 'completed' and 'results' in exp]
        
        if not successful_experiments:
            logger.warning("No successful experiments to visualize")
            return created_plots
        
        # 1. Match Rate Comparison
        try:
            plt.figure(figsize=(12, 6))
            
            exp_names = [exp['experiment_id'][-15:] for exp in successful_experiments]
            match_rates = [exp['results']['average_match_rate'] for exp in successful_experiments]
            vehicle_types = [exp['parameters']['vehicle_type'] for exp in successful_experiments]
            
            colors = ['green' if vt == 'green' else 'yellow' if vt == 'yellow' else 'blue' 
                     for vt in vehicle_types]
            
            bars = plt.bar(range(len(exp_names)), match_rates, color=colors, alpha=0.7)
            plt.xlabel('Experiment')
            plt.ylabel('Match Rate')
            plt.title('Match Rate Comparison Across Experiments')
            plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, match_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_file = output_dir / "match_rate_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_file))
            
        except Exception as e:
            logger.error(f"Failed to create match rate plot: {e}")
        
        # 2. Acceptance Probability vs Match Rate Scatter
        try:
            plt.figure(figsize=(10, 8))
            
            acceptance_probs = [exp['results']['average_acceptance_probability'] 
                              for exp in successful_experiments]
            match_rates = [exp['results']['average_match_rate'] 
                          for exp in successful_experiments]
            vehicle_types = [exp['parameters']['vehicle_type'] 
                           for exp in successful_experiments]
            
            for vtype in set(vehicle_types):
                mask = [vt == vtype for vt in vehicle_types]
                vtype_acceptance = [acceptance_probs[i] for i in range(len(mask)) if mask[i]]
                vtype_match = [match_rates[i] for i in range(len(mask)) if mask[i]]
                
                plt.scatter(vtype_acceptance, vtype_match, label=f'{vtype} taxi', 
                           alpha=0.7, s=100)
            
            plt.xlabel('Average Acceptance Probability')
            plt.ylabel('Average Match Rate')
            plt.title('Acceptance Probability vs Match Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / "acceptance_vs_match_scatter.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_file))
            
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
        
        logger.info(f"📊 Created {len(created_plots)} visualizations in {output_dir}")
        return created_plots
    
    def generate_full_report(self, days_back: int = 7) -> str:
        """Generate a comprehensive report of recent experiments."""
        experiments = self.list_experiments(days_back=days_back)
        
        if not experiments:
            return f"No experiments found in the last {days_back} days."
        
        # Load all experiment data
        experiment_data = []
        for exp_meta in experiments:
            data = self.load_experiment_results(exp_meta['experiment_id'])
            if data:
                experiment_data.append(data)
        
        # Generate report
        report = f"""
🏆 Rideshare Experiment Analysis Report
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: Last {days_back} days
Total Experiments: {len(experiment_data)}

"""
        
        # Individual experiment summaries
        report += "📋 Individual Experiment Summaries:\n"
        report += "="*60 + "\n"
        
        for exp_data in experiment_data:
            report += self.generate_experiment_summary(exp_data)
            report += "\n" + "-"*60 + "\n"
        
        # Comparison analysis
        if len(experiment_data) > 1:
            report += "\n" + self.create_comparison_report(experiment_data)
        
        # Create visualizations
        plot_files = self.create_visualizations(experiment_data)
        if plot_files:
            report += f"\n📊 Visualizations created:\n"
            for plot_file in plot_files:
                report += f"  • {plot_file}\n"
        
        return report

def main():
    """Command-line interface for the results manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Results Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List experiments
    list_cmd = subparsers.add_parser('list', help='List available experiments')
    list_cmd.add_argument('--days', type=int, default=30, help='Days back to search')
    
    # Show specific experiment
    show_cmd = subparsers.add_parser('show', help='Show specific experiment')
    show_cmd.add_argument('experiment_id', help='Experiment ID to show')
    
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
            print(f"  • {exp['experiment_id']} - {exp['last_modified'].strftime('%Y-%m-%d %H:%M')}")
    
    elif args.command == 'show':
        result = manager.load_experiment_results(args.experiment_id)
        if result:
            print(manager.generate_experiment_summary(result))
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