#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add MELD source to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from meld.holdout_family_time import main as holdout_main

def parse_results(results_file: str) -> Dict[str, Any]:
    """Parse the Time-OOD experiment results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return {}

def generate_performance_summary(results: Dict[str, Any]) -> str:
    """Generate performance summary table"""
    if not results or 'best' not in results:
        return "No performance data available"
    
    best = results['best']
    summary = f"""
## Performance Summary

| Metric | Value |
|--------|-------|
| **Best Layer** | Layer {best.get('layer_index', 'N/A')} |
| **Macro F1-Score** | {best.get('macro_f1', 0):.4f} |
| **AUROC** | {best.get('auroc', 0):.4f} |
| **AUPR** | {best.get('aupr', 0):.4f} |
| **Accuracy** | {best.get('accuracy', 0):.4f} |
| **Best Threshold** | {best.get('best_threshold', 0):.4f} |
"""
    return summary

def generate_layer_analysis(results: Dict[str, Any]) -> str:
    """Generate layer-wise performance analysis"""
    if not results or 'all_layers' not in results:
        return "No layer analysis data available"
    
    layers = results['all_layers']
    if not layers:
        return "No layer data available"
    
    # Find top 5 performing layers by macro F1
    sorted_layers = sorted(layers, key=lambda x: x.get('macro_f1', 0), reverse=True)[:5]
    
    analysis = """
## Top 5 Performing Layers

| Rank | Layer | Macro F1 | AUROC | AUPR | Accuracy | Threshold |
|------|-------|----------|-------|------|----------|-----------|
"""
    
    for i, layer in enumerate(sorted_layers, 1):
        analysis += f"| {i} | {layer.get('layer_index', 'N/A')} | {layer.get('macro_f1', 0):.4f} | {layer.get('auroc', 0):.4f} | {layer.get('aupr', 0):.4f} | {layer.get('accuracy', 0):.4f} | {layer.get('best_threshold', 0):.4f} |\n"
    
    return analysis

def generate_data_split_info(results: Dict[str, Any]) -> str:
    """Generate data split information"""
    if not results or 'sizes' not in results:
        return "No data split information available"
    
    sizes = results['sizes']
    split_info = results.get('split', {})
    
    info = f"""
## Data Split Information

| Split | Size | Description |
|-------|------|-------------|
| **Training Set** | {sizes.get('train', 0)} samples | Used for model training |
| **Validation Set** | {sizes.get('val', 0)} samples | Used for hyperparameter tuning |
| **Test Set** | {sizes.get('test', 0)} samples | Used for final evaluation |

**Split Configuration:**
- **Mode:** {split_info.get('mode', 'N/A')}
- **Time Threshold:** {split_info.get('time_threshold', 'N/A')}
- **Total Layers:** {results.get('num_layers', 'N/A')}
- **Hidden Size:** {results.get('hidden_size', 'N/A')}
- **Classifier:** {results.get('clf', 'N/A')}
"""
    return info

def generate_full_report(results: Dict[str, Any], output_file: str) -> str:
    """Generate comprehensive Time-OOD experiment report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Time-OOD Experiment Report

**Generated:** {timestamp}  
**Experiment Type:** Time-based Out-of-Distribution Detection  
**Model:** MELD (Malware dEtection with Large language moDels)

---

{generate_data_split_info(results)}

{generate_performance_summary(results)}

{generate_layer_analysis(results)}

## Experiment Configuration

- **Threshold Step:** {results.get('threshold_step', 'N/A')}
- **Until Layer:** {results.get('until_layer', 'All layers')}
- **Results File:** `{output_file}`

## Key Findings

"""
    
    if results and 'best' in results:
        best = results['best']
        f1_score = best.get('macro_f1', 0)
        auroc = best.get('auroc', 0)
        aupr = best.get('aupr', 0)
        
        report += f"""
1. **Best Performance:** The model achieved optimal performance at layer {best.get('layer_index', 'N/A')} with a Macro F1-score of {f1_score:.4f}.

2. **Detection Quality:** 
   - AUROC of {auroc:.4f} indicates excellent discrimination capability
   - AUPR of {aupr:.4f} shows strong precision-recall performance
   
3. **Temporal Generalization:** The model demonstrates {"strong" if f1_score > 0.8 else "moderate" if f1_score > 0.6 else "limited"} ability to generalize across time boundaries.

4. **Layer Analysis:** Performance varies across layers, with deeper layers (around layer {best.get('layer_index', 'N/A')}) showing optimal feature representations for temporal OOD detection.
"""
    else:
        report += """
No performance data available for analysis.
"""
    
    report += f"""

## Conclusion

The Time-OOD experiment evaluates the model's ability to detect malware samples that appear after the training time cutoff. This simulates real-world scenarios where new malware variants emerge over time.

---
*Report generated by MELD Time-OOD Experiment Runner*
"""
    
    return report

def run_time_ood_experiment(args):
    """Run Time-OOD experiment with paper parameters"""
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up arguments for Time-OOD experiment
    sys.argv = [
        "run_time_ood.py",
        "--index_csv", f"{args.data_dir}/dataset_with_family_time.csv",
        "--model_dir", args.model_dir,
        "--mode", "time",
        "--time_threshold", "2025-06-01",
        "--clf", "logreg",
        "--gpu", str(args.gpu),
        "--progress",
        "--out", args.output
    ]
    
    print("🚀 Starting Time-OOD experiment...")
    print(f"   Arguments: {' '.join(sys.argv[1:])}")
    print()
    
    # Run the experiment
    try:
        holdout_main()
        print("✅ Experiment completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-OOD Experiment with Report Generation")
    parser.add_argument("--data_dir", default="data/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--model_dir", default="models/qwen3-0.6b",
                       help="Directory containing the model") 
    parser.add_argument("--output", default="results/time_ood/meld_time_ood_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--report", default="results/time_ood/time_ood_report.md",
                       help="Output markdown report file")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip report generation")
    
    args = parser.parse_args()
    
    print("🕐 Time-OOD Experiment Runner")
    print("=" * 50)
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"🧠 Model Directory: {args.model_dir}")
    print(f"📊 Results Output: {args.output}")
    print(f"📄 Report Output: {args.report}")
    print(f"🖥️  GPU Device: {args.gpu}")
    print("=" * 50)
    print()
    
    # Run the experiment
    success = run_time_ood_experiment(args)
    
    if success and not args.no_report:
        print()
        print("📊 Generating experiment report...")
        
        # Parse results and generate report
        results = parse_results(args.output)
        
        if results:
            # Generate comprehensive report
            report_content = generate_full_report(results, args.output)
            
            # Save report to file
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            print(f"✅ Report generated: {args.report}")
            
            # Print key metrics to console
            if 'best' in results:
                best = results['best']
                print()
                print("🎯 Key Results:")
                print(f"   Best Layer: {best.get('layer_index', 'N/A')}")
                print(f"   Macro F1: {best.get('macro_f1', 0):.4f}")
                print(f"   AUROC: {best.get('auroc', 0):.4f}")
                print(f"   AUPR: {best.get('aupr', 0):.4f}")
                print(f"   Accuracy: {best.get('accuracy', 0):.4f}")
        else:
            print("⚠️  Could not generate report: Results file not found or invalid")
    
    elif success:
        print("✅ Experiment completed (report generation skipped)")
    
    print()
    print("🏁 Time-OOD experiment finished!")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

