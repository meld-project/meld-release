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
    """Parse the Family-OOD experiment results from JSON file"""
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
- **Target Family:** {split_info.get('test_family', 'N/A')}
- **Total Layers:** {results.get('num_layers', 'N/A')}
- **Hidden Size:** {results.get('hidden_size', 'N/A')}
- **Classifier:** {results.get('clf', 'N/A')}
"""
    return info

def generate_family_ood_report(results: Dict[str, Any], target_family: str, output_file: str) -> str:
    """Generate comprehensive Family-OOD experiment report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Family-OOD Experiment Report

**Generated:** {timestamp}  
**Experiment Type:** Family-based Out-of-Distribution Detection  
**Target Family:** {target_family}  
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
   - AUROC of {auroc:.4f} indicates {"excellent" if auroc > 0.9 else "good" if auroc > 0.8 else "moderate"} discrimination capability
   - AUPR of {aupr:.4f} shows {"strong" if aupr > 0.9 else "good" if aupr > 0.8 else "moderate"} precision-recall performance
   
3. **Family Generalization:** The model demonstrates {"strong" if f1_score > 0.8 else "moderate" if f1_score > 0.6 else "limited"} ability to detect the {target_family} family as out-of-distribution.

4. **Layer Analysis:** Performance varies across layers, with {"deeper" if best.get('layer_index', 0) > 15 else "middle" if best.get('layer_index', 0) > 8 else "shallow"} layers (around layer {best.get('layer_index', 'N/A')}) showing optimal feature representations for {target_family} family OOD detection.
"""
    else:
        report += """
No performance data available for analysis.
"""
    
    report += f"""

## Conclusion

The Family-OOD experiment evaluates the model's ability to detect malware samples from the {target_family} family when they are held out during training. This simulates real-world scenarios where new malware families emerge that were not seen during model training.

---
*Report generated by MELD Family-OOD Experiment Runner*
"""
    
    return report

def run_batch_experiments(args, families: List[str]) -> Dict[str, Dict[str, Any]]:
    """Run Family-OOD experiments for all families"""
    
    print("🚀 Batch Family-OOD Experiments")
    print("=" * 60)
    print(f"🎯 Target Families: {', '.join(families)}")
    print(f"📊 Total Experiments: {len(families)}")
    print("=" * 60)
    print()
    
    results_summary = {}
    successful_experiments = 0
    failed_experiments = 0
    
    for i, family in enumerate(families, 1):
        print(f"🔄 [{i}/{len(families)}] Running experiment for: {family}")
        
        # Create a copy of args for this family
        family_args = argparse.Namespace(**vars(args))
        family_args.target_family = family
        
        # Run the experiment
        success, output_file = run_family_ood_experiment(family_args)
        
        if success:
            # Parse results
            results = parse_results(output_file)
            if results:
                results_summary[family] = {
                    'results': results,
                    'output_file': output_file,
                    'success': True
                }
                successful_experiments += 1
                
                # Generate individual report if not disabled
                if not args.no_report:
                    report_file = args.report if args.report else f"results/family_ood/family_ood_{family.lower()}_report.md"
                    report_content = generate_family_ood_report(results, family, output_file)
                    
                    report_path = Path(report_file)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(report_path, 'w') as f:
                        f.write(report_content)
                    
                    print(f"   📄 Report: {report_file}")
                
                # Print key metrics
                if 'best' in results:
                    best = results['best']
                    print(f"   🎯 Best Layer: {best.get('layer_index', 'N/A')}")
                    print(f"   📊 Macro F1: {best.get('macro_f1', 0):.4f}")
                    print(f"   📈 AUROC: {best.get('auroc', 0):.4f}")
            else:
                results_summary[family] = {'success': False, 'error': 'Failed to parse results'}
                failed_experiments += 1
        else:
            results_summary[family] = {'success': False, 'error': 'Experiment failed'}
            failed_experiments += 1
        
        print()
    
    print("📊 Batch Experiment Summary:")
    print(f"   ✅ Successful: {successful_experiments}")
    print(f"   ❌ Failed: {failed_experiments}")
    print(f"   📈 Success Rate: {successful_experiments/len(families)*100:.1f}%")
    print()
    
    return results_summary

def generate_batch_summary_report(results_summary: Dict[str, Dict[str, Any]], args) -> str:
    """Generate summary report for batch experiments"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    successful_families = [f for f, r in results_summary.items() if r.get('success', False)]
    failed_families = [f for f, r in results_summary.items() if not r.get('success', False)]
    
    report = f"""# Family-OOD Batch Experiment Summary Report

**Generated:** {timestamp}  
**Experiment Type:** Family-based Out-of-Distribution Detection (Batch)  
**Model:** MELD (Malware dEtection with Large language moDels)

---

## Experiment Overview

| Metric | Value |
|--------|-------|
| **Total Families** | {len(results_summary)} |
| **Successful Experiments** | {len(successful_families)} |
| **Failed Experiments** | {len(failed_families)} |
| **Success Rate** | {len(successful_families)/len(results_summary)*100:.1f}% |

## Sample Limits Configuration

"""
    
    if args.max_train_samples > 0 or args.max_val_samples > 0 or args.max_test_samples > 0:
        report += f"""
| Parameter | Value |
|-----------|-------|
| **Max Training Samples** | {args.max_train_samples if args.max_train_samples > 0 else 'No limit'} |
| **Max Validation Samples** | {args.max_val_samples if args.max_val_samples > 0 else 'No limit'} |
| **Max Test Samples** | {args.max_test_samples if args.max_test_samples > 0 else 'No limit'} |
"""
    else:
        report += "No sample limits applied - using all available samples.\n"
    
    # Performance comparison table
    if successful_families:
        report += """
## Performance Comparison

| Family | Best Layer | Macro F1 | AUROC | AUPR | Accuracy | Threshold |
|--------|------------|----------|-------|------|----------|-----------|
"""
        
        # Sort by F1 score for ranking
        family_performance = []
        for family in successful_families:
            results = results_summary[family]['results']
            if 'best' in results:
                best = results['best']
                family_performance.append((
                    family,
                    best.get('layer_index', 'N/A'),
                    best.get('macro_f1', 0),
                    best.get('auroc', 0),
                    best.get('aupr', 0),
                    best.get('accuracy', 0),
                    best.get('best_threshold', 0)
                ))
        
        # Sort by F1 score descending
        family_performance.sort(key=lambda x: x[2], reverse=True)
        
        for family, layer, f1, auroc, aupr, acc, thresh in family_performance:
            report += f"| {family} | {layer} | {f1:.4f} | {auroc:.4f} | {aupr:.4f} | {acc:.4f} | {thresh:.4f} |\n"
    
    # Failed experiments
    if failed_families:
        report += f"""
## Failed Experiments

The following families failed to complete:

"""
        for family in failed_families:
            error = results_summary[family].get('error', 'Unknown error')
            report += f"- **{family}**: {error}\n"
    
    # Key findings
    if successful_families:
        avg_f1 = sum(results_summary[f]['results']['best']['macro_f1'] for f in successful_families) / len(successful_families)
        avg_auroc = sum(results_summary[f]['results']['best']['auroc'] for f in successful_families) / len(successful_families)
        best_family = max(successful_families, key=lambda f: results_summary[f]['results']['best']['macro_f1'])
        worst_family = min(successful_families, key=lambda f: results_summary[f]['results']['best']['macro_f1'])
        
        report += f"""
## Key Findings

1. **Overall Performance**: Average Macro F1-score across all families is {avg_f1:.4f}, with AUROC of {avg_auroc:.4f}.

2. **Best Performing Family**: {best_family} achieved the highest F1-score of {results_summary[best_family]['results']['best']['macro_f1']:.4f}.

3. **Most Challenging Family**: {worst_family} was the most difficult to detect with F1-score of {results_summary[worst_family]['results']['best']['macro_f1']:.4f}.

4. **Detection Quality**: {"Excellent" if avg_auroc > 0.9 else "Good" if avg_auroc > 0.8 else "Moderate"} average discrimination capability across families.

## Individual Reports

"""
        for family in successful_families:
            report_file = f"family_ood_{family.lower()}_report.md"
            report += f"- **{family}**: [{report_file}]({report_file})\n"
    
    report += f"""

## Conclusion

The batch Family-OOD experiments evaluate the model's ability to detect different malware families as out-of-distribution. This comprehensive evaluation across {len(results_summary)} families provides insights into the model's generalization capabilities and family-specific detection challenges.

---
*Report generated by MELD Family-OOD Batch Experiment Runner*
"""
    
    return report

def run_family_ood_experiment(args):
    """Run Family-OOD experiment for specific family"""
    
    output_file = f"results/family_ood/meld_family_ood_{args.target_family.lower()}.json"
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up arguments for Family-OOD experiment
    sys_args = [
        "run_family_ood.py",
        "--index_csv", f"{args.data_dir}/dataset_with_family_time.csv",
        "--model_dir", args.model_dir,
        "--mode", "family", 
        "--test_family", args.target_family,
        "--clf", "logreg",
        "--gpu", str(args.gpu),
        "--progress",
        "--out", output_file
    ]
    
    # Add sample limit parameters if specified
    if args.max_train_samples > 0:
        sys_args.extend(["--train_limit", str(args.max_train_samples)])
    if args.max_val_samples > 0:
        sys_args.extend(["--val_limit", str(args.max_val_samples)])  
    if args.max_test_samples > 0:
        sys_args.extend(["--test_limit", str(args.max_test_samples)])
    
    sys.argv = sys_args
    
    print("🚀 Starting Family-OOD experiment...")
    print(f"   Arguments: {' '.join(sys_args[1:])}")
    print()
    
    # Run the experiment
    try:
        holdout_main()
        print("✅ Experiment completed successfully!")
        return True, output_file
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return False, output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Family-OOD Experiment with Sample Limits")
    parser.add_argument("--target_family", 
                       choices=["AgentTesla", "AsyncRAT", "Formbook", "LummaStealer",
                               "MassLogger", "RemcosRAT", "SnakeKeylogger", "Stealc"],
                       help="Target family for OOD detection")
    parser.add_argument("--data_dir", default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--model_dir", default="models/qwen3-0.6b",
                       help="Directory containing the model")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    
    # Sample limit parameters
    parser.add_argument("--max_train_samples", type=int, default=0,
                       help="Maximum training samples (0 = no limit)")
    parser.add_argument("--max_val_samples", type=int, default=0,
                       help="Maximum validation samples (0 = no limit)")
    parser.add_argument("--max_test_samples", type=int, default=0,
                       help="Maximum test samples (0 = no limit, maintains 1:1 balance)")
    
    # Convenience parameter for overall limit
    parser.add_argument("--max_samples", type=int, default=0,
                       help="Set all sample limits to this value (overrides individual limits)")
    
    # Report generation parameters
    parser.add_argument("--report", default="",
                       help="Output markdown report file (auto-generated if not specified)")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip report generation")
    
    # Batch experiment parameters
    parser.add_argument("--batch", action="store_true",
                       help="Run experiments for all families (ignores --target_family)")
    parser.add_argument("--batch_report", default="results/family_ood/family_ood_summary_report.md",
                       help="Output summary report for batch experiments")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch and not args.target_family:
        parser.error("Either --target_family or --batch must be specified")
    
    # Apply overall limit if specified
    if args.max_samples > 0:
        args.max_train_samples = args.max_samples
        args.max_val_samples = args.max_samples // 10  # 10% for validation
        args.max_test_samples = args.max_samples
    
    # Define all available families
    ALL_FAMILIES = ["AgentTesla", "AsyncRAT", "Formbook", "LummaStealer", 
                    "MassLogger", "RemcosRAT", "SnakeKeylogger", "Stealc"]
    
    if args.batch:
        # Batch mode: run all families
        print(f"👥 Family-OOD Batch Experiment Runner")
        print("=" * 50)
        print(f"📁 Data Directory: {args.data_dir}")
        print(f"🧠 Model Directory: {args.model_dir}")
        print(f"🖥️  GPU Device: {args.gpu}")
        
        # Display sample limits if set
        if args.max_train_samples > 0 or args.max_val_samples > 0 or args.max_test_samples > 0:
            print("📊 Sample Limits:")
            if args.max_train_samples > 0:
                print(f"   Training: {args.max_train_samples}")
            if args.max_val_samples > 0:
                print(f"   Validation: {args.max_val_samples}")
            if args.max_test_samples > 0:
                print(f"   Test: {args.max_test_samples}")
        else:
            print("📊 Sample Limits: None (using all available samples)")
        
        print("=" * 50)
        print()
        
        # Run batch experiments
        results_summary = run_batch_experiments(args, ALL_FAMILIES)
        
        # Generate batch summary report
        if not args.no_report:
            print("📊 Generating batch summary report...")
            summary_report = generate_batch_summary_report(results_summary, args)
            
            report_path = Path(args.batch_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(summary_report)
            
            print(f"✅ Batch summary report generated: {args.batch_report}")
        
        successful_count = sum(1 for r in results_summary.values() if r.get('success', False))
        print()
        print("🏁 Batch experiments completed!")
        print(f"   ✅ Successful: {successful_count}/{len(ALL_FAMILIES)}")
        
        sys.exit(0 if successful_count > 0 else 1)
        
    else:
        # Single family mode
        print(f"👥 Family-OOD Experiment Runner")
        print("=" * 50)
        print(f"🎯 Target Family: {args.target_family}")
        print(f"📁 Data Directory: {args.data_dir}")
        print(f"🧠 Model Directory: {args.model_dir}")
        print(f"🖥️  GPU Device: {args.gpu}")
        
        # Display sample limits if set
        if args.max_train_samples > 0 or args.max_val_samples > 0 or args.max_test_samples > 0:
            print("📊 Sample Limits:")
            if args.max_train_samples > 0:
                print(f"   Training: {args.max_train_samples}")
            if args.max_val_samples > 0:
                print(f"   Validation: {args.max_val_samples}")
            if args.max_test_samples > 0:
                print(f"   Test: {args.max_test_samples}")
        else:
            print("📊 Sample Limits: None (using all available samples)")
        
        print("=" * 50)
        print()
        
        # Run single experiment
        success, output_file = run_family_ood_experiment(args)
        
        if success and not args.no_report:
            print()
            print("📊 Generating experiment report...")
            
            # Parse results and generate report
            results = parse_results(output_file)
            
            if results:
                # Generate comprehensive report
                report_file = args.report if args.report else f"results/family_ood/family_ood_{args.target_family.lower()}_report.md"
                report_content = generate_family_ood_report(results, args.target_family, output_file)
                
                # Save report to file
                report_path = Path(report_file)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(report_path, 'w') as f:
                    f.write(report_content)
                
                print(f"✅ Report generated: {report_file}")
                
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
        print("🏁 Family-OOD experiment finished!")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)

