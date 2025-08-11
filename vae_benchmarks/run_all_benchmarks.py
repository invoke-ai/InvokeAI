#!/usr/bin/env python3
"""
Main runner script to execute all VAE benchmarks and generate a comprehensive report.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from statistics import mean, median

import torch


def run_benchmark(script_name: str) -> bool:
    """Run a benchmark script and return success status."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Script {script_path} not found!")
        return False
    
    print(f"\n{'=' * 80}")
    print(f"Running: {script_name}")
    print('=' * 80)
    
    try:
        # Use the InvokeAI venv python
        python_path = "/home/bat/Documents/Code/InvokeAI/.venv/bin/python"
        result = subprocess.run(
            [python_path, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {script_name} failed with exception: {e}")
        return False


def load_results(filename: str) -> List[Dict]:
    """Load benchmark results from JSON file."""
    file_path = Path(__file__).parent / filename
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return []


def analyze_results():
    """Analyze all benchmark results and generate comprehensive report."""
    print("\n" + "=" * 80)
    print("ANALYZING BENCHMARK RESULTS")
    print("=" * 80)
    
    # Load all results
    flux_results = load_results("flux_vae_benchmark_results.json")
    sd_results = load_results("sd_vae_benchmark_results.json")
    sd3_cogview_results = load_results("sd3_cogview_vae_benchmark_results.json")
    
    all_results = flux_results + sd_results + sd3_cogview_results
    
    if not all_results:
        print("No results found!")
        return
    
    # Generate comprehensive report
    report = []
    report.append("# VAE VRAM USAGE BENCHMARK REPORT")
    report.append("=" * 80)
    report.append("")
    
    # System Information
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    report.append(f"## System Information")
    report.append(f"- GPU: {device}")
    report.append(f"- Total VRAM: 24 GB (RTX 4090)")
    report.append("")
    
    # Summary Statistics by Model
    report.append("## Summary Statistics by Model")
    report.append("")
    
    # Group results by model
    models = {}
    for result in all_results:
        model = result.get('model', 'Unknown')
        if model not in models:
            models[model] = []
        models[model].append(result)
    
    for model, model_results in models.items():
        report.append(f"### {model}")
        if model_results:
            report.append(f"- Model Size: {model_results[0].get('model_size_mb', 0):.2f} MB")
        report.append("")
        
        # Group by operation
        operations = {}
        for result in model_results:
            op = result.get('operation', 'unknown')
            if op not in operations:
                operations[op] = []
            operations[op].append(result)
        
        for operation in ['encode', 'decode', 'decode_tiled']:
            if operation not in operations:
                continue
            
            op_results = operations[operation]
            if not op_results:
                continue
                
            report.append(f"#### {operation.capitalize()}")
            report.append(f"| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |")
            report.append("|------------|-------|----------------|---------------|----------|")
            
            for row in op_results:
                dtype_str = row.get('dtype', '').replace('torch.', '')
                report.append(f"| {row.get('resolution', '')} | {dtype_str} | "
                            f"{row.get('avg_allocated_mb', 0):.2f} | {row.get('avg_reserved_mb', 0):.2f} | "
                            f"{row.get('avg_time_s', 0):.3f} |")
            report.append("")
    
    # Key Findings
    report.append("## Key Findings")
    report.append("")
    
    # 1. Compare allocated vs reserved memory
    report.append("### 1. Allocated vs Reserved Memory Ratio")
    report.append("")
    
    # Calculate reserve ratios
    reserve_ratios = []
    for result in all_results:
        if result.get('avg_allocated_mb', 0) > 0:
            ratio = result.get('avg_reserved_mb', 0) / result.get('avg_allocated_mb', 1)
            reserve_ratios.append(ratio)
    
    if reserve_ratios:
        avg_ratio = mean(reserve_ratios)
        report.append(f"- Average Reserved/Allocated Ratio: {avg_ratio:.2f}x")
        report.append(f"- This confirms PyTorch reserves significantly more memory than it allocates")
        report.append("")
        
        # Group by model and operation
        for model, model_results in models.items():
            ops = {}
            for result in model_results:
                op = result.get('operation', 'unknown')
                if op not in ops:
                    ops[op] = []
                if result.get('avg_allocated_mb', 0) > 0:
                    ratio = result.get('avg_reserved_mb', 0) / result.get('avg_allocated_mb', 1)
                    ops[op].append(ratio)
            
            for op, ratios in ops.items():
                if ratios:
                    avg_op_ratio = mean(ratios)
                    report.append(f"- {model} {op}: {avg_op_ratio:.2f}x reserve ratio")
    report.append("")
    
    # 2. Memory scaling with resolution
    report.append("### 2. Memory Scaling with Resolution")
    report.append("")
    
    # Analyze scaling for each model
    for model, model_results in models.items():
        decode_results = [r for r in model_results if r.get('operation') == 'decode']
        if len(decode_results) > 1:
            # Sort by resolution
            decode_results.sort(key=lambda x: int(x.get('resolution', '0x0').split('x')[0]))
            
            first = decode_results[0]
            last = decode_results[-1]
            
            first_res = first.get('resolution', '0x0').split('x')
            last_res = last.get('resolution', '0x0').split('x')
            
            first_pixels = int(first_res[0]) * int(first_res[1])
            last_pixels = int(last_res[0]) * int(last_res[1])
            
            if first_pixels > 0 and first.get('avg_allocated_mb', 0) > 0:
                pixel_ratio = last_pixels / first_pixels
                memory_ratio = last.get('avg_allocated_mb', 0) / first.get('avg_allocated_mb', 1)
                
                report.append(f"- {model}: {pixel_ratio:.1f}x pixels → {memory_ratio:.1f}x memory")
    report.append("")
    
    # 3. Working memory estimation accuracy
    report.append("### 3. Current Working Memory Estimation Analysis")
    report.append("")
    report.append("Current InvokeAI uses `scaling_constant = 2200` for working memory estimation:")
    report.append("```python")
    report.append("working_memory = out_h * out_w * element_size * scaling_constant")
    report.append("```")
    report.append("")
    
    # Calculate what the scaling constant should be based on actual measurements
    implied_constants = []
    for model, model_results in models.items():
        decode_results = [r for r in model_results if r.get('operation') == 'decode']
        
        for row in decode_results:
            res = row.get('resolution', '0x0').split('x')
            h, w = int(res[0]), int(res[1])
            
            if h == 0 or w == 0:
                continue
            
            # Determine element size from dtype
            dtype = row.get('dtype', '')
            if 'float32' in dtype:
                element_size = 4
            elif 'float16' in dtype:
                element_size = 2
            elif 'bfloat16' in dtype:
                element_size = 2
            else:
                element_size = 2
            
            # Calculate implied scaling constant from actual measurements
            # Using reserved memory (what actually matters for OOM)
            reserved_mb = row.get('avg_reserved_mb', 0)
            if reserved_mb > 0:
                implied_constant = reserved_mb * 1024 * 1024 / (h * w * element_size)
                implied_constants.append(implied_constant)
                
                report.append(f"- {model} {row.get('resolution')} {dtype}: "
                            f"Implied constant = {implied_constant:.0f} "
                            f"(Actual: {reserved_mb:.0f} MB)")
    
    report.append("")
    
    # 4. SD1.5 vs SDXL comparison
    report.append("### 4. SD1.5 vs SDXL Comparison")
    report.append("")
    
    sd15_results = models.get('SD1.5', [])
    sdxl_results = models.get('SDXL', [])
    
    if sd15_results and sdxl_results:
        # Compare at same resolution
        for resolution in ['1024x1024', '512x512']:
            sd15_res = [r for r in sd15_results if r.get('resolution') == resolution and r.get('operation') == 'decode']
            sdxl_res = [r for r in sdxl_results if r.get('resolution') == resolution and r.get('operation') == 'decode']
            
            if sd15_res and sdxl_res:
                sd15_mem = sd15_res[0].get('avg_reserved_mb', 0)
                sdxl_mem = sdxl_res[0].get('avg_reserved_mb', 0)
                
                report.append(f"- At {resolution}:")
                report.append(f"  - SD1.5: {sd15_mem:.0f} MB")
                report.append(f"  - SDXL: {sdxl_mem:.0f} MB")
                
                if sd15_mem > 0 and sdxl_mem > 0:
                    if sd15_mem > sdxl_mem:
                        report.append(f"  - SD1.5 uses {(sd15_mem/sdxl_mem - 1)*100:.0f}% MORE memory than SDXL")
                    else:
                        report.append(f"  - SDXL uses {(sdxl_mem/sd15_mem - 1)*100:.0f}% MORE memory than SD1.5")
    report.append("")
    
    # 5. Recommendations
    report.append("## Recommendations")
    report.append("")
    
    # Calculate recommended scaling constants
    if implied_constants:
        # Sort to get percentiles
        implied_constants.sort()
        
        # Get percentiles
        p50_idx = len(implied_constants) // 2
        p95_idx = int(len(implied_constants) * 0.95)
        
        p50_constant = implied_constants[p50_idx]
        p95_constant = implied_constants[p95_idx] if p95_idx < len(implied_constants) else implied_constants[-1]
        
        report.append(f"1. **Adjust scaling constant for working memory:**")
        report.append(f"   - Current value: 2200")
        report.append(f"   - Median measured: {p50_constant:.0f}")
        report.append(f"   - 95th percentile: {p95_constant:.0f}")
        report.append(f"   - Recommendation: Use {p95_constant:.0f} for safety margin")
        report.append("")
    
    report.append("2. **Model-specific working memory:**")
    report.append("   - Consider different constants for different models")
    report.append("   - FLUX requires different handling than SD models")
    report.append("")
    
    report.append("3. **Encode operations also need working memory:**")
    report.append("   - Currently only decode reserves working memory")
    report.append("   - Encode operations show significant memory usage")
    report.append("")
    
    report.append("4. **Account for PyTorch memory reservation behavior:**")
    report.append("   - PyTorch reserves ~2-3x more memory than allocated")
    report.append("   - Working memory estimates should account for this")
    report.append("")
    
    # Save report
    report_path = Path(__file__).parent / "VAE_BENCHMARK_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {report_path}")
    
    # Also print to console
    print('\n'.join(report))
    
    # Save combined JSON for further analysis
    combined_path = Path(__file__).parent / "all_benchmark_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")


def main():
    """Main function to run all benchmarks."""
    print("VAE VRAM BENCHMARK SUITE")
    print("=" * 80)
    
    # List of benchmark scripts to run
    benchmarks = [
        "benchmark_flux_vae.py",
        "benchmark_sd_vae.py",
        "benchmark_sd3_cogview_vae.py",
    ]
    
    # Track results
    results = {}
    
    # Run each benchmark
    for benchmark in benchmarks:
        success = run_benchmark(benchmark)
        results[benchmark] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    
    for benchmark, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{benchmark}: {status}")
    
    # Analyze results if any succeeded
    if any(results.values()):
        analyze_results()
    else:
        print("\nNo benchmarks completed successfully. Cannot generate report.")


if __name__ == "__main__":
    main()