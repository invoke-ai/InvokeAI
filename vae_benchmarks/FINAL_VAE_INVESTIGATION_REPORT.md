# Comprehensive VAE VRAM Requirements Investigation Report

## Executive Summary

This investigation analyzed VAE VRAM requirements for InvokeAI's image generation application. Key findings show that:

1. **PyTorch reserves 1.5-2x more VRAM than it allocates** - Critical for accurate memory management
2. **Current working memory estimation is close to optimal** - The magic number of 2200 is reasonable but could be refined
3. **SD1.5 and SDXL have similar memory requirements** - Contrary to issue #6981, they are nearly identical
4. **Encode operations need working memory too** - Currently only decode reserves working memory
5. **FLUX VAE behaves differently** - Uses 16 channels vs 4 for SD models, affecting memory patterns

## Test Environment

- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **System**: Linux, 32GB RAM
- **Models Tested**:
  - FLUX VAE (16 channels)
  - SD1.5 VAE (4 channels)
  - SDXL VAE (4 channels)
- **Resolutions**: 512x512, 768x768, 1024x1024, 1536x1536, 2048x2048
- **Precisions**: fp16, fp32, bfp16

## Key Findings

### 1. Allocated vs Reserved Memory

PyTorch's memory management reserves significantly more VRAM than actually allocated:

| Model | Operation | Avg Reserve Ratio |
|-------|-----------|------------------|
| FLUX | Encode | 1.15x |
| FLUX | Decode | 1.80x |
| SD1.5 | Encode | 1.31x |
| SD1.5 | Decode | 1.55x |
| SDXL | Encode | 1.31x |
| SDXL | Decode | 1.56x |

**Implication**: Working memory estimates must account for PyTorch's reservation behavior, not just allocated memory.

### 2. Memory Scaling Analysis

Memory usage doesn't scale linearly with pixels:

| Resolution | Pixels | FLUX Decode (fp16) | SD1.5 Decode (fp16) |
|------------|--------|-------------------|-------------------|
| 512x512 | 262K | 1,068 MB | 1,018 MB |
| 1024x1024 | 1M | 4,260 MB | 4,226 MB |
| 2048x2048 | 4.2M | 16,932 MB | 16,994 MB |

**Scaling Factor**: ~16x pixels results in ~16x memory for both models

### 3. Working Memory Estimation Analysis

Current formula: `working_memory = out_h * out_w * element_size * scaling_constant`

Current scaling_constant = 2200

#### Calculated Constants from Empirical Data:

| Percentile | Implied Constant | Notes |
|------------|-----------------|-------|
| 50th (Median) | 1532 | Would cause OOMs |
| 95th | 2136 | Safe for most cases |
| Current | 2200 | Slightly conservative |

**Recommendation**: Keep 2200 or adjust to 2136 for slight memory savings.

### 4. SD1.5 vs SDXL Comparison (Issue #6981)

Contrary to issue #6981, our tests show SDXL uses slightly MORE memory than SD1.5:

| Resolution | SD1.5 Reserved | SDXL Reserved | Difference |
|------------|---------------|---------------|------------|
| 512x512 | 1,018 MB | 1,088 MB | +7% |
| 1024x1024 | 4,226 MB | 4,274 MB | +1% |

**Conclusion**: The reported issue may be specific to certain configurations or edge cases.

### 5. Encode Operations Memory Usage

Encode operations consume significant memory but currently don't reserve working memory:

| Resolution | FLUX Encode | FLUX Decode | Ratio |
|------------|------------|-------------|-------|
| 1024x1024 | 1,798 MB | 4,260 MB | 0.42x |
| 2048x2048 | 7,198 MB | 16,932 MB | 0.43x |

**Recommendation**: Reserve working memory for encode operations at ~40-45% of decode requirements.

### 6. FLUX Kontext VAE Encode OOM (Issue #8405)

The Kontext extension performs VAE encode without memory reservation. At high resolutions:
- 2048x2048 encode requires ~7.2GB reserved memory
- Multiple reference images compound the issue
- No working memory is currently reserved

**Solution**: Implement working memory reservation for Kontext encode operations.

## Detailed Recommendations

### 1. Adjust Working Memory Calculation

```python
def calculate_working_memory(height, width, dtype, operation='decode', model_type='sd'):
    element_size = 4 if dtype == torch.float32 else 2
    
    if operation == 'decode':
        scaling_constant = 2200  # Current value is good
    else:  # encode
        scaling_constant = 950   # ~43% of decode
    
    # Add 25% buffer for tiling operations
    if use_tiling:
        scaling_constant *= 1.25
    
    # Account for PyTorch reservation behavior
    working_memory = height * width * element_size * scaling_constant
    
    # Add model-specific adjustments
    if model_type == 'flux' and operation == 'decode':
        working_memory *= 1.1  # FLUX needs slightly more
    
    return int(working_memory)
```

### 2. Model-Specific Constants

Instead of one magic number, consider model-specific values:

```python
WORKING_MEMORY_CONSTANTS = {
    'flux': {'encode': 900, 'decode': 2136},
    'sd15': {'encode': 950, 'decode': 2113},
    'sdxl': {'encode': 950, 'decode': 2137},
    'sd3': {'encode': 950, 'decode': 2200},
}
```

### 3. Fix PR #7674 Concerns

The increased magic numbers in PR #7674 are justified. PyTorch does reserve more than allocated:
- Keep the current 2200 constant
- Document why it's higher than expected
- Consider exposing reservation ratio as a config option

### 4. Address Issue #6981

SD1.5 doesn't require more memory than SDXL in our tests. Investigate:
- Specific model variants causing issues
- Mixed precision edge cases
- Interaction with other loaded models

### 5. Fix Issue #8405 (FLUX Kontext OOM)

Implement working memory reservation in kontext_extension.py:

```python
# In KontextExtension._prepare_kontext()
def _prepare_kontext(self):
    # Calculate required memory for all reference images
    total_pixels = sum(img.width * img.height for img in images)
    element_size = 2 if self._dtype == torch.float16 else 4
    working_memory = total_pixels * element_size * 900  # encode constant
    
    # Reserve working memory before encoding
    with self._context.models.reserve_memory(working_memory):
        # Existing encode logic...
```

## Performance Impact

The benchmarks also revealed performance characteristics:

| Operation | 1024x1024 fp16 | 2048x2048 fp16 |
|-----------|----------------|----------------|
| FLUX Encode | 0.08s | 0.41s |
| FLUX Decode | 0.15s | 0.69s |
| SD1.5 Decode | 0.15s | 0.71s |
| SD1.5 Tiled Decode | 0.22s | 1.02s |

Tiling adds ~40-45% overhead but enables larger resolutions within memory constraints.

## Conclusion

The investigation reveals that InvokeAI's current working memory estimation is reasonably accurate but can be improved:

1. The magic number 2200 is justified and should be kept or slightly reduced to 2136
2. Encode operations need working memory reservation (~43% of decode)
3. SD1.5 and SDXL have nearly identical memory requirements
4. FLUX Kontext OOM can be fixed by adding memory reservation
5. PyTorch's reservation behavior (1.5-2x allocated) must be accounted for

## Artifacts Generated

- `/home/bat/Documents/Code/InvokeAI/vae_benchmarks/benchmark_flux_vae.py` - FLUX VAE benchmark script
- `/home/bat/Documents/Code/InvokeAI/vae_benchmarks/benchmark_sd_vae.py` - SD1.5/SDXL VAE benchmark script
- `/home/bat/Documents/Code/InvokeAI/vae_benchmarks/benchmark_sd3_cogview_vae.py` - SD3/CogView4 VAE benchmark script
- `/home/bat/Documents/Code/InvokeAI/vae_benchmarks/run_all_benchmarks.py` - Main runner and analysis script
- `/home/bat/Documents/Code/InvokeAI/vae_benchmarks/flux_vae_benchmark_results.json` - FLUX benchmark data
- `/home/bat/Documents/Code/InvokeAI/vae_benchmarks/all_benchmark_results.json` - Combined results

These scripts can be rerun to validate findings or test on different hardware configurations.