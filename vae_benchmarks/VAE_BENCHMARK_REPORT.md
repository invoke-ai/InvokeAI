# VAE VRAM USAGE BENCHMARK REPORT
================================================================================

## System Information
- GPU: NVIDIA GeForce RTX 4090
- Total VRAM: 24 GB (RTX 4090)

## Summary Statistics by Model

### FLUX
- Model Size: 159.87 MB

#### Encode
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 512x512 | float16 | 384.28 | 452.00 | 0.018 |
| 768x768 | float16 | 864.28 | 1014.00 | 0.044 |
| 1024x1024 | float16 | 1536.28 | 1798.00 | 0.079 |
| 1536x1536 | float16 | 3456.28 | 4050.00 | 0.201 |
| 2048x2048 | float16 | 6144.28 | 7198.00 | 0.407 |
| 512x512 | float32 | 794.00 | 850.00 | 0.032 |
| 768x768 | float32 | 1774.00 | 1892.00 | 0.080 |
| 1024x1024 | float32 | 3146.00 | 3350.00 | 0.146 |
| 1536x1536 | float32 | 7066.00 | 7520.00 | 0.405 |
| 2048x2048 | float32 | 12554.00 | 15410.00 | 0.992 |
| 512x512 | bfloat16 | 384.28 | 452.00 | 0.017 |
| 768x768 | bfloat16 | 864.28 | 1014.00 | 0.044 |
| 1024x1024 | bfloat16 | 1536.28 | 1798.00 | 0.080 |
| 1536x1536 | bfloat16 | 3456.28 | 4036.00 | 0.202 |
| 2048x2048 | bfloat16 | 6144.28 | 7172.00 | 0.408 |

#### Decode
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 512x512 | float16 | 546.12 | 1068.00 | 0.033 |
| 768x768 | float16 | 1226.28 | 2376.00 | 0.083 |
| 1024x1024 | float16 | 2178.50 | 4260.00 | 0.153 |
| 1536x1536 | float16 | 4900.00 | 9538.00 | 0.364 |
| 2048x2048 | float16 | 8708.00 | 16932.00 | 0.693 |
| 512x512 | float32 | 898.25 | 1422.00 | 0.062 |
| 768x768 | float32 | 2018.56 | 3126.00 | 0.151 |
| 1024x1024 | float32 | 3587.00 | 5520.00 | 0.272 |
| 1536x1536 | float32 | 8067.38 | 11806.00 | 0.683 |
| 2048x2048 | float32 | 14341.13 | 19904.00 | 1.377 |
| 512x512 | bfloat16 | 546.12 | 1068.00 | 0.033 |
| 768x768 | bfloat16 | 1226.28 | 2376.00 | 0.084 |
| 1024x1024 | bfloat16 | 2178.50 | 4258.00 | 0.154 |
| 1536x1536 | bfloat16 | 4900.00 | 9536.00 | 0.366 |
| 2048x2048 | bfloat16 | 8708.00 | 16928.00 | 0.697 |

### SD1.5
- Model Size: 159.56 MB

#### Encode
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 512x512 | float16 | 384.28 | 534.40 | 0.018 |
| 768x768 | float16 | 864.28 | 1194.40 | 0.045 |
| 1024x1024 | float16 | 1536.28 | 2118.00 | 0.082 |
| 1536x1536 | float16 | 384.88 | 535.60 | 0.221 |
| 2048x2048 | float16 | 385.84 | 544.00 | 0.440 |
| 512x512 | float32 | 640.56 | 783.60 | 0.033 |
| 768x768 | float32 | 1440.56 | 1743.60 | 0.082 |
| 1024x1024 | float32 | 2560.56 | 3107.60 | 0.151 |
| 1536x1536 | float32 | 641.75 | 786.00 | 0.428 |
| 2048x2048 | float32 | 643.69 | 790.00 | 0.834 |

#### Decode
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 512x512 | float16 | 610.06 | 1018.00 | 0.032 |
| 768x768 | float16 | 1370.13 | 2344.00 | 0.083 |
| 1024x1024 | float16 | 2434.22 | 4226.00 | 0.154 |
| 1536x1536 | float16 | 5474.51 | 9538.00 | 0.372 |
| 2048x2048 | float16 | 9730.90 | 16993.60 | 0.710 |
| 512x512 | float32 | 962.36 | 1532.00 | 0.062 |
| 768x768 | float32 | 2162.50 | 3222.00 | 0.153 |
| 1024x1024 | float32 | 3842.70 | 5686.00 | 0.279 |
| 1536x1536 | float32 | 8643.26 | 12158.40 | 0.696 |
| 2048x2048 | float32 | 15364.05 | 20536.00 | 1.406 |
| 512x512 | float32 | 962.36 | 1554.00 | 0.063 |
| 768x768 | float32 | 2162.50 | 3224.00 | 0.155 |
| 1024x1024 | float32 | 3842.70 | 5687.60 | 0.280 |
| 1536x1536 | float32 | 8643.26 | 12158.00 | 0.697 |
| 2048x2048 | float32 | 15364.05 | 20535.60 | 1.408 |

#### Decode_tiled
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 1024x1024 | float16 | 616.38 | 1030.00 | 0.217 |
| 1536x1536 | float16 | 625.51 | 1020.00 | 0.500 |
| 2048x2048 | float16 | 649.93 | 1031.60 | 1.018 |
| 1024x1024 | float32 | 973.01 | 1532.00 | 0.396 |
| 1536x1536 | float32 | 992.26 | 1532.80 | 0.908 |
| 2048x2048 | float32 | 1039.99 | 1544.00 | 1.800 |
| 1024x1024 | float32 | 973.14 | 1553.60 | 0.398 |
| 1536x1536 | float32 | 991.26 | 1554.00 | 0.910 |
| 2048x2048 | float32 | 1039.11 | 1565.60 | 1.801 |

### SDXL
- Model Size: 159.56 MB

#### Encode
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 512x512 | float16 | 384.28 | 534.40 | 0.017 |
| 768x768 | float16 | 864.28 | 1194.40 | 0.045 |
| 1024x1024 | float16 | 1536.28 | 2118.00 | 0.082 |
| 1536x1536 | float16 | 384.88 | 555.60 | 0.221 |
| 2048x2048 | float16 | 385.84 | 544.00 | 0.440 |
| 512x512 | float32 | 640.56 | 783.60 | 0.033 |
| 768x768 | float32 | 1440.56 | 1743.60 | 0.082 |
| 1024x1024 | float32 | 2560.56 | 3107.60 | 0.151 |
| 1536x1536 | float32 | 641.75 | 786.00 | 0.428 |
| 2048x2048 | float32 | 643.69 | 790.00 | 0.834 |

#### Decode
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 512x512 | float16 | 610.06 | 1088.00 | 0.034 |
| 768x768 | float16 | 1370.13 | 2402.00 | 0.085 |
| 1024x1024 | float16 | 2434.22 | 4274.00 | 0.156 |
| 1536x1536 | float16 | 5474.51 | 9574.00 | 0.374 |
| 2048x2048 | float16 | 9730.90 | 16993.60 | 0.710 |
| 512x512 | float32 | 962.36 | 1532.00 | 0.062 |
| 768x768 | float32 | 2162.50 | 3222.00 | 0.153 |
| 1024x1024 | float32 | 3842.70 | 5686.00 | 0.279 |
| 1536x1536 | float32 | 8643.26 | 12158.40 | 0.697 |
| 2048x2048 | float32 | 15364.05 | 20536.00 | 1.407 |
| 512x512 | float32 | 962.36 | 1554.00 | 0.063 |
| 768x768 | float32 | 2162.50 | 3224.00 | 0.155 |
| 1024x1024 | float32 | 3842.70 | 5687.60 | 0.280 |
| 1536x1536 | float32 | 8643.26 | 12158.00 | 0.697 |
| 2048x2048 | float32 | 15364.05 | 20535.60 | 1.407 |

#### Decode_tiled
| Resolution | Dtype | Allocated (MB) | Reserved (MB) | Time (s) |
|------------|-------|----------------|---------------|----------|
| 1024x1024 | float16 | 615.38 | 1100.00 | 0.217 |
| 1536x1536 | float16 | 624.51 | 1090.00 | 0.502 |
| 2048x2048 | float16 | 649.43 | 1101.60 | 1.018 |
| 1024x1024 | float32 | 973.01 | 1532.00 | 0.397 |
| 1536x1536 | float32 | 992.26 | 1532.80 | 0.909 |
| 2048x2048 | float32 | 1039.99 | 1544.00 | 1.801 |
| 1024x1024 | float32 | 973.14 | 1553.60 | 0.399 |
| 1536x1536 | float32 | 991.26 | 1554.00 | 0.909 |
| 2048x2048 | float32 | 1039.11 | 1565.60 | 1.801 |

## Key Findings

### 1. Allocated vs Reserved Memory Ratio

- Average Reserved/Allocated Ratio: 1.49x
- This confirms PyTorch reserves significantly more memory than it allocates

- FLUX encode: 1.15x reserve ratio
- FLUX decode: 1.80x reserve ratio
- SD1.5 encode: 1.31x reserve ratio
- SD1.5 decode: 1.55x reserve ratio
- SD1.5 decode_tiled: 1.57x reserve ratio
- SDXL encode: 1.31x reserve ratio
- SDXL decode: 1.56x reserve ratio
- SDXL decode_tiled: 1.61x reserve ratio

### 2. Memory Scaling with Resolution

- FLUX: 16.0x pixels → 15.9x memory
- SD1.5: 16.0x pixels → 25.2x memory
- SDXL: 16.0x pixels → 25.2x memory

### 3. Current Working Memory Estimation Analysis

Current InvokeAI uses `scaling_constant = 2200` for working memory estimation:
```python
working_memory = out_h * out_w * element_size * scaling_constant
```

- FLUX 512x512 torch.float16: Implied constant = 2136 (Actual: 1068 MB)
- FLUX 768x768 torch.float16: Implied constant = 2112 (Actual: 2376 MB)
- FLUX 1024x1024 torch.float16: Implied constant = 2130 (Actual: 4260 MB)
- FLUX 1536x1536 torch.float16: Implied constant = 2120 (Actual: 9538 MB)
- FLUX 2048x2048 torch.float16: Implied constant = 2116 (Actual: 16932 MB)
- FLUX 512x512 torch.float32: Implied constant = 1422 (Actual: 1422 MB)
- FLUX 768x768 torch.float32: Implied constant = 1389 (Actual: 3126 MB)
- FLUX 1024x1024 torch.float32: Implied constant = 1380 (Actual: 5520 MB)
- FLUX 1536x1536 torch.float32: Implied constant = 1312 (Actual: 11806 MB)
- FLUX 2048x2048 torch.float32: Implied constant = 1244 (Actual: 19904 MB)
- FLUX 512x512 torch.bfloat16: Implied constant = 2136 (Actual: 1068 MB)
- FLUX 768x768 torch.bfloat16: Implied constant = 2112 (Actual: 2376 MB)
- FLUX 1024x1024 torch.bfloat16: Implied constant = 2129 (Actual: 4258 MB)
- FLUX 1536x1536 torch.bfloat16: Implied constant = 2119 (Actual: 9536 MB)
- FLUX 2048x2048 torch.bfloat16: Implied constant = 2116 (Actual: 16928 MB)
- SD1.5 512x512 torch.float16: Implied constant = 2036 (Actual: 1018 MB)
- SD1.5 768x768 torch.float16: Implied constant = 2084 (Actual: 2344 MB)
- SD1.5 1024x1024 torch.float16: Implied constant = 2113 (Actual: 4226 MB)
- SD1.5 1536x1536 torch.float16: Implied constant = 2120 (Actual: 9538 MB)
- SD1.5 2048x2048 torch.float16: Implied constant = 2124 (Actual: 16994 MB)
- SD1.5 512x512 torch.float32: Implied constant = 1532 (Actual: 1532 MB)
- SD1.5 768x768 torch.float32: Implied constant = 1432 (Actual: 3222 MB)
- SD1.5 1024x1024 torch.float32: Implied constant = 1422 (Actual: 5686 MB)
- SD1.5 1536x1536 torch.float32: Implied constant = 1351 (Actual: 12158 MB)
- SD1.5 2048x2048 torch.float32: Implied constant = 1284 (Actual: 20536 MB)
- SD1.5 512x512 torch.float32: Implied constant = 1554 (Actual: 1554 MB)
- SD1.5 768x768 torch.float32: Implied constant = 1433 (Actual: 3224 MB)
- SD1.5 1024x1024 torch.float32: Implied constant = 1422 (Actual: 5688 MB)
- SD1.5 1536x1536 torch.float32: Implied constant = 1351 (Actual: 12158 MB)
- SD1.5 2048x2048 torch.float32: Implied constant = 1283 (Actual: 20536 MB)
- SDXL 512x512 torch.float16: Implied constant = 2176 (Actual: 1088 MB)
- SDXL 768x768 torch.float16: Implied constant = 2135 (Actual: 2402 MB)
- SDXL 1024x1024 torch.float16: Implied constant = 2137 (Actual: 4274 MB)
- SDXL 1536x1536 torch.float16: Implied constant = 2128 (Actual: 9574 MB)
- SDXL 2048x2048 torch.float16: Implied constant = 2124 (Actual: 16994 MB)
- SDXL 512x512 torch.float32: Implied constant = 1532 (Actual: 1532 MB)
- SDXL 768x768 torch.float32: Implied constant = 1432 (Actual: 3222 MB)
- SDXL 1024x1024 torch.float32: Implied constant = 1422 (Actual: 5686 MB)
- SDXL 1536x1536 torch.float32: Implied constant = 1351 (Actual: 12158 MB)
- SDXL 2048x2048 torch.float32: Implied constant = 1284 (Actual: 20536 MB)
- SDXL 512x512 torch.float32: Implied constant = 1554 (Actual: 1554 MB)
- SDXL 768x768 torch.float32: Implied constant = 1433 (Actual: 3224 MB)
- SDXL 1024x1024 torch.float32: Implied constant = 1422 (Actual: 5688 MB)
- SDXL 1536x1536 torch.float32: Implied constant = 1351 (Actual: 12158 MB)
- SDXL 2048x2048 torch.float32: Implied constant = 1283 (Actual: 20536 MB)

### 4. SD1.5 vs SDXL Comparison

- At 1024x1024:
  - SD1.5: 4226 MB
  - SDXL: 4274 MB
  - SDXL uses 1% MORE memory than SD1.5
- At 512x512:
  - SD1.5: 1018 MB
  - SDXL: 1088 MB
  - SDXL uses 7% MORE memory than SD1.5

## Recommendations

1. **Adjust scaling constant for working memory:**
   - Current value: 2200
   - Median measured: 1532
   - 95th percentile: 2136
   - Recommendation: Use 2136 for safety margin

2. **Model-specific working memory:**
   - Consider different constants for different models
   - FLUX requires different handling than SD models

3. **Encode operations also need working memory:**
   - Currently only decode reserves working memory
   - Encode operations show significant memory usage

4. **Account for PyTorch memory reservation behavior:**
   - PyTorch reserves ~2-3x more memory than allocated
   - Working memory estimates should account for this
