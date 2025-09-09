#!/bin/bash
# Run InvokeAI with optimized settings for Qwen-Image models

echo "Starting InvokeAI with Qwen-Image memory optimizations..."
echo "----------------------------------------"
echo "Recommendations for 24GB VRAM systems:"
echo "1. Set VRAM cache to 8-10GB in InvokeAI settings"
echo "2. Set RAM cache to 20-30GB (based on available system RAM)"
echo "3. Use bfloat16 precision (default in our loader)"
echo "----------------------------------------"

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0

# Optional: Limit CPU threads to prevent memory thrashing
export OMP_NUM_THREADS=8

# Run InvokeAI with your root directory
invokeai-web --root ~/invokeai/ \
  --precision bfloat16 \
  --max_cache_size 8.0 \
  --max_vram_cache_size 8.0

# Alternative: Use with config file
# invokeai-web --root ~/invokeai/ --config qwen_test_config.yaml