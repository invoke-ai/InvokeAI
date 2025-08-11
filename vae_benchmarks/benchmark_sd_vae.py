#!/usr/bin/env python3
"""
Benchmark script for SD1.5/SDXL VAE memory usage.
Tests encode and decode operations at various resolutions.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from diffusers import AutoencoderKL
from PIL import Image

# Add InvokeAI to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from invokeai.backend.util.devices import TorchDevice


def get_memory_stats(device: torch.device) -> Dict[str, float]:
    """Get current GPU memory statistics in MB."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024 / 1024,
        }
    return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0, "max_reserved_mb": 0}


def clear_memory(device: torch.device):
    """Clear GPU memory and reset statistics."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def load_sd_vae(model_path: str, device: torch.device, dtype: torch.dtype, model_type: str) -> AutoencoderKL:
    """Load SD VAE model."""
    print(f"Loading {model_type} VAE from {model_path}")
    
    # Check if it's a single file or directory
    model_path = Path(model_path)
    
    if model_path.is_file():
        # Load from single file (checkpoint)
        vae = AutoencoderKL.from_single_file(
            model_path,
            torch_dtype=dtype,
        )
    else:
        # Load from directory (diffusers format)
        vae = AutoencoderKL.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
    
    vae = vae.to(device)
    vae.eval()
    
    # Disable tiling by default for consistent benchmarks
    vae.disable_tiling()
    
    return vae


def create_test_image(height: int, width: int) -> torch.Tensor:
    """Create a test image tensor."""
    # Create a random image tensor in [-1, 1] range
    img_tensor = torch.randn(1, 3, height, width) * 0.5  # Scale down for more realistic values
    return img_tensor


def benchmark_vae_encode(
    vae: AutoencoderKL,
    resolution: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    use_fp32: bool = False,
    num_warmup: int = 2,
    num_runs: int = 5
) -> Dict:
    """Benchmark VAE encode operation."""
    height, width = resolution
    
    # Create test image
    image_tensor = create_test_image(height, width).to(device=device, dtype=dtype)
    
    # Store original dtype
    orig_dtype = vae.dtype
    
    # Warmup runs
    for _ in range(num_warmup):
        if use_fp32:
            vae.to(dtype=torch.float32)
        
        with torch.no_grad():
            with torch.inference_mode():
                dist = vae.encode(image_tensor).latent_dist
                _ = dist.sample()
        
        if use_fp32:
            vae.to(dtype=orig_dtype)
        
        clear_memory(device)
    
    # Actual benchmark runs
    results = []
    for _ in range(num_runs):
        clear_memory(device)
        
        # Measure memory before
        mem_before = get_memory_stats(device)
        
        if use_fp32:
            vae.to(dtype=torch.float32)
            image_tensor = image_tensor.to(dtype=torch.float32)
        
        start_time = time.time()
        
        with torch.no_grad():
            with torch.inference_mode():
                dist = vae.encode(image_tensor).latent_dist
                latents = dist.sample()
                latents = vae.config.scaling_factor * latents
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        encode_time = time.time() - start_time
        
        # Measure memory after (peak)
        mem_after = get_memory_stats(device)
        
        if use_fp32:
            vae.to(dtype=orig_dtype)
            image_tensor = image_tensor.to(dtype=orig_dtype)
        
        # Calculate memory used
        allocated_diff = mem_after["max_allocated_mb"] - mem_before["allocated_mb"]
        reserved_diff = mem_after["max_reserved_mb"] - mem_before["reserved_mb"]
        
        results.append({
            "time_s": encode_time,
            "allocated_mb": allocated_diff,
            "reserved_mb": reserved_diff,
            "peak_allocated_mb": mem_after["max_allocated_mb"],
            "peak_reserved_mb": mem_after["max_reserved_mb"],
            "latent_shape": list(latents.shape),
        })
        
        del latents, dist
    
    # Calculate averages
    avg_result = {
        "resolution": f"{height}x{width}",
        "operation": "encode",
        "dtype": str(torch.float32 if use_fp32 else dtype),
        "avg_time_s": sum(r["time_s"] for r in results) / len(results),
        "avg_allocated_mb": sum(r["allocated_mb"] for r in results) / len(results),
        "avg_reserved_mb": sum(r["reserved_mb"] for r in results) / len(results),
        "max_allocated_mb": max(r["peak_allocated_mb"] for r in results),
        "max_reserved_mb": max(r["peak_reserved_mb"] for r in results),
        "latent_shape": results[0]["latent_shape"],
    }
    
    return avg_result


def benchmark_vae_decode(
    vae: AutoencoderKL,
    resolution: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    use_fp32: bool = False,
    use_tiling: bool = False,
    tile_size: int = 512,
    num_warmup: int = 2,
    num_runs: int = 5
) -> Dict:
    """Benchmark VAE decode operation."""
    height, width = resolution
    
    # Calculate latent dimensions (SD uses 1/8 scale factor)
    latent_height = height // 8
    latent_width = width // 8
    
    # Create test latents
    latents = torch.randn(1, 4, latent_height, latent_width).to(device=device, dtype=dtype)
    
    # Store original dtype
    orig_dtype = vae.dtype
    
    # Configure tiling
    if use_tiling:
        vae.enable_tiling()
        vae.tile_sample_min_size = tile_size
        vae.tile_latent_min_size = tile_size // 8
        vae.tile_overlap_factor = 0.25
    else:
        vae.disable_tiling()
    
    # Warmup runs
    for _ in range(num_warmup):
        if use_fp32:
            vae.to(dtype=torch.float32)
            test_latents = latents.to(dtype=torch.float32)
        else:
            test_latents = latents.to(dtype=dtype)
        
        with torch.no_grad():
            with torch.inference_mode():
                scaled_latents = test_latents / vae.config.scaling_factor
                _ = vae.decode(scaled_latents, return_dict=False)[0]
        
        if use_fp32:
            vae.to(dtype=orig_dtype)
        
        clear_memory(device)
    
    # Actual benchmark runs
    results = []
    for _ in range(num_runs):
        clear_memory(device)
        
        # Measure memory before
        mem_before = get_memory_stats(device)
        
        if use_fp32:
            vae.to(dtype=torch.float32)
            test_latents = latents.to(dtype=torch.float32)
        else:
            test_latents = latents.to(dtype=dtype)
        
        start_time = time.time()
        
        with torch.no_grad():
            with torch.inference_mode():
                scaled_latents = test_latents / vae.config.scaling_factor
                image = vae.decode(scaled_latents, return_dict=False)[0]
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        decode_time = time.time() - start_time
        
        # Measure memory after (peak)
        mem_after = get_memory_stats(device)
        
        if use_fp32:
            vae.to(dtype=orig_dtype)
        
        # Calculate memory used
        allocated_diff = mem_after["max_allocated_mb"] - mem_before["allocated_mb"]
        reserved_diff = mem_after["max_reserved_mb"] - mem_before["reserved_mb"]
        
        results.append({
            "time_s": decode_time,
            "allocated_mb": allocated_diff,
            "reserved_mb": reserved_diff,
            "peak_allocated_mb": mem_after["max_allocated_mb"],
            "peak_reserved_mb": mem_after["max_reserved_mb"],
            "output_shape": list(image.shape),
        })
        
        del image, scaled_latents
    
    # Calculate averages
    avg_result = {
        "resolution": f"{height}x{width}",
        "operation": "decode" + ("_tiled" if use_tiling else ""),
        "dtype": str(torch.float32 if use_fp32 else dtype),
        "avg_time_s": sum(r["time_s"] for r in results) / len(results),
        "avg_allocated_mb": sum(r["allocated_mb"] for r in results) / len(results),
        "avg_reserved_mb": sum(r["reserved_mb"] for r in results) / len(results),
        "max_allocated_mb": max(r["peak_allocated_mb"] for r in results),
        "max_reserved_mb": max(r["peak_reserved_mb"] for r in results),
        "latent_shape": list(latents.shape),
        "output_shape": results[0]["output_shape"],
        "tiling": use_tiling,
        "tile_size": tile_size if use_tiling else None,
    }
    
    return avg_result


def main():
    """Main benchmark function."""
    # Configuration
    models = [
        {
            "name": "SD1.5",
            "path": "/home/bat/invokeai-4.0.0/models/sd-1/vae/sd-vae-ft-mse",
        },
        {
            "name": "SDXL",
            "path": "/home/bat/invokeai-4.0.0/models/sdxl/vae/sdxl-vae-fp16-fix",
        },
    ]
    
    device = TorchDevice.choose_torch_device()
    
    # Test configurations
    resolutions = [
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1536, 1536),
        (2048, 2048),
    ]
    
    # Test both fp16 and fp32 modes
    test_configs = [
        {"dtype": torch.float16, "use_fp32": False},
        {"dtype": torch.float16, "use_fp32": True},  # Mixed precision mode
        {"dtype": torch.float32, "use_fp32": False},
    ]
    
    print(f"Device: {device}")
    print("=" * 80)
    
    all_results = []
    
    for model_config in models:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\nTesting {model_name} VAE")
        print(f"Model path: {model_path}")
        print("-" * 40)
        
        for config in test_configs:
            dtype = config["dtype"]
            use_fp32 = config["use_fp32"]
            
            dtype_str = "fp32" if use_fp32 else str(dtype)
            print(f"\nTesting with dtype: {dtype_str}")
            
            # Load model
            clear_memory(device)
            
            try:
                vae = load_sd_vae(model_path, device, dtype, model_name)
                
                # Get model size in memory
                model_size_mb = sum(p.numel() * p.element_size() for p in vae.parameters()) / 1024 / 1024
                print(f"Model size in memory: {model_size_mb:.2f} MB")
                
                for resolution in resolutions:
                    print(f"\nResolution: {resolution[0]}x{resolution[1]}")
                    
                    # Test encode
                    try:
                        encode_result = benchmark_vae_encode(vae, resolution, device, dtype, use_fp32)
                        encode_result["model"] = model_name
                        encode_result["model_size_mb"] = model_size_mb
                        all_results.append(encode_result)
                        
                        print(f"  Encode - Allocated: {encode_result['avg_allocated_mb']:.2f} MB, "
                              f"Reserved: {encode_result['avg_reserved_mb']:.2f} MB, "
                              f"Time: {encode_result['avg_time_s']:.3f}s")
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"  Encode - OOM: {e}")
                    except Exception as e:
                        print(f"  Encode - Error: {e}")
                    
                    # Test decode (normal)
                    try:
                        decode_result = benchmark_vae_decode(vae, resolution, device, dtype, use_fp32, use_tiling=False)
                        decode_result["model"] = model_name
                        decode_result["model_size_mb"] = model_size_mb
                        all_results.append(decode_result)
                        
                        print(f"  Decode - Allocated: {decode_result['avg_allocated_mb']:.2f} MB, "
                              f"Reserved: {decode_result['avg_reserved_mb']:.2f} MB, "
                              f"Time: {decode_result['avg_time_s']:.3f}s")
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"  Decode - OOM: {e}")
                    except Exception as e:
                        print(f"  Decode - Error: {e}")
                    
                    # Test decode (tiled) for larger resolutions
                    if resolution[0] >= 1024:
                        try:
                            decode_tiled_result = benchmark_vae_decode(
                                vae, resolution, device, dtype, use_fp32, 
                                use_tiling=True, tile_size=512
                            )
                            decode_tiled_result["model"] = model_name
                            decode_tiled_result["model_size_mb"] = model_size_mb
                            all_results.append(decode_tiled_result)
                            
                            print(f"  Decode (Tiled) - Allocated: {decode_tiled_result['avg_allocated_mb']:.2f} MB, "
                                  f"Reserved: {decode_tiled_result['avg_reserved_mb']:.2f} MB, "
                                  f"Time: {decode_tiled_result['avg_time_s']:.3f}s")
                        except torch.cuda.OutOfMemoryError as e:
                            print(f"  Decode (Tiled) - OOM: {e}")
                        except Exception as e:
                            print(f"  Decode (Tiled) - Error: {e}")
                
                # Clean up model
                del vae
                
            except Exception as e:
                print(f"Failed to load model: {e}")
            
            clear_memory(device)
    
    # Save results
    import json
    output_file = Path(__file__).parent / "sd_vae_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE - SD VAE")
    print("=" * 120)
    print(f"{'Model':<8} {'Resolution':<12} {'Operation':<15} {'Dtype':<12} {'Allocated (MB)':<15} {'Reserved (MB)':<15} {'Time (s)':<10}")
    print("-" * 120)
    
    for result in all_results:
        print(f"{result['model']:<8} {result['resolution']:<12} {result['operation']:<15} {str(result['dtype']):<12} "
              f"{result['avg_allocated_mb']:<15.2f} {result['avg_reserved_mb']:<15.2f} "
              f"{result['avg_time_s']:<10.3f}")


if __name__ == "__main__":
    main()