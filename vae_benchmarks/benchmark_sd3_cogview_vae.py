#!/usr/bin/env python3
"""
Benchmark script for SD3 and CogView4 VAE memory usage.
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


def load_vae(model_path: str, device: torch.device, dtype: torch.dtype, model_type: str) -> AutoencoderKL:
    """Load VAE model."""
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
    
    # Disable tiling for SD3/CogView4 (as shown in the invocation code)
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
    num_warmup: int = 2,
    num_runs: int = 5
) -> Dict:
    """Benchmark VAE encode operation."""
    height, width = resolution
    
    # Create test image
    image_tensor = create_test_image(height, width).to(device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            with torch.inference_mode():
                dist = vae.encode(image_tensor).latent_dist
                _ = dist.sample()
        clear_memory(device)
    
    # Actual benchmark runs
    results = []
    for _ in range(num_runs):
        clear_memory(device)
        
        # Measure memory before
        mem_before = get_memory_stats(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            with torch.inference_mode():
                dist = vae.encode(image_tensor).latent_dist
                latents = dist.sample().to(dtype=vae.dtype)
                latents = vae.config.scaling_factor * latents
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        encode_time = time.time() - start_time
        
        # Measure memory after (peak)
        mem_after = get_memory_stats(device)
        
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
        "dtype": str(dtype),
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
    num_warmup: int = 2,
    num_runs: int = 5
) -> Dict:
    """Benchmark VAE decode operation."""
    height, width = resolution
    
    # SD3 and CogView4 use different latent channel counts
    # SD3 uses 16 channels, CogView4 uses standard 4 channels
    # We'll detect based on the model config
    if hasattr(vae.config, 'latent_channels'):
        latent_channels = vae.config.latent_channels
    elif hasattr(vae.config, 'out_channels'):
        latent_channels = vae.config.out_channels
    else:
        # Default to 4 for standard VAE
        latent_channels = 4
    
    # Calculate latent dimensions (1/8 scale factor)
    latent_height = height // 8
    latent_width = width // 8
    
    # Create test latents
    latents = torch.randn(1, latent_channels, latent_height, latent_width).to(device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            with torch.inference_mode():
                scaled_latents = latents / vae.config.scaling_factor
                _ = vae.decode(scaled_latents, return_dict=False)[0]
        clear_memory(device)
    
    # Actual benchmark runs
    results = []
    for _ in range(num_runs):
        clear_memory(device)
        
        # Measure memory before
        mem_before = get_memory_stats(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            with torch.inference_mode():
                scaled_latents = latents / vae.config.scaling_factor
                image = vae.decode(scaled_latents, return_dict=False)[0]
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        decode_time = time.time() - start_time
        
        # Measure memory after (peak)
        mem_after = get_memory_stats(device)
        
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
        "operation": "decode",
        "dtype": str(dtype),
        "avg_time_s": sum(r["time_s"] for r in results) / len(results),
        "avg_reserved_mb": sum(r["reserved_mb"] for r in results) / len(results),
        "avg_allocated_mb": sum(r["allocated_mb"] for r in results) / len(results),
        "max_allocated_mb": max(r["peak_allocated_mb"] for r in results),
        "max_reserved_mb": max(r["peak_reserved_mb"] for r in results),
        "latent_shape": list(latents.shape),
        "latent_channels": latent_channels,
        "output_shape": results[0]["output_shape"],
    }
    
    return avg_result


def main():
    """Main benchmark function."""
    # Configuration
    models = [
        {
            "name": "SD3",
            "path": "/home/bat/invokeai-4.0.0/models/sd-3/main/SD3.5-medium/vae",
        },
        {
            "name": "CogView4",
            "path": "/home/bat/invokeai-4.0.0/models/cogview4/main/CogView4/vae",
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
    
    dtypes = [torch.float16, torch.float32]
    
    # Check if bfloat16 is supported
    if device.type == "cuda":
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
            dtypes.append(torch.bfloat16)
            del test_tensor
        except:
            print("bfloat16 not supported on this device")
    
    print(f"Device: {device}")
    print("=" * 80)
    
    all_results = []
    
    for model_config in models:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\nTesting {model_name} VAE")
        print(f"Model path: {model_path}")
        print("-" * 40)
        
        for dtype in dtypes:
            print(f"\nTesting with dtype: {dtype}")
            
            # Load model
            clear_memory(device)
            
            try:
                vae = load_vae(model_path, device, dtype, model_name)
                
                # Get model size in memory
                model_size_mb = sum(p.numel() * p.element_size() for p in vae.parameters()) / 1024 / 1024
                print(f"Model size in memory: {model_size_mb:.2f} MB")
                
                # Print VAE config info
                if hasattr(vae.config, 'latent_channels'):
                    print(f"Latent channels: {vae.config.latent_channels}")
                elif hasattr(vae.config, 'out_channels'):
                    print(f"Out channels: {vae.config.out_channels}")
                    
                print(f"Scaling factor: {vae.config.scaling_factor}")
                
                for resolution in resolutions:
                    print(f"\nResolution: {resolution[0]}x{resolution[1]}")
                    
                    # Test encode
                    try:
                        encode_result = benchmark_vae_encode(vae, resolution, device, dtype)
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
                    
                    # Test decode
                    try:
                        decode_result = benchmark_vae_decode(vae, resolution, device, dtype)
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
                
                # Clean up model
                del vae
                
            except Exception as e:
                print(f"Failed to load model: {e}")
            
            clear_memory(device)
    
    # Save results
    import json
    output_file = Path(__file__).parent / "sd3_cogview_vae_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE - SD3/CogView4 VAE")
    print("=" * 120)
    print(f"{'Model':<10} {'Resolution':<12} {'Operation':<10} {'Dtype':<12} {'Allocated (MB)':<15} {'Reserved (MB)':<15} {'Time (s)':<10}")
    print("-" * 120)
    
    for result in all_results:
        print(f"{result['model']:<10} {result['resolution']:<12} {result['operation']:<10} {str(result['dtype']):<12} "
              f"{result['avg_allocated_mb']:<15.2f} {result['avg_reserved_mb']:<15.2f} "
              f"{result['avg_time_s']:<10.3f}")


if __name__ == "__main__":
    main()