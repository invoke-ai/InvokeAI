#!/usr/bin/env python3
"""
Benchmark script for FLUX VAE memory usage.
Tests encode and decode operations at various resolutions.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file

# Add InvokeAI to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from invokeai.backend.flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
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
    

def load_flux_vae(model_path: str, device: torch.device, dtype: torch.dtype) -> AutoEncoder:
    """Load FLUX VAE model."""
    # FLUX VAE params from the codebase
    ae_params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    
    print(f"Loading FLUX VAE from {model_path}")
    model = AutoEncoder(ae_params)
    
    # Load weights
    sd = load_file(model_path)
    model.load_state_dict(sd, assign=True)
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return model


def create_test_image(height: int, width: int) -> torch.Tensor:
    """Create a test image tensor."""
    # Create a random image tensor in [-1, 1] range
    img_tensor = torch.randn(1, 3, height, width) * 0.5  # Scale down for more realistic values
    return img_tensor


def benchmark_vae_encode(
    vae: AutoEncoder,
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
            _ = vae.encode(image_tensor, sample=True)
        clear_memory(device)
    
    # Actual benchmark runs
    results = []
    for _ in range(num_runs):
        clear_memory(device)
        
        # Measure memory before
        mem_before = get_memory_stats(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            latents = vae.encode(image_tensor, sample=True)
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
        
        del latents
    
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
    vae: AutoEncoder,
    resolution: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    num_warmup: int = 2,
    num_runs: int = 5
) -> Dict:
    """Benchmark VAE decode operation."""
    height, width = resolution
    
    # Calculate latent dimensions (FLUX uses 1/8 scale factor)
    latent_height = height // 8
    latent_width = width // 8
    
    # Create test latents
    latents = torch.randn(1, 16, latent_height, latent_width).to(device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = vae.decode(latents)
        clear_memory(device)
    
    # Actual benchmark runs
    results = []
    for _ in range(num_runs):
        clear_memory(device)
        
        # Measure memory before
        mem_before = get_memory_stats(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            image = vae.decode(latents)
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
        
        del image
    
    # Calculate averages
    avg_result = {
        "resolution": f"{height}x{width}",
        "operation": "decode",
        "dtype": str(dtype),
        "avg_time_s": sum(r["time_s"] for r in results) / len(results),
        "avg_allocated_mb": sum(r["allocated_mb"] for r in results) / len(results),
        "avg_reserved_mb": sum(r["reserved_mb"] for r in results) / len(results),
        "max_allocated_mb": max(r["peak_allocated_mb"] for r in results),
        "max_reserved_mb": max(r["peak_reserved_mb"] for r in results),
        "latent_shape": list(latents.shape),
        "output_shape": results[0]["output_shape"],
    }
    
    return avg_result


def main():
    """Main benchmark function."""
    # Configuration
    model_path = "/home/bat/invokeai-4.0.0/models/flux/vae/FLUX.1-schnell_ae.safetensors"
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
    print(f"Model path: {model_path}")
    print("=" * 80)
    
    all_results = []
    
    for dtype in dtypes:
        print(f"\nTesting with dtype: {dtype}")
        print("-" * 40)
        
        # Load model once per dtype
        clear_memory(device)
        vae = load_flux_vae(model_path, device, dtype)
        
        # Get model size in memory
        model_size_mb = sum(p.numel() * p.element_size() for p in vae.parameters()) / 1024 / 1024
        print(f"Model size in memory: {model_size_mb:.2f} MB")
        
        for resolution in resolutions:
            print(f"\nResolution: {resolution[0]}x{resolution[1]}")
            
            # Test encode
            try:
                encode_result = benchmark_vae_encode(vae, resolution, device, dtype)
                encode_result["model"] = "FLUX"
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
                decode_result["model"] = "FLUX"
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
        clear_memory(device)
    
    # Save results
    import json
    output_file = Path(__file__).parent / "flux_vae_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - FLUX VAE")
    print("=" * 100)
    print(f"{'Resolution':<12} {'Operation':<10} {'Dtype':<12} {'Allocated (MB)':<15} {'Reserved (MB)':<15} {'Time (s)':<10}")
    print("-" * 100)
    
    for result in all_results:
        print(f"{result['resolution']:<12} {result['operation']:<10} {str(result['dtype']):<12} "
              f"{result['avg_allocated_mb']:<15.2f} {result['avg_reserved_mb']:<15.2f} "
              f"{result['avg_time_s']:<10.3f}")


if __name__ == "__main__":
    main()