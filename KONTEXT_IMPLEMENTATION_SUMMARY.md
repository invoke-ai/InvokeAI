# FLUX Kontext Implementation Summary

## Overview
Successfully implemented the complete architectural plan for adding "Kontext" reference image conditioning to the FLUX workflow within InvokeAI. The implementation follows the existing modular, invocation-based architecture.

## Implementation Steps Completed

### Step 1: Data Structure Definition ✅
**File:** `invokeai/app/invocations/fields.py`
- Added `FluxKontextConditioningField` class
- Added field description for `flux_kontext_conditioning`
- Follows the same pattern as existing `FluxFillConditioningField` and `FluxReduxConditioningField`

### Step 2: User-Facing Node ✅
**File:** `invokeai/app/invocations/flux_kontext.py` (new file)
- Created `FluxKontextInvocation` class
- Created `FluxKontextOutput` class  
- Provides simple interface for users to input reference images
- Packages image into `FluxKontextConditioningField` type

### Step 3: Denoise Orchestrator Integration ✅
**File:** `invokeai/app/invocations/flux_denoise.py`
- Added import for `FluxKontextConditioningField`
- Added `kontext_conditioning` input field to `FluxDenoiseInvocation` class
- Field accepts optional connection input for Kontext conditioning

### Step 4: Backend Logic via KontextExtension ✅
**File:** `invokeai/backend/flux/extensions/kontext_extension.py` (new file)

#### KontextExtension Features:
- **Reference Image Processing**: Encodes reference images to latents using VAE
- **Position ID Generation**: Creates positional IDs with offset to distinguish reference tokens
- **Token Concatenation**: Combines reference image tokens with main generation tokens
- **Batch Handling**: Automatically handles batch size mismatches

#### Integration in FluxDenoiseInvocation:
- Instantiates KontextExtension when `kontext_conditioning` is provided
- Requires `controlnet_vae` for image encoding (reuses existing VAE infrastructure)
- Stores original sequence length before applying extension
- Applies extension before denoise call to combine image tokens and IDs
- Passes combined tensors to transformer model
- **Critical Fix**: Extracts only main image tokens after denoising to avoid unpacking errors

## Key Technical Details

### Token Concatenation Approach
The implementation uses the same core approach as the original ComfyUI example:
- Reference image is VAE-encoded and patchified into tokens
- Positional IDs are generated with `idx_offset=1` to distinguish from main image tokens (which use `idx_offset=0`)
- Reference tokens are concatenated to main image tokens along the sequence dimension
- Combined sequence is processed by the FLUX transformer

### VAE Reuse
- Leverages existing `controlnet_vae` field instead of adding a separate VAE field
- Reuses `FluxVaeEncodeInvocation.vae_encode()` static method
- Consistent with existing architectural patterns

### Error Handling
- Validates that VAE is provided when Kontext conditioning is used
- Proper device and dtype handling throughout the pipeline
- Batch size normalization for edge cases

### Sequence Length Management
**Issue**: When concatenating reference tokens with main image tokens, the sequence length doubles (e.g., 4096 → 8192), but the `unpack` function expects only the original main image sequence length.

**Solution**: 
1. Store original sequence length before applying KontextExtension
2. Pass combined sequence (main + reference tokens) to transformer
3. After denoising, extract only the main image portion: `x[:, :original_seq_len, :]`
4. Unpack only the main image tokens, avoiding shape mismatch errors

## Files Modified/Created

### New Files:
1. `invokeai/app/invocations/flux_kontext.py` - User-facing invocation node
2. `invokeai/backend/flux/extensions/kontext_extension.py` - Backend processing logic

### Modified Files:
1. `invokeai/app/invocations/fields.py` - Added data structures
2. `invokeai/app/invocations/flux_denoise.py` - Integrated extension into pipeline

## Verification
- All files pass Python syntax validation (`python3 -m py_compile`)
- Implementation follows established InvokeAI patterns
- Maintains compatibility with existing FLUX features

## Usage
Users can now:
1. Add a "Kontext Conditioning - FLUX" node to their workflow
2. Connect a reference image to the node
3. Connect the output to the "Kontext Conditioning" input of a FLUX Denoise node
4. Ensure a VAE is connected to the "ControlNet VAE" input
5. Generate images that reference the provided image

## Architecture Benefits
- **Modular**: Clean separation between user interface and backend logic
- **Extensible**: Easy to add additional parameters (e.g., weight control) in the future
- **Consistent**: Follows existing InvokeAI patterns for conditioning
- **Efficient**: Pre-processes reference image once, reuses across denoising steps

The implementation is complete and ready for testing with actual FLUX models.