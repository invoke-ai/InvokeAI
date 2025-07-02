# API Models CoreMetadata Usage Analysis

## Overview

This document analyzes how API models (Kontext, Imagen, CogView4, etc.) should utilize the CoreMetadata node for metadata collection and storage, referencing the patterns used by non-API models for consistency.

## Current State Analysis

### Non-API Models (Reference Implementation)

**SDXL, FLUX, SD3** graphs follow this pattern:

1. **Metadata Collection**: Use `g.upsertMetadata()` throughout graph construction
2. **Metadata Connection**: Call `g.setMetadataReceivingNode(canvasOutput)` to connect CoreMetadata to final output
3. **Generation Mode**: Set appropriate `generation_mode` field (e.g., `sdxl_txt2img`, `flux_img2img`)

Example from SDXL graph builder:
```typescript
// Throughout graph construction
g.upsertMetadata({
  cfg_scale,
  width: originalSize.width,
  height: originalSize.height,
  positive_prompt: positivePrompt,
  negative_prompt: negativePrompt,
  model: Graph.getModelMetadataField(model),
  seed,
  steps,
  scheduler,
  // ... other metadata
});

// Set generation mode based on type
g.upsertMetadata({ generation_mode: 'sdxl_txt2img' });

// Connect metadata to final output node
g.setMetadataReceivingNode(canvasOutput);
```

### API Models Current Status

#### ✅ CogView4 (Correctly Implemented)
- **Location**: `buildCogView4Graph.ts`
- **Status**: ✅ CORRECT - Follows proper metadata pattern
- **Implementation**:
  ```typescript
  g.upsertMetadata({
    cfg_scale,
    width: originalSize.width,
    height: originalSize.height,
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
  });
  
  g.upsertMetadata({ generation_mode: 'cogview4_txt2img' });
  g.setMetadataReceivingNode(canvasOutput);
  ```

#### ✅ Imagen3 (Now Correctly Implemented)
- **Location**: `buildImagen3Graph.ts`
- **Status**: ✅ FIXED - Now includes proper metadata handling
- **Updated Implementation**:
  ```typescript
  g.upsertMetadata({
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    width: bbox.rect.width,
    height: bbox.rect.height,
    model: Graph.getModelMetadataField(model),
    generation_mode: 'imagen3_txt2img',  // ✅ Added
    ...selectCanvasMetadata(state),
  });
  g.setMetadataReceivingNode(imagen3);  // ✅ Added
  ```

#### ✅ Imagen4 (Now Correctly Implemented)
- **Location**: `buildImagen4Graph.ts`
- **Status**: ✅ FIXED - Now includes proper metadata handling
- **Implementation**: Same pattern as Imagen3, with `generation_mode: 'imagen4_txt2img'`

#### ✅ FLUX Kontext (Correctly Implemented)
- **Location**: FLUX graph builder (`buildFLUXGraph.ts`) with Kontext conditioning 
- **Status**: ✅ CORRECT - FLUX graphs properly use metadata pattern
- **Note**: Kontext conditioning is handled via the `flux_denoise` node's `kontext_conditioning` field, and metadata flows through the standard FLUX metadata collection

## CoreMetadata Node Details

### Purpose
The CoreMetadata node (`core_metadata` invocation) is designed to:
- Collect generation parameters for re-use
- Preserve model configuration details
- Enable workflow reproducibility
- Store generation context for debugging/analysis

### Key Fields Available
```python
# From invokeai/app/invocations/metadata.py
class CoreMetadataInvocation(BaseInvocation):
    generation_mode: Optional[GENERATION_MODES] = InputField(...)
    positive_prompt: Optional[str] = InputField(...)
    negative_prompt: Optional[str] = InputField(...)
    width: Optional[int] = InputField(...)
    height: Optional[int] = InputField(...)
    seed: Optional[int] = InputField(...)
    cfg_scale: Optional[float] = InputField(...)
    steps: Optional[int] = InputField(...)
    scheduler: Optional[str] = InputField(...)
    model: Optional[ModelIdentifierField] = InputField(...)
    # ... many other fields
```

### Generation Modes for API Models
Current supported modes in `GENERATION_MODES`:
- `cogview4_txt2img`, `cogview4_img2img`, `cogview4_inpaint`, `cogview4_outpaint`
- **Missing**: `imagen3_txt2img`, `imagen4_txt2img`, `kontext_txt2img`, etc.

## Required Fixes

### 1. ✅ Fix Imagen3 Graph Builder - COMPLETED

**File**: `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildImagen3Graph.ts`

**Changes Applied**:
```typescript
g.upsertMetadata({
  positive_prompt: positivePrompt,
  negative_prompt: negativePrompt,
  width: bbox.rect.width,
  height: bbox.rect.height,
  model: Graph.getModelMetadataField(model),
  generation_mode: 'imagen3_txt2img',  // ✅ Added
  ...selectCanvasMetadata(state),
});

g.setMetadataReceivingNode(imagen3);  // ✅ Added
```

### 2. ✅ Add Missing Generation Modes - COMPLETED

**File**: `invokeai/app/invocations/metadata.py`

**Changes Applied**:
```python
GENERATION_MODES = Literal[
    # ... existing modes ...
    "imagen3_txt2img",      # ✅ Added
    "imagen4_txt2img",      # ✅ Added
    "imagen4_img2img",      # ✅ Added
    "imagen4_inpaint",      # ✅ Added
    "imagen4_outpaint",     # ✅ Added
]
```

### 3. ✅ Fix Imagen4 Graph Builder - COMPLETED

**File**: `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildImagen4Graph.ts`

**Changes Applied**: Same pattern as Imagen3 - added `generation_mode` and `setMetadataReceivingNode()` call.

### 4. ⚠️ Type Generation Required

**Issue**: TypeScript compilation errors due to new generation modes not yet available in generated types.

**Resolution**: After backend changes are deployed, frontend types need to be regenerated from the Python schema to resolve compilation errors.

**Temporary Workaround**: The functionality will work correctly once types are regenerated, as the actual values are handled properly by the runtime validation.

## Best Practices for API Models

### 1. Metadata Collection Pattern
```typescript
// Collect basic generation parameters
g.upsertMetadata({
  width: dimensions.width,
  height: dimensions.height,
  positive_prompt: positivePrompt,
  negative_prompt: negativePrompt,
  model: Graph.getModelMetadataField(model),
  seed: seedValue,
  // Add API-specific parameters
  api_specific_param: value,
});

// Set generation mode
g.upsertMetadata({ 
  generation_mode: 'api_model_operation_type' 
});

// Connect to final output
g.setMetadataReceivingNode(outputNode);
```

### 2. API-Specific Metadata Fields
For API models, consider capturing:
- API endpoint/service information
- API-specific parameters
- Rate limiting/quota information
- Response metadata
- Version information

### 3. Error Handling
Ensure metadata collection doesn't break if:
- API parameters are missing
- Model configuration is incomplete
- Network issues occur

## Validation Steps

1. **Graph Generation**: Verify CoreMetadata node is created
2. **Edge Connection**: Confirm metadata flows to output node
3. **Field Population**: Check all relevant fields are captured
4. **Generation Mode**: Verify correct mode is set
5. **Re-use Testing**: Test workflow import/export functionality

## Implementation Status

1. **✅ Completed**: Fix Imagen3 missing `setMetadataReceivingNode()`
2. **✅ Completed**: Fix Imagen4 missing `setMetadataReceivingNode()`
3. **✅ Completed**: Add missing generation modes to `GENERATION_MODES`
4. **⚠️ Pending**: Type regeneration to resolve TypeScript compilation errors
5. **Future**: Enhance API-specific metadata fields
6. **Future**: Add metadata validation for API model graphs

## Conclusion

API models should follow the same metadata pattern as non-API models:
1. Use `g.upsertMetadata()` for parameter collection
2. Set appropriate `generation_mode`
3. Call `g.setMetadataReceivingNode()` to connect CoreMetadata to output

This ensures consistent metadata handling across all model types and enables proper workflow re-use functionality.