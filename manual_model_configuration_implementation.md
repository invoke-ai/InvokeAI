# Manual Model Configuration Feature Implementation

## Overview
Added a manual configuration panel to the model manager's URL/Filepath installation form that allows users to manually configure model entry parameters instead of relying on automatic probing/scanning.

## Files Modified

### Frontend Components

1. **`InstallModelForm.tsx`**
   - Added manual configuration checkbox toggle
   - Added state management for manual configuration mode and config data
   - Modified the `useInstallModel` hook call to pass manual config when enabled
   - Added import for `ManualModelConfigPanel`

2. **`ManualModelConfigPanel.tsx`** (New Component)
   - Created comprehensive manual configuration panel with form fields
   - Includes prominent warning section with risks and disclaimers
   - Provides form controls for:
     - Model name and description
     - Model type (main, lora, controlnet, vae, etc.)
     - Base model type (sd-1, sd-2, sdxl, flux, etc.)
     - Format (diffusers, checkpoint, lycoris, etc.)
     - Prediction type (epsilon, v_prediction, sample)
     - Variant (normal, inpaint, depth)
   - Uses useCallback hooks for all change handlers to comply with linting rules

### Translation Keys

3. **`en.json`**
   - Added comprehensive translation keys for manual configuration feature:
     - Warning messages and risk descriptions
     - Form labels and placeholders
     - Model type, base model, format, and variant options
     - Helper text for various fields

## Key Features

### Risk Warnings
The implementation prominently displays warnings to users about the risks of manual configuration:
- No automatic probing/scanning
- Model may not work if misconfigured
- Incorrect settings can cause errors

### Comprehensive Configuration Options
Users can manually specify:
- **Model Type**: main, lora, controlnet, vae, ip_adapter, t2i_adapter, control_lora, embedding, spandrel_image_to_image
- **Base Model**: sd-1, sd-2, sdxl, sdxl-refiner, sd-3, flux, cogview4, any
- **Format**: diffusers, checkpoint, lycoris, invokeai, embed_file, embed_folder, onnx, bnb_quantized_nf4b
- **Prediction Type**: epsilon, v_prediction, sample
- **Variant**: normal, inpaint, depth

### Integration with Existing Flow
- Manual configuration is optional and disabled by default
- When enabled, the manual config object is passed to the existing `useInstallModel` hook
- The backend API already supports receiving a `config` parameter to override auto-probed values
- Seamlessly integrates with the existing model installation workflow

## Backend Integration
The feature leverages the existing backend API endpoint (`/api/v2/models/install`) which accepts:
- `source`: URL or file path
- `inplace`: Boolean for in-place installation
- `config`: Optional configuration object to override auto-probed values

When manual configuration is enabled, the config object contains the user-specified model parameters instead of relying on automatic detection.

## User Experience
1. User navigates to Model Manager → Add Model → URL or Local Path tab
2. User enters model URL or file path as usual
3. User can optionally check "Manual Configuration" checkbox
4. When enabled, a warning panel appears explaining the risks
5. User can fill in model configuration details manually
6. Installation proceeds with manual config instead of auto-probing

## Risk Mitigation
- Clear warning messages about bypassing automatic validation
- Prominent visual indicators (warning alert with icon)
- Explicit enumeration of risks
- Manual configuration is opt-in, not default behavior
- Users maintain ability to use automatic detection (default behavior)

This implementation provides advanced users with the flexibility to manually configure model parameters while clearly communicating the associated risks and maintaining the safety of automatic detection as the default approach.