# RefImage/IPAdapter Model Switching Implementation

This document describes the implementation of automatic RefImage/IPAdapter model switching when the base model architecture changes in InvokeAI.

## Overview

When a user changes the base model architecture (e.g., from SDXL to FLUX), the system now automatically switches RefImage/IPAdapter models to the first compatible model available for the new architecture. If no compatible models are available, the RefImage/IPAdapter models are cleared.

## Implementation Details

### Files Modified

1. **`invokeai/frontend/web/src/services/api/types.ts`**
   - Added `isApiModelConfig` type guard function
   - This function identifies API-based models (ChatGPT-4o, Imagen3, Imagen4, FLUX-Kontext API)

2. **`invokeai/frontend/web/src/app/store/middleware/listenerMiddleware/listeners/modelSelected.ts`**
   - Enhanced the `addModelSelectedListener` function to handle RefImage/IPAdapter model switching
   - Added logic to automatically switch both global and regional reference image models

### Key Components

#### Type Guard Function
```typescript
export const isApiModelConfig = (config: AnyModelConfig): config is ApiModelConfig => {
  return (
    isChatGPT4oModelConfig(config) ||
    isImagen3ModelConfig(config) ||
    isImagen4ModelConfig(config) ||
    isFluxKontextApiModelConfig(config)
  );
};
```

#### Model Selection Logic
The implementation includes a selector function that identifies all available global reference image models:

```typescript
const selectGlobalReferenceImageModels = (state: RootState): AnyModelConfig[] => {
  const allModels = selectIPAdapterModels(state);
  return allModels.filter((model: AnyModelConfig) => 
    isIPAdapterModelConfig(model) ||
    isFluxReduxModelConfig(model) ||
    isChatGPT4oModelConfig(model) ||
    isFluxKontextApiModelConfig(model) ||
    isFluxKontextModelConfig(model)
  );
};
```

### Functionality

#### What Happens When Base Model Changes

1. **Model Compatibility Check**: The system checks if existing RefImage/IPAdapter models are compatible with the new base model architecture

2. **Automatic Switching**: If incompatible models are found:
   - Finds the first compatible model for the new architecture
   - Automatically switches to that model
   - Logs the model switch for debugging

3. **Fallback Behavior**: If no compatible models are available:
   - Clears the incompatible model
   - Increments the `modelsCleared` counter
   - Shows a warning toast to the user

#### Areas Affected

1. **Global Reference Images**: Standalone reference images managed in the `refImages` slice
2. **Regional Guidance Reference Images**: Reference images associated with regional guidance entities on the canvas

### User Experience

- **Seamless Transition**: Users can switch between model architectures without manually reconfiguring RefImage/IPAdapter models
- **Intelligent Defaults**: The system automatically selects the first available compatible model
- **Clear Feedback**: Users receive notifications when models are cleared due to incompatibility
- **Preserved Configuration**: When switching to a compatible model, other settings (weight, method, etc.) are preserved when possible

### Technical Benefits

1. **Reduced Manual Work**: Users don't need to manually reconfigure RefImage/IPAdapter models when switching architectures
2. **Prevents Errors**: Automatically prevents incompatible model configurations that would cause generation failures
3. **Maintains Workflow**: Users can focus on creative work rather than technical model management
4. **Consistent Behavior**: The same logic applies to both global and regional reference images

### Integration Points

The implementation integrates with several existing systems:

- **Model Management**: Uses existing model selection and compatibility checking systems
- **State Management**: Works with Redux slices for canvas, reference images, and parameters
- **User Feedback**: Integrates with the existing toast notification system
- **Logging**: Provides detailed debug logging for troubleshooting

### Error Handling

- Graceful fallback when no compatible models are available
- Comprehensive logging for debugging model switching decisions
- User notifications through toast messages
- Maintains system stability even when model switching fails

## Usage

This feature works automatically without any user configuration required. When users:

1. Switch from one base model architecture to another (e.g., SDXL → FLUX)
2. Have existing RefImage/IPAdapter models configured
3. The system will automatically:
   - Check compatibility of existing models
   - Switch to the first compatible model for the new architecture
   - Or clear the model if none are compatible
   - Notify the user of any changes made

This implementation ensures a smooth user experience when working with different model architectures while maintaining the integrity of the generation pipeline.