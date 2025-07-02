# Reference Image Model Switching Implementation

## Overview

This implementation adds automatic model switching functionality for reference images when users change the main model. The system now automatically switches reference image models to compatible ones based on the architecture, with special handling for API models.

## Changes Made

### Enhanced Model Selection Listener

Modified `invokeai/frontend/web/src/app/store/middleware/listenerMiddleware/listeners/modelSelected.ts` to include:

1. **Automatic Reference Image Model Switching**: When the main model changes, the system automatically updates reference image models to maintain compatibility.

2. **API Model Special Handling**: For API models (ChatGPT-4o, Imagen3, Imagen4, FluxKontext), the system attempts to find an API model with the same name as the selected main model.

3. **Architecture-Based Fallback**: If no same-name API model is found, or for non-API models, the system selects the first compatible model for the given architecture.

4. **Global and Regional Reference Images**: The implementation handles both global reference images and regional guidance reference images.

## Implementation Details

### Key Features

1. **Compatible Model Detection**: Filters available models to find those compatible with the new main model's architecture (same base model).

2. **Smart Model Selection Logic**:
   - For API models: Attempts to match by name first, then falls back to first compatible model
   - For other models: Selects the first compatible model for the architecture

3. **Model Clearing**: If no compatible models are available, incompatible models are cleared to prevent errors.

4. **Comprehensive Coverage**: Handles both global reference images and regional guidance reference images.

### Error State Display

The existing error state display functionality was already in place:
- Reference image preview icons show an exclamation mark when no model is selected (`!entity.config.model`)
- The preview component has `data-is-error={!entity.config.model}` attribute for styling
- Error state is indicated by red border and error icon overlay

## User Experience

1. **Seamless Model Switching**: When users switch to a different model, reference image models automatically update to compatible ones.

2. **API Model Intelligence**: For API models, the system tries to maintain the same model name when possible.

3. **Clear Error Indication**: When no compatible model is available, the error state is clearly displayed.

4. **No User Interruption**: The switching happens automatically without requiring user intervention.

## Technical Implementation

### Type Safety
- All functions use proper TypeScript types
- No `any` types in the final implementation
- Proper type guards for model detection

### Performance
- Leverages existing Redux selectors for efficient state access
- Minimal computational overhead during model switching

### Logging
- Comprehensive debug logging for troubleshooting
- Clear messages for model switching and clearing actions

## Testing Considerations

When testing this feature:
1. Switch between different model architectures (SD, SDXL, FLUX, etc.)
2. Test with API models to verify name-matching behavior
3. Verify error states appear when no compatible models are available
4. Check both global and regional reference images are handled correctly