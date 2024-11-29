import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';

export const getRegionalGuidanceWarnings = (
  entity: CanvasRegionalGuidanceState,
  model: ParameterModel | null
): string[] => {
  const warnings: string[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push('parameters.invoke.layer.emptyLayer');
  } else {
    if (entity.positivePrompt === null && entity.negativePrompt === null && entity.referenceImages.length === 0) {
      // Must have at least 1 prompt or IP Adapter
      warnings.push('parameters.invoke.layer.rgNoPromptsOrIPAdapters');
    }

    if (model) {
      if (model.base === 'sd-3' || model.base === 'sd-2') {
        // Unsupported model architecture
        warnings.push('parameters.invoke.layer.unsupportedModel');
      } else if (model.base === 'flux') {
        // Some features are not supported for flux models
        if (entity.negativePrompt !== null) {
          warnings.push('parameters.invoke.layer.rgNegativePromptNotSupported');
        }
        if (entity.referenceImages.length > 0) {
          warnings.push('parameters.invoke.layer.rgReferenceImagesNotSupported');
        }
        if (entity.autoNegative) {
          warnings.push('parameters.invoke.layer.rgAutoNegativeNotSupported');
        }
      } else {
        entity.referenceImages.forEach(({ ipAdapter }) => {
          if (!ipAdapter.model) {
            // No model selected
            warnings.push('parameters.invoke.layer.ipAdapterNoModelSelected');
          } else if (ipAdapter.model.base !== model.base) {
            // Supported model architecture but doesn't match
            warnings.push('parameters.invoke.layer.ipAdapterIncompatibleBaseModel');
          }

          if (!ipAdapter.image) {
            // No image selected
            warnings.push('parameters.invoke.layer.ipAdapterNoImageSelected');
          }
        });
      }
    }
  }

  return warnings;
};

export const getGlobalReferenceImageWarnings = (
  entity: CanvasReferenceImageState,
  model: ParameterModel | null
): string[] => {
  const warnings: string[] = [];

  if (!entity.ipAdapter.model) {
    // No model selected
    warnings.push('parameters.invoke.layer.ipAdapterNoModelSelected');
  } else if (model) {
    if (model.base === 'sd-3' || model.base === 'sd-2') {
      // Unsupported model architecture
      warnings.push('parameters.invoke.layer.unsupportedModel');
    } else if (entity.ipAdapter.model.base !== model.base) {
      // Supported model architecture but doesn't match
      warnings.push('parameters.invoke.layer.ipAdapterIncompatibleBaseModel');
    }
  }

  if (!entity.ipAdapter.image) {
    // No image selected
    warnings.push('parameters.invoke.layer.ipAdapterNoImageSelected');
  }

  return warnings;
};

export const getControlLayerWarnings = (entity: CanvasControlLayerState, model: ParameterModel | null): string[] => {
  const warnings: string[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push('parameters.invoke.layer.emptyLayer');
  } else {
    if (!entity.controlAdapter.model) {
      // No model selected
      warnings.push('parameters.invoke.layer.controlAdapterNoModelSelected');
    } else if (model) {
      if (model.base === 'sd-3' || model.base === 'sd-2') {
        // Unsupported model architecture
        warnings.push('parameters.invoke.layer.unsupportedModel');
      } else if (entity.controlAdapter.model.base !== model.base) {
        // Supported model architecture but doesn't match
        warnings.push('parameters.invoke.layer.controlAdapterIncompatibleBaseModel');
      }
    }
  }

  return warnings;
};

export const getRasterLayerWarnings = (entity: CanvasRasterLayerState, _model: ParameterModel | null): string[] => {
  const warnings: string[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push('parameters.invoke.layer.emptyLayer');
  }

  return warnings;
};

export const getInpaintMaskWarnings = (entity: CanvasInpaintMaskState, _model: ParameterModel | null): string[] => {
  const warnings: string[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push('parameters.invoke.layer.emptyLayer');
  }

  return warnings;
};
