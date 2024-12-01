import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';

export const WARNINGS = {
  EMPTY_LAYER: 'parameters.invoke.layer.emptyLayer',
  UNSUPPORTED_MODEL: 'parameters.invoke.layer.unsupportedModel',
  RG_NO_PROMPTS_OR_IP_ADAPTERS: 'parameters.invoke.layer.rgNoPromptsOrIPAdapters',
  RG_NEGATIVE_PROMPT_NOT_SUPPORTED: 'parameters.invoke.layer.rgNegativePromptNotSupported',
  RG_REFERENCE_IMAGES_NOT_SUPPORTED: 'parameters.invoke.layer.rgReferenceImagesNotSupported',
  RG_AUTO_NEGATIVE_NOT_SUPPORTED: 'parameters.invoke.layer.rgAutoNegativeNotSupported',
  IP_ADAPTER_NO_MODEL_SELECTED: 'parameters.invoke.layer.ipAdapterNoModelSelected',
  IP_ADAPTER_INCOMPATIBLE_BASE_MODEL: 'parameters.invoke.layer.ipAdapterIncompatibleBaseModel',
  IP_ADAPTER_NO_IMAGE_SELECTED: 'parameters.invoke.layer.ipAdapterNoImageSelected',
  CONTROL_ADAPTER_NO_MODEL_SELECTED: 'parameters.invoke.layer.controlAdapterNoModelSelected',
  CONTROL_ADAPTER_INCOMPATIBLE_BASE_MODEL: 'parameters.invoke.layer.controlAdapterIncompatibleBaseModel',
} as const;

type WarningTKey = (typeof WARNINGS)[keyof typeof WARNINGS];

export const getRegionalGuidanceWarnings = (
  entity: CanvasRegionalGuidanceState,
  model: ParameterModel | null
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push(WARNINGS.EMPTY_LAYER);
  } else {
    if (entity.positivePrompt === null && entity.negativePrompt === null && entity.referenceImages.length === 0) {
      // Must have at least 1 prompt or IP Adapter
      warnings.push(WARNINGS.RG_NO_PROMPTS_OR_IP_ADAPTERS);
    }

    if (model) {
      if (model.base === 'sd-3' || model.base === 'sd-2') {
        // Unsupported model architecture
        warnings.push(WARNINGS.UNSUPPORTED_MODEL);
      } else if (model.base === 'flux') {
        // Some features are not supported for flux models
        if (entity.negativePrompt !== null) {
          warnings.push(WARNINGS.RG_NEGATIVE_PROMPT_NOT_SUPPORTED);
        }
        if (entity.referenceImages.length > 0) {
          warnings.push(WARNINGS.RG_REFERENCE_IMAGES_NOT_SUPPORTED);
        }
        if (entity.autoNegative) {
          warnings.push(WARNINGS.RG_AUTO_NEGATIVE_NOT_SUPPORTED);
        }
      } else {
        entity.referenceImages.forEach(({ ipAdapter }) => {
          if (!ipAdapter.model) {
            // No model selected
            warnings.push(WARNINGS.IP_ADAPTER_NO_MODEL_SELECTED);
          } else if (ipAdapter.model.base !== model.base) {
            // Supported model architecture but doesn't match
            warnings.push(WARNINGS.IP_ADAPTER_INCOMPATIBLE_BASE_MODEL);
          }

          if (!ipAdapter.image) {
            // No image selected
            warnings.push(WARNINGS.IP_ADAPTER_NO_IMAGE_SELECTED);
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
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (!entity.ipAdapter.model) {
    // No model selected
    warnings.push(WARNINGS.IP_ADAPTER_NO_MODEL_SELECTED);
  } else if (model) {
    if (model.base === 'sd-3' || model.base === 'sd-2') {
      // Unsupported model architecture
      warnings.push(WARNINGS.UNSUPPORTED_MODEL);
    } else if (entity.ipAdapter.model.base !== model.base) {
      // Supported model architecture but doesn't match
      warnings.push(WARNINGS.IP_ADAPTER_INCOMPATIBLE_BASE_MODEL);
    }
  }

  if (!entity.ipAdapter.image) {
    // No image selected
    warnings.push(WARNINGS.IP_ADAPTER_NO_IMAGE_SELECTED);
  }

  return warnings;
};

export const getControlLayerWarnings = (
  entity: CanvasControlLayerState,
  model: ParameterModel | null
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push(WARNINGS.EMPTY_LAYER);
  } else {
    if (!entity.controlAdapter.model) {
      // No model selected
      warnings.push(WARNINGS.CONTROL_ADAPTER_NO_MODEL_SELECTED);
    } else if (model) {
      if (model.base === 'sd-3' || model.base === 'sd-2') {
        // Unsupported model architecture
        warnings.push(WARNINGS.UNSUPPORTED_MODEL);
      } else if (entity.controlAdapter.model.base !== model.base) {
        // Supported model architecture but doesn't match
        warnings.push(WARNINGS.CONTROL_ADAPTER_INCOMPATIBLE_BASE_MODEL);
      }
    }
  }

  return warnings;
};

export const getRasterLayerWarnings = (
  entity: CanvasRasterLayerState,
  _model: ParameterModel | null
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push(WARNINGS.EMPTY_LAYER);
  }

  return warnings;
};

export const getInpaintMaskWarnings = (
  entity: CanvasInpaintMaskState,
  _model: ParameterModel | null
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state - skip other checks
    warnings.push(WARNINGS.EMPTY_LAYER);
  }

  return warnings;
};
