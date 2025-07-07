import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  RefImageState,
} from 'features/controlLayers/store/types';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import type { AnyModelConfig, MainModelConfig } from 'services/api/types';

const WARNINGS = {
  UNSUPPORTED_MODEL: 'controlLayers.warnings.unsupportedModel',
  RG_NO_PROMPTS_OR_IP_ADAPTERS: 'controlLayers.warnings.rgNoPromptsOrIPAdapters',
  RG_NEGATIVE_PROMPT_NOT_SUPPORTED: 'controlLayers.warnings.rgNegativePromptNotSupported',
  RG_REFERENCE_IMAGES_NOT_SUPPORTED: 'controlLayers.warnings.rgReferenceImagesNotSupported',
  RG_AUTO_NEGATIVE_NOT_SUPPORTED: 'controlLayers.warnings.rgAutoNegativeNotSupported',
  RG_NO_REGION: 'controlLayers.warnings.rgNoRegion',
  IP_ADAPTER_NO_MODEL_SELECTED: 'controlLayers.warnings.ipAdapterNoModelSelected',
  IP_ADAPTER_INCOMPATIBLE_BASE_MODEL: 'controlLayers.warnings.ipAdapterIncompatibleBaseModel',
  IP_ADAPTER_NO_IMAGE_SELECTED: 'controlLayers.warnings.ipAdapterNoImageSelected',
  CONTROL_ADAPTER_NO_MODEL_SELECTED: 'controlLayers.warnings.controlAdapterNoModelSelected',
  CONTROL_ADAPTER_INCOMPATIBLE_BASE_MODEL: 'controlLayers.warnings.controlAdapterIncompatibleBaseModel',
  CONTROL_ADAPTER_NO_CONTROL: 'controlLayers.warnings.controlAdapterNoControl',
  FLUX_FILL_NO_WORKY_WITH_CONTROL_LORA: 'controlLayers.warnings.fluxFillIncompatibleWithControlLoRA',
} as const;

type WarningTKey = (typeof WARNINGS)[keyof typeof WARNINGS];

export const getRegionalGuidanceWarnings = (
  entity: CanvasRegionalGuidanceState,
  model: MainModelConfig | null | undefined
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state
    warnings.push(WARNINGS.RG_NO_REGION);
  }

  if (entity.positivePrompt === null && entity.negativePrompt === null && entity.referenceImages.length === 0) {
    // Must have at least 1 prompt or IP Adapter
    warnings.push(WARNINGS.RG_NO_PROMPTS_OR_IP_ADAPTERS);
  }

  if (model) {
    if (model.base === 'sd-3' || model.base === 'sd-2') {
      // Unsupported model architecture
      warnings.push(WARNINGS.UNSUPPORTED_MODEL);
      return warnings;
    }

    if (model.base === 'flux') {
      // Some features are not supported for flux models
      if (entity.negativePrompt !== null) {
        warnings.push(WARNINGS.RG_NEGATIVE_PROMPT_NOT_SUPPORTED);
      }
      if (entity.autoNegative) {
        warnings.push(WARNINGS.RG_AUTO_NEGATIVE_NOT_SUPPORTED);
      }
    }

    entity.referenceImages.forEach(({ config }) => {
      if (!config.model) {
        // No model selected
        warnings.push(WARNINGS.IP_ADAPTER_NO_MODEL_SELECTED);
      } else if (config.model.base !== model.base) {
        // Supported model architecture but doesn't match
        warnings.push(WARNINGS.IP_ADAPTER_INCOMPATIBLE_BASE_MODEL);
      }

      if (!config.image) {
        // No image selected
        warnings.push(WARNINGS.IP_ADAPTER_NO_IMAGE_SELECTED);
      }
    });
  }

  return warnings;
};

export const areBasesCompatibleForRefImage = (
  first?: ModelIdentifierField | AnyModelConfig | null,
  second?: ModelIdentifierField | AnyModelConfig | null
): boolean => {
  if (!first || !second) {
    return false;
  }
  if (first.base !== second.base) {
    return false;
  }
  if (
    first.base === 'flux' &&
    (first.name.toLowerCase().includes('kontext') || second.name.toLowerCase().includes('kontext')) &&
    first.key !== second.key
  ) {
    // FLUX Kontext requires the main model and the reference image model to be the same model
    return false;
  }
  return true;
};

export const getGlobalReferenceImageWarnings = (
  entity: RefImageState,
  model: MainModelConfig | null | undefined
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (model) {
    if (model.base === 'sd-3' || model.base === 'sd-2') {
      // Unsupported model architecture
      warnings.push(WARNINGS.UNSUPPORTED_MODEL);
      return warnings;
    }

    const { config } = entity;

    if (!config.model) {
      // No model selected
      warnings.push(WARNINGS.IP_ADAPTER_NO_MODEL_SELECTED);
    } else if (!areBasesCompatibleForRefImage(config.model, model)) {
      // Supported model architecture but doesn't match
      warnings.push(WARNINGS.IP_ADAPTER_INCOMPATIBLE_BASE_MODEL);
    }

    if (!entity.config.image) {
      // No image selected
      warnings.push(WARNINGS.IP_ADAPTER_NO_IMAGE_SELECTED);
    }
  }

  return warnings;
};

export const getControlLayerWarnings = (
  entity: CanvasControlLayerState,
  model: MainModelConfig | null | undefined
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  if (entity.objects.length === 0) {
    // Layer is in empty state
    warnings.push(WARNINGS.CONTROL_ADAPTER_NO_CONTROL);
  }

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
    } else if (
      model.base === 'flux' &&
      model.variant === 'inpaint' &&
      entity.controlAdapter.model.type === 'control_lora'
    ) {
      // FLUX inpaint variants are FLUX Fill models - not compatible w/ Control LoRA
      warnings.push(WARNINGS.FLUX_FILL_NO_WORKY_WITH_CONTROL_LORA);
    }
  }

  return warnings;
};

export const getRasterLayerWarnings = (
  _entity: CanvasRasterLayerState,
  _model: MainModelConfig | null | undefined
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  // There are no warnings at the moment for raster layers.

  return warnings;
};

export const getInpaintMaskWarnings = (
  _entity: CanvasInpaintMaskState,
  _model: MainModelConfig | null | undefined
): WarningTKey[] => {
  const warnings: WarningTKey[] = [];

  // There are no warnings at the moment for inpaint masks.

  return warnings;
};
