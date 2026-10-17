import type { ExternalModelFormValues } from './schemas';
import type { ExternalModelCapabilities, ModelConfig, ModelRecordChanges, StarterModel } from './types';

import { externalModelSchema } from './schemas';

export type ExternalModelSettingsModel = Pick<
  ModelConfig,
  'capabilities' | 'default_settings' | 'key' | 'name' | 'provider_id' | 'provider_model_id' | 'source' | 'type'
>;

export const supportsExternalModelSettings = (model: Pick<ModelConfig, 'type'>): boolean =>
  model.type === 'external_image_generator';

/**
 * Whether the starter catalog manages the record's capabilities and defaults: the backend overlays the catalog's
 * values on every response for a record whose source names a starter, so editing them would not persist. Records
 * without a source are matched by the source an external install would record.
 */
export const isStarterManagedExternalModel = (
  model: Pick<ExternalModelSettingsModel, 'provider_id' | 'provider_model_id'> & { source?: string | null },
  starters: readonly Pick<StarterModel, 'source'>[]
): boolean => {
  const source =
    typeof model.source === 'string' && model.source !== ''
      ? model.source
      : `external://${model.provider_id ?? ''}/${model.provider_model_id ?? ''}`;

  return starters.some((starter) => starter.source === source);
};

const toOptionalInteger = (value: number | null | undefined): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/** The editor's flat values for a record: what a draft of the external surface is a difference from. */
export const toExternalModelValues = (model: ExternalModelSettingsModel): ExternalModelFormValues => {
  const capabilities = model.capabilities ?? {};
  const modes = new Set(capabilities.modes ?? ['txt2img']);

  return {
    defaultHeight: toOptionalInteger(model.default_settings?.height),
    defaultNumImages: toOptionalInteger(model.default_settings?.num_images),
    defaultWidth: toOptionalInteger(model.default_settings?.width),
    maxImageHeight: toOptionalInteger(capabilities.max_image_size?.height),
    maxImageWidth: toOptionalInteger(capabilities.max_image_size?.width),
    maxImagesPerRequest: toOptionalInteger(capabilities.max_images_per_request),
    maxReferenceImages: toOptionalInteger(capabilities.max_reference_images),
    modeImg2img: modes.has('img2img'),
    modeInpaint: modes.has('inpaint'),
    modeTxt2img: modes.has('txt2img'),
    providerId: model.provider_id ?? '',
    providerModelId: model.provider_model_id ?? '',
    supportsNegativePrompt: capabilities.supports_negative_prompt ?? true,
    supportsReferenceImages: capabilities.supports_reference_images === true,
    supportsSeed: capabilities.supports_seed === true,
  };
};

/**
 * The PATCH body for saved values. The backend replaces `capabilities` as a whole, so the fields this editor does not
 * show (aspect-ratio tables, presets, mask format) are carried over from the current record.
 */
export const toExternalModelChanges = (
  values: ExternalModelFormValues,
  model: ExternalModelSettingsModel
): Pick<ModelRecordChanges, 'capabilities' | 'default_settings' | 'provider_id' | 'provider_model_id'> => {
  const capabilities: ExternalModelCapabilities = {
    ...model.capabilities,
    max_image_size:
      values.maxImageWidth !== null && values.maxImageHeight !== null
        ? { height: values.maxImageHeight, width: values.maxImageWidth }
        : null,
    max_images_per_request: values.maxImagesPerRequest,
    max_reference_images: values.maxReferenceImages,
    modes: [
      ...(values.modeTxt2img ? (['txt2img'] as const) : []),
      ...(values.modeImg2img ? (['img2img'] as const) : []),
      ...(values.modeInpaint ? (['inpaint'] as const) : []),
    ],
    supports_negative_prompt: values.supportsNegativePrompt,
    supports_reference_images: values.supportsReferenceImages,
    supports_seed: values.supportsSeed,
  };
  const hasDefaults = values.defaultWidth !== null || values.defaultHeight !== null || values.defaultNumImages !== null;

  return {
    capabilities,
    default_settings: hasDefaults
      ? { height: values.defaultHeight, num_images: values.defaultNumImages, width: values.defaultWidth }
      : null,
    provider_id: values.providerId.trim(),
    provider_model_id: values.providerModelId.trim(),
  };
};

/** The first schema message, like the other record editors' messages, so the UI shows it as is. */
export const validateExternalModel = (values: ExternalModelFormValues): string | null => {
  const result = externalModelSchema.safeParse(values);

  return result.success ? null : (result.error.issues[0]?.message ?? 'Invalid provider settings.');
};
