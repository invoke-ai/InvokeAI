import type {
  ComponentModelConfig,
  FluxReduxImageInfluence,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateWidgetValues,
  IPAdapterMethod,
  VaeModelConfig,
} from '@features/generation/contracts';

import {
  cloneGenerateWidgetValues,
  createReferenceImageId,
  DEFAULT_LORA_WEIGHT_CONFIG,
  getCompatibleReferenceImages,
  getDefaultGenerateSettings,
  getDefaultLoraWeight,
  getDefaultReferenceImageConfig,
  getGenerateModelSelectionResult,
  getGenerationUiPolicy,
  getMaxReferenceImages,
  getModelDefaultVae,
  getSettingsWithModelDefaults,
  hasArchitectureCapabilities,
  hasModelDefaultVae,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isReferenceImageSupported,
  isVaeCompatibleWithGenerateModel,
} from '@features/generation/settings';

import {
  getBoolean,
  getCfgRescaleMultiplier,
  getCfgScale,
  getMetadataSize,
  getScheduler,
  getSeed,
  getSteps,
  getSupportedClipSkip,
  withDimensions,
} from './imageRecall';

/**
 * Wire payload of the backend's `recall_parameters_updated` socket event, emitted
 * by `POST /api/v1/recall/{queue_id}`. `parameters` carries the request body after
 * the backend resolved model names to keys and image names to `{ image_name,
 * width, height }`; in strict mode every omitted key is present as `null` (lists
 * as `[]`), and `append: true` rides along as a pseudo-parameter.
 */
export interface RecallParametersUpdatedEvent {
  parameters: Record<string, unknown>;
  queue_id: string;
  user_id: string;
}

export type RecallParameterField =
  | 'model'
  | 'vae'
  | 'prompts'
  | 'seed'
  | 'size'
  | 'steps'
  | 'cfg'
  | 'scheduler'
  | 'seamless'
  | 'clipSkip'
  | 'loras'
  | 'referenceImages';

export type RecallParameterSkipReason = 'unsupported' | 'unresolved' | 'incompatible' | 'invalid' | 'limit';

export interface RecallParameterSkip {
  detail?: string;
  key: string;
  reason: RecallParameterSkipReason;
}

export interface RecallParametersResult {
  fields: RecallParameterField[];
  /** The project's own reference images (after any model switch), for callers that must undo a replacement. */
  retainedReferenceImages: GenerateReferenceImage[];
  skipped: RecallParameterSkip[];
  values: GenerateWidgetValues;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export const isRecallParametersUpdatedEvent = (payload: unknown): payload is RecallParametersUpdatedEvent =>
  isRecord(payload) && typeof payload.user_id === 'string' && isRecord(payload.parameters);

/** Backend graphs that feed `cfgScale` to a `guidance` input, so the API's `guidance` names the same knob. */
const GUIDANCE_BASES = new Set(['flux', 'flux2', 'qwen-image', 'z-image']);

/** Applied without reading architecture policy; every other handled key is snapped, coerced or defaulted by it. */
const POLICY_FREE_KEYS = new Set([
  'append',
  'guidance',
  'loras',
  'negative_prompt',
  'positive_prompt',
  'seamless_x',
  'seamless_y',
  'seed',
]);

/** Whether a payload can be applied now: without the table, only policy-free keys may be. */
export const isRecallParametersAvailable = (parameters: Record<string, unknown>): boolean =>
  hasArchitectureCapabilities() ||
  Object.keys(parameters).every((key) => POLICY_FREE_KEYS.has(key) || !HANDLED_KEYS.has(key));

const HANDLED_KEYS = new Set([
  'append',
  'cfg_rescale_multiplier',
  'cfg_scale',
  'clip_skip',
  'guidance',
  'height',
  'ip_adapters',
  'loras',
  'model',
  'negative_prompt',
  'positive_prompt',
  'reference_images',
  'scheduler',
  'seamless_x',
  'seamless_y',
  'seed',
  'steps',
  'vae_model',
  'width',
]);

const IP_ADAPTER_METHODS = new Set<IPAdapterMethod>(['full', 'style', 'composition', 'style_strong', 'style_precise']);
const FLUX_REDUX_INFLUENCES = new Set<FluxReduxImageInfluence>(['lowest', 'low', 'medium', 'high', 'highest']);

const findModel = <T extends ComponentModelConfig>(candidates: readonly T[], identifier: string): T | undefined =>
  candidates.find((model) => model.key === identifier) ?? candidates.find((model) => model.name === identifier);

const toReferenceAsset = (value: unknown): GenerateReferenceImageAsset | null =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  value.image_name.length > 0 &&
  isFiniteNumber(value.width) &&
  value.width > 0 &&
  isFiniteNumber(value.height) &&
  value.height > 0
    ? { original: { image: { height: value.height, image_name: value.image_name, width: value.width } } }
    : null;

const toReferenceImage = (config: GenerateReferenceImageConfig): GenerateReferenceImage => ({
  config,
  id: createReferenceImageId(),
  isEnabled: true,
});

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/**
 * Apply present payload keys; strict nulls reset to effective-model defaults. Report unsupported fields and
 * unresolved models in skipped. The caller verifies reference availability and commits.
 */
export const buildRecallParametersSettings = ({
  currentValues,
  models,
  parameters,
  supportedModels,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  models: readonly ComponentModelConfig[];
  parameters: Record<string, unknown>;
  supportedModels: readonly GenerateModelConfig[];
  vaeModels: readonly VaeModelConfig[];
}): RecallParametersResult => {
  const fields = new Set<RecallParameterField>();
  const skipped: RecallParameterSkip[] = [];
  const skip = (key: string, reason: RecallParameterSkipReason, detail?: string) =>
    skipped.push(detail === undefined ? { key, reason } : { detail, key, reason });
  const has = (key: string) => key in parameters;
  const isReset = (key: string) => has(key) && parameters[key] === null;
  const append = parameters.append === true;
  let values: GenerateWidgetValues = cloneGenerateWidgetValues(currentValues);

  for (const key of Object.keys(parameters)) {
    if (HANDLED_KEYS.has(key) || parameters[key] === null) {
      continue;
    }
    if (Array.isArray(parameters[key]) && parameters[key].length === 0) {
      continue;
    }
    skip(key, 'unsupported');
  }

  if (typeof parameters.model === 'string') {
    const model = findModel(supportedModels, parameters.model);

    if (!model) {
      skip('model', 'unresolved', parameters.model);
    } else if (model.key !== values.model.key) {
      // Apply model defaults before the selector's compatibility transition for references, VAE, and components.
      const { settings } = getGenerateModelSelectionResult({
        currentValues: getSettingsWithModelDefaults(values, model),
        model,
        models,
      });
      values = {
        ...values,
        ...settings,
        model,
        ...(hasModelDefaultVae(model) ? { vae: getModelDefaultVae(model, vaeModels) } : {}),
      };
      fields.add('model');
    }
  }

  const defaults = getDefaultGenerateSettings(values.model);

  if (typeof parameters.vae_model === 'string') {
    const vae = findModel(vaeModels, parameters.vae_model);

    if (!vae) {
      skip('vae_model', 'unresolved', parameters.vae_model);
    } else if (!isVaeCompatibleWithGenerateModel(values.model, vae)) {
      skip('vae_model', 'incompatible', vae.name);
    } else {
      values = { ...values, vae };
      fields.add('vae');
    }
  } else if (isReset('vae_model')) {
    values = { ...values, vae: hasModelDefaultVae(values.model) ? getModelDefaultVae(values.model, vaeModels) : null };
    fields.add('vae');
  }

  const applyField = (
    key: string,
    field: RecallParameterField,
    read: () => Partial<GenerateWidgetValues> | null,
    reset: () => Partial<GenerateWidgetValues>
  ) => {
    if (!has(key)) {
      return;
    }

    const patch = parameters[key] === null ? reset() : read();

    if (patch) {
      values = { ...values, ...patch };
      fields.add(field);
    } else {
      skip(key, 'invalid');
    }
  };

  applyField(
    'positive_prompt',
    'prompts',
    () => (typeof parameters.positive_prompt === 'string' ? { positivePrompt: parameters.positive_prompt } : null),
    () => ({ positivePrompt: '' })
  );
  applyField(
    'negative_prompt',
    'prompts',
    () => (typeof parameters.negative_prompt === 'string' ? { negativePrompt: parameters.negative_prompt } : null),
    () => ({ negativePrompt: '' })
  );
  applyField(
    'seed',
    'seed',
    () => {
      const seed = getSeed(parameters);
      return seed === null ? null : { seed, seedMode: 'fixed' };
    },
    () => ({ seedMode: 'random' })
  );
  applyField(
    'steps',
    'steps',
    () => {
      const steps = getSteps(parameters);
      return steps === null ? null : { steps };
    },
    () => ({ steps: defaults.steps })
  );
  applyField(
    'cfg_scale',
    'cfg',
    () => {
      const cfgScale = getCfgScale(parameters);
      return cfgScale === null ? null : { cfgScale };
    },
    () => ({ cfgScale: defaults.cfgScale })
  );
  applyField(
    'cfg_rescale_multiplier',
    'cfg',
    () => {
      const cfgRescaleMultiplier = getCfgRescaleMultiplier(parameters);
      return cfgRescaleMultiplier === null ? null : { cfgRescaleMultiplier };
    },
    () => ({ cfgRescaleMultiplier: defaults.cfgRescaleMultiplier })
  );
  applyField(
    'scheduler',
    'scheduler',
    () => {
      const scheduler = getScheduler(parameters);
      return scheduler === null ? null : { scheduler };
    },
    () => ({ scheduler: defaults.scheduler })
  );
  if (parameters.clip_skip === null) {
    values = { ...values, clipSkip: defaults.clipSkip };
    fields.add('clipSkip');
  } else if (has('clip_skip')) {
    const clipSkip = getSupportedClipSkip(parameters, values.model);

    if (getGenerationUiPolicy(values.model, { cfgScale: 1 }).clipSkipMax === null) {
      skip('clip_skip', 'incompatible', values.model.name);
    } else if (clipSkip === null) {
      skip('clip_skip', 'invalid');
    } else {
      values = { ...values, clipSkip };
      fields.add('clipSkip');
    }
  }
  applyField(
    'seamless_x',
    'seamless',
    () => {
      const seamlessXAxis = getBoolean(parameters, 'seamless_x');
      return seamlessXAxis === null ? null : { seamlessXAxis };
    },
    () => ({ seamlessXAxis: false })
  );
  applyField(
    'seamless_y',
    'seamless',
    () => {
      const seamlessYAxis = getBoolean(parameters, 'seamless_y');
      return seamlessYAxis === null ? null : { seamlessYAxis };
    },
    () => ({ seamlessYAxis: false })
  );

  if (has('guidance') && parameters.guidance !== null) {
    // Only meaningful for graphs that wire cfgScale to `guidance`; an explicit
    // cfg_scale in the same request wins because it names the field directly.
    if (!GUIDANCE_BASES.has(values.model.base)) {
      skip('guidance', 'incompatible', values.model.name);
    } else if (getCfgScale(parameters) !== null) {
      skip('guidance', 'unsupported', 'cfg_scale was also provided');
    } else if (isFiniteNumber(parameters.guidance) && parameters.guidance >= 1) {
      values = { ...values, cfgScale: parameters.guidance };
      fields.add('cfg');
    } else {
      skip('guidance', 'invalid');
    }
  }

  const size = getMetadataSize(parameters, values.model);
  const resetSize = {
    ...(isReset('width') ? { width: defaults.width } : {}),
    ...(isReset('height') ? { height: defaults.height } : {}),
  };

  if (typeof parameters.width === 'number' && size.width === undefined) {
    skip('width', 'invalid');
  }
  if (typeof parameters.height === 'number' && size.height === undefined) {
    skip('height', 'invalid');
  }
  if (Object.keys(size).length > 0 || Object.keys(resetSize).length > 0) {
    values = withDimensions(values, { ...size, ...resetSize });
    fields.add('size');
  }

  if (Array.isArray(parameters.loras)) {
    const loras: GenerateLora[] = [];

    for (const entry of parameters.loras) {
      if (!isRecord(entry) || typeof entry.model_key !== 'string') {
        skip('loras', 'invalid');
        continue;
      }

      const model = models.find((candidate) => candidate.key === entry.model_key);

      if (!isLoraModelConfig(model)) {
        skip('loras', 'unresolved', entry.model_key);
        continue;
      }
      if (!isLoraCompatibleWithModel(model, values.model)) {
        skip('loras', 'incompatible', model.name);
        continue;
      }
      if (loras.some((lora) => lora.model.key === model.key)) {
        continue;
      }

      loras.push({
        isEnabled: entry.is_enabled !== false,
        model,
        weight: isFiniteNumber(entry.weight)
          ? clamp(entry.weight, DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin, DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax)
          : getDefaultLoraWeight(model),
      });
    }

    values = { ...values, loras };
    fields.add('loras');
  } else if (isReset('loras')) {
    values = { ...values, loras: [] };
    fields.add('loras');
  }

  const ipAdapters = Array.isArray(parameters.ip_adapters)
    ? parameters.ip_adapters
    : isReset('ip_adapters')
      ? []
      : null;
  const referenceImages = Array.isArray(parameters.reference_images)
    ? parameters.reference_images
    : isReset('reference_images')
      ? []
      : null;

  const retainedReferenceImages = values.referenceImages;

  if (ipAdapters !== null || referenceImages !== null) {
    const recalled: GenerateReferenceImage[] = [];

    for (const entry of ipAdapters ?? []) {
      if (!isRecord(entry) || typeof entry.model_key !== 'string') {
        skip('ip_adapters', 'invalid');
        continue;
      }

      const model = models.find((candidate) => candidate.key === entry.model_key);
      const image = toReferenceAsset(entry.image);

      if (model?.type !== 'flux_redux' && model?.type !== 'ip_adapter') {
        skip('ip_adapters', 'unresolved', entry.model_key);
      } else if (!image) {
        // The backend omits `image` when the file could not be loaded.
        skip('ip_adapters', 'unresolved', `${entry.model_key}: image missing`);
      } else if (model.type === 'flux_redux') {
        recalled.push(
          toReferenceImage({
            image,
            imageInfluence: FLUX_REDUX_INFLUENCES.has(entry.image_influence as FluxReduxImageInfluence)
              ? (entry.image_influence as FluxReduxImageInfluence)
              : 'highest',
            model,
            type: 'flux_redux',
          })
        );
      } else if (model?.type === 'ip_adapter') {
        recalled.push(
          toReferenceImage({
            beginEndStepPct: [
              isFiniteNumber(entry.begin_step_percent) ? clamp(entry.begin_step_percent, 0, 1) : 0,
              isFiniteNumber(entry.end_step_percent) ? clamp(entry.end_step_percent, 0, 1) : 1,
            ],
            clipVisionModel: model.base === 'flux' ? 'ViT-L' : 'ViT-H',
            image,
            method: IP_ADAPTER_METHODS.has(entry.method as IPAdapterMethod)
              ? (entry.method as IPAdapterMethod)
              : 'full',
            model,
            type: 'ip_adapter',
            weight: isFiniteNumber(entry.weight) ? entry.weight : 1,
          })
        );
      }
    }

    for (const entry of referenceImages ?? []) {
      const image = isRecord(entry) ? toReferenceAsset(entry.image) : null;

      if (!image) {
        skip('reference_images', 'invalid');
        continue;
      }

      recalled.push(toReferenceImage(getDefaultReferenceImageConfig(values.model, models, image)));
    }

    if (recalled.length > 0 && !isReferenceImageSupported(values.model)) {
      if (ipAdapters?.length) {
        skip('ip_adapters', 'incompatible', values.model.name);
      }
      if (referenceImages?.length) {
        skip('reference_images', 'incompatible', values.model.name);
      }
    }

    const compatible = getCompatibleReferenceImages(recalled, values.model, models);
    let next = append ? [...values.referenceImages, ...compatible] : compatible;
    const limit = getMaxReferenceImages(values.model);

    if (next.length > limit) {
      skip('reference_images', 'limit', String(limit));
      next = next.slice(0, limit);
    }

    if (!append || compatible.length > 0) {
      values = { ...values, referenceImages: next };
      fields.add('referenceImages');
    }
  }

  return { fields: [...fields], retainedReferenceImages, skipped, values };
};

const SKIP_REASON_LABELS: Record<RecallParameterSkipReason, string> = {
  incompatible: 'incompatible with the current model',
  invalid: 'invalid value',
  limit: 'over the reference image limit',
  unresolved: 'not found',
  unsupported: 'not supported by the Generate panel',
};

/** One line naming every parameter that was not applied and why, for the notice. */
export const getRecallParametersSkipMessage = (skipped: readonly RecallParameterSkip[]): string =>
  skipped
    .map(({ detail, key, reason }) => `${key}: ${SKIP_REASON_LABELS[reason]}${detail ? ` (${detail})` : ''}`)
    .join('; ');

export const getRecallParametersMessage = (fields: readonly RecallParameterField[]): string =>
  `${fields.length} field${fields.length === 1 ? '' : 's'} applied to Generate.`;
