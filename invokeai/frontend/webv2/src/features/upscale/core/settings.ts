import type {
  ComponentModelConfig,
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaePrecision,
} from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import {
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeCompatibleWithGenerateModel,
  isVaeModelConfig,
  MAX_NEGATIVE_PROMPT_HEIGHT_PX,
  MAX_POSITIVE_PROMPT_HEIGHT_PX,
  MIN_NEGATIVE_PROMPT_HEIGHT_PX,
  MIN_POSITIVE_PROMPT_HEIGHT_PX,
  sanitizeBatchCount,
} from '@features/generation/settings';
import { isSeedMode, SEED_MAX } from '@platform/core/seed';

import type { SpandrelModelConfig, TileControlNetModelConfig, UpscaleWidgetValues } from './types';

/**
 * Which main-model architectures the Upscale widget can drive, and what each one needs.
 *
 * This is the single source of truth: the graph compiler keys its builders off the same record
 * (`UPSCALE_PATH_BUILDERS`, enforced by `satisfies`), validation reads the requirements from here,
 * and the widget shows a control only where the architecture has something for it to steer. Adding
 * an architecture should mean adding one entry and one builder, not hunting for a base string.
 */
export interface UpscaleArchitecture {
  /**
   * A tile ControlNet anchors the second pass to the upscaled frame. Where one exists the widget
   * requires it and the Structure slider drives its weights; where it does not, Structure has
   * nothing to scale and is hidden rather than left inert.
   */
  readonly usesTileControlNet: boolean;
  /**
   * Whether the pipeline's parts must be picked separately. FLUX transformers ship as bare
   * checkpoints far more often than not, and `flux_model_loader` refuses those without an explicit
   * T5 encoder, CLIP Embed *and* VAE -- only a complete SDNQ pipeline install can supply its own.
   */
  readonly needsExplicitComponents: boolean;
  /**
   * The size granularity the architecture's denoise node demands. Spandrel's autoscale rounds to a
   * multiple of 8, which `flux_denoise` (multiple_of=16) rejects outright — after the upscale has
   * already run. Where this is coarser than 8 the compiler resizes the frame down to it and passes
   * the size as a literal, so the graph cannot be built with a size the backend will refuse.
   */
  readonly denoiseGrid: number;
  /** Guidance-distilled architectures ignore a negative prompt; showing one invites a silent no-op. */
  readonly usesNegativePrompt: boolean;
  /** Shown in validation messages, so a user reads "FLUX.1" rather than "flux". */
  readonly label: string;
}

export const UPSCALE_ARCHITECTURES = {
  'sd-1': {
    denoiseGrid: 8,
    label: 'SD1.5',
    needsExplicitComponents: false,
    usesNegativePrompt: true,
    usesTileControlNet: true,
  },
  sdxl: {
    denoiseGrid: 8,
    label: 'SDXL',
    needsExplicitComponents: false,
    usesNegativePrompt: true,
    usesTileControlNet: true,
  },
  flux: {
    denoiseGrid: 16,
    label: 'FLUX.1',
    needsExplicitComponents: true,
    usesNegativePrompt: false,
    usesTileControlNet: false,
  },
} as const satisfies Record<string, UpscaleArchitecture>;

export type UpscaleBase = keyof typeof UPSCALE_ARCHITECTURES;

export const isUpscaleBase = (base: unknown): base is UpscaleBase =>
  typeof base === 'string' && base in UPSCALE_ARCHITECTURES;

export const upscaleArchitectureFor = (model: MainModelConfig | null): UpscaleArchitecture | null =>
  model && isUpscaleBase(model.base) ? UPSCALE_ARCHITECTURES[model.base] : null;

/** "SD1.5, SDXL or FLUX.1" — built from the table so the copy cannot go stale. */
export const supportedUpscaleArchitectureLabels = (): string => {
  const labels = Object.values(UPSCALE_ARCHITECTURES).map((architecture) => architecture.label);
  const last = labels[labels.length - 1];

  return labels.length > 1 ? `${labels.slice(0, -1).join(', ')} or ${last}` : (last ?? '');
};

/**
 * A self-contained SDNQ pipeline carries its own transformer, CLIP, T5 and VAE, so asking the user
 * to pick them would be busywork. Every other FLUX format must have them chosen explicitly.
 *
 * The backend is stricter still -- it also checks the pipeline actually contains those submodels
 * (`is_self_contained_sdnq_flux1_pipeline`), which the model list does not expose. An SDNQ install
 * missing a submodel therefore passes here and is refused by the loader with a precise message,
 * which is the right way round.
 */
export const needsExplicitComponents = (model: MainModelConfig | null): boolean =>
  (upscaleArchitectureFor(model)?.needsExplicitComponents ?? false) && model?.format !== 'sdnq_quantized';

/** What Spandrel's autoscale produces: the scaled dimension floored to a multiple of 8. */
export const spandrelAutoscaleDimension = (dimension: number, scale: number): number =>
  Math.floor(Math.trunc(dimension * scale) / 8) * 8;

/** The same dimension floored to what the architecture's denoise node will accept. */
export const upscaleDenoiseDimension = (dimension: number, scale: number, grid: number): number =>
  Math.max(grid, Math.floor(spandrelAutoscaleDimension(dimension, scale) / grid) * grid);

export const UPSCALE_SCALE_SLIDER_MIN = 2;
export const UPSCALE_SCALE_SLIDER_MAX = 8;
export const UPSCALE_SCALE_MIN = 1;
export const UPSCALE_SCALE_MAX = 16;
export const UPSCALE_CREATIVITY_MIN = -10;
export const UPSCALE_CREATIVITY_MAX = 10;
export const UPSCALE_STRUCTURE_MIN = -10;
export const UPSCALE_STRUCTURE_MAX = 10;
export const UPSCALE_TILE_SIZE_MIN = 512;
export const UPSCALE_TILE_SIZE_MAX = 1536;
export const UPSCALE_TILE_OVERLAP_MIN = 16;
export const UPSCALE_TILE_OVERLAP_MAX = 512;

export const UPSCALE_PRESETS = {
  conservative: { creativity: -5, structure: 5 },
  balanced: { creativity: 0, structure: 0 },
  creative: { creativity: 5, structure: -2 },
  artistic: { creativity: 8, structure: -5 },
} as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const normalizePromptHeight = (value: unknown, min: number, max: number, fallback: number): number =>
  isFiniteNumber(value) ? Math.min(max, Math.max(min, value)) : fallback;

export const isSupportedUpscaleMainModel = (value: unknown): value is MainModelConfig =>
  isMainModelConfig(value) && isUpscaleBase(value.base);

const isComponentModelOfType = (value: unknown, type: string): value is ComponentModelConfig =>
  isRecord(value) && value.type === type && isModelIdentifierConfig(value);

const pickComponent = (
  stored: unknown,
  models: readonly ModelConfig[],
  type: string,
  required: boolean
): ComponentModelConfig | null => {
  if (isComponentModelOfType(stored, type)) {
    return stored;
  }

  // Only auto-pick where the architecture cannot run without one; elsewhere a stale selection is
  // simply dropped rather than replaced with an arbitrary encoder.
  return required
    ? ((models.find((model) => isComponentModelOfType(model, type)) as ComponentModelConfig) ?? null)
    : null;
};

const pickCompatibleVae = (
  stored: unknown,
  models: readonly ModelConfig[],
  model: MainModelConfig | null
): UpscaleWidgetValues['vae'] => {
  if (stored && isVaeModelConfig(stored) && model && isVaeCompatibleWithGenerateModel(model, stored)) {
    return stored;
  }

  return model
    ? ((models.find(
        (candidate) => isVaeModelConfig(candidate) && isVaeCompatibleWithGenerateModel(model, candidate)
      ) as UpscaleWidgetValues['vae']) ?? null)
    : null;
};

export const isSpandrelModelConfig = (value: unknown): value is SpandrelModelConfig =>
  isModelIdentifierConfig(value) && value.type === 'spandrel_image_to_image';

export const isTileControlNetModelConfig = (value: unknown): value is TileControlNetModelConfig =>
  isModelIdentifierConfig(value) && value.type === 'controlnet';

export const isTileControlNetCandidate = (
  value: unknown,
  model: Pick<MainModelConfig, 'base'> | null
): value is TileControlNetModelConfig => {
  if (!model || !isTileControlNetModelConfig(value) || value.base !== model.base) {
    return false;
  }

  const name = value.name.toLowerCase();

  return name.includes('tile') || name.includes('union');
};

export const isUpscaleImage = (value: unknown): value is ImageWithDims =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  value.image_name.length > 0 &&
  isFiniteNumber(value.width) &&
  value.width > 0 &&
  isFiniteNumber(value.height) &&
  value.height > 0;

const isVaePrecision = (value: unknown): value is VaePrecision => value === 'fp16' || value === 'fp32';

const normalizeLoras = (value: unknown): GenerateLora[] =>
  Array.isArray(value)
    ? value.filter(
        (item): item is GenerateLora =>
          isRecord(item) &&
          isLoraModelConfig(item.model) &&
          typeof item.isEnabled === 'boolean' &&
          isFiniteNumber(item.weight)
      )
    : [];

export const createDefaultUpscaleWidgetValues = (models: readonly ModelConfig[] = []): UpscaleWidgetValues => {
  const model = (models.find(isSupportedUpscaleMainModel) as MainModelConfig | undefined) ?? null;
  const needsComponents = needsExplicitComponents(model);

  return {
    batchCount: 1,
    cfgScale: 2,
    clipEmbedModel: pickComponent(null, models, 'clip_embed', needsComponents),
    clipSkip: 0,
    creativity: 0,
    inputImage: null,
    loras: [],
    model,
    negativePrompt: '',
    negativePromptEnabled: true,
    negativePromptHeightPx: DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
    positivePrompt: '',
    positivePromptHeightPx: DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
    scale: 4,
    scheduler: 'kdpm_2',
    seed: 0,
    seedMode: 'random',
    steps: 30,
    structure: 0,
    t5EncoderModel: pickComponent(null, models, 't5_encoder', needsComponents),
    tileControlnetModel: models.find((candidate) => isTileControlNetCandidate(candidate, model)) ?? null,
    tileOverlap: 128,
    tileSize: 1024,
    upscaleModel: (models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null,
    vae: needsComponents ? pickCompatibleVae(null, models, model) : null,
    vaePrecision: 'fp32',
  };
};

/**
 * Heal partial persisted values without clamping invalid user input; invocation validation supplies actionable
 * range errors.
 */
export const normalizeUpscaleWidgetValues = (value: unknown): UpscaleWidgetValues | null => {
  if (!isRecord(value)) {
    return null;
  }

  const defaults = createDefaultUpscaleWidgetValues();

  return {
    batchCount: sanitizeBatchCount(value.batchCount),
    cfgScale: isFiniteNumber(value.cfgScale) ? value.cfgScale : defaults.cfgScale,
    clipEmbedModel: isComponentModelOfType(value.clipEmbedModel, 'clip_embed') ? value.clipEmbedModel : null,
    clipSkip: isFiniteNumber(value.clipSkip) ? value.clipSkip : defaults.clipSkip,
    creativity: isFiniteNumber(value.creativity) ? value.creativity : defaults.creativity,
    inputImage:
      value.inputImage === null || value.inputImage === undefined
        ? null
        : isUpscaleImage(value.inputImage)
          ? value.inputImage
          : null,
    loras: normalizeLoras(value.loras),
    model: isMainModelConfig(value.model) ? value.model : null,
    negativePrompt: typeof value.negativePrompt === 'string' ? value.negativePrompt : defaults.negativePrompt,
    negativePromptEnabled:
      typeof value.negativePromptEnabled === 'boolean' ? value.negativePromptEnabled : defaults.negativePromptEnabled,
    negativePromptHeightPx: normalizePromptHeight(
      value.negativePromptHeightPx,
      MIN_NEGATIVE_PROMPT_HEIGHT_PX,
      MAX_NEGATIVE_PROMPT_HEIGHT_PX,
      defaults.negativePromptHeightPx
    ),
    positivePrompt: typeof value.positivePrompt === 'string' ? value.positivePrompt : defaults.positivePrompt,
    positivePromptHeightPx: normalizePromptHeight(
      value.positivePromptHeightPx,
      MIN_POSITIVE_PROMPT_HEIGHT_PX,
      MAX_POSITIVE_PROMPT_HEIGHT_PX,
      defaults.positivePromptHeightPx
    ),
    scale: isFiniteNumber(value.scale) ? value.scale : defaults.scale,
    scheduler: typeof value.scheduler === 'string' ? value.scheduler : defaults.scheduler,
    seed: isFiniteNumber(value.seed) ? value.seed : defaults.seed,
    // Values saved before seed modes carry the random toggle instead.
    seedMode: isSeedMode(value.seedMode)
      ? value.seedMode
      : typeof value.shouldRandomizeSeed === 'boolean'
        ? value.shouldRandomizeSeed
          ? 'random'
          : 'fixed'
        : defaults.seedMode,
    steps: isFiniteNumber(value.steps) ? value.steps : defaults.steps,
    structure: isFiniteNumber(value.structure) ? value.structure : defaults.structure,
    t5EncoderModel: isComponentModelOfType(value.t5EncoderModel, 't5_encoder') ? value.t5EncoderModel : null,
    tileControlnetModel: isTileControlNetModelConfig(value.tileControlnetModel) ? value.tileControlnetModel : null,
    tileOverlap: isFiniteNumber(value.tileOverlap) ? value.tileOverlap : defaults.tileOverlap,
    tileSize: isFiniteNumber(value.tileSize) ? value.tileSize : defaults.tileSize,
    upscaleModel: isSpandrelModelConfig(value.upscaleModel) ? value.upscaleModel : null,
    vae: isVaeModelConfig(value.vae) ? value.vae : null,
    vaePrecision: isVaePrecision(value.vaePrecision) ? value.vaePrecision : defaults.vaePrecision,
  };
};

export const syncUpscaleWidgetValuesWithModels = (
  values: UpscaleWidgetValues,
  models: readonly ModelConfig[]
): UpscaleWidgetValues => {
  const modelsByKey = new Map(models.map((model) => [model.key, model]));
  const storedMain = values.model ? modelsByKey.get(values.model.key) : undefined;
  const model: MainModelConfig | null = isSupportedUpscaleMainModel(storedMain)
    ? storedMain
    : ((models.find(isSupportedUpscaleMainModel) as MainModelConfig | undefined) ?? null);
  const storedSpandrel = values.upscaleModel ? modelsByKey.get(values.upscaleModel.key) : undefined;
  const upscaleModel: SpandrelModelConfig | null = isSpandrelModelConfig(storedSpandrel)
    ? storedSpandrel
    : ((models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null);
  const storedControlNet = values.tileControlnetModel ? modelsByKey.get(values.tileControlnetModel.key) : undefined;
  const tileControlnetModel = isTileControlNetCandidate(storedControlNet, model)
    ? storedControlNet
    : (models.find((candidate) => isTileControlNetCandidate(candidate, model)) ?? null);
  const needsComponents = needsExplicitComponents(model);
  const t5EncoderModel = pickComponent(
    values.t5EncoderModel ? modelsByKey.get(values.t5EncoderModel.key) : undefined,
    models,
    't5_encoder',
    needsComponents
  );
  const clipEmbedModel = pickComponent(
    values.clipEmbedModel ? modelsByKey.get(values.clipEmbedModel.key) : undefined,
    models,
    'clip_embed',
    needsComponents
  );
  const storedVae = values.vae ? modelsByKey.get(values.vae.key) : undefined;
  const vae =
    storedVae && isVaeModelConfig(storedVae) && model && isVaeCompatibleWithGenerateModel(model, storedVae)
      ? storedVae
      : needsComponents
        ? pickCompatibleVae(null, models, model)
        : null;
  const loras = model
    ? values.loras.flatMap((lora) => {
        const installed = modelsByKey.get(lora.model.key);

        return installed && isLoraModelConfig(installed) && isLoraCompatibleWithModel(installed, model)
          ? [{ ...lora, model: installed }]
          : [];
      })
    : [];

  if (
    values.model === model &&
    values.upscaleModel === upscaleModel &&
    values.tileControlnetModel === tileControlnetModel &&
    values.t5EncoderModel === t5EncoderModel &&
    values.clipEmbedModel === clipEmbedModel &&
    values.vae === vae &&
    loras.length === values.loras.length &&
    loras.every((lora, index) => lora.model === values.loras[index]?.model)
  ) {
    return values;
  }

  return { ...values, clipEmbedModel, loras, model, t5EncoderModel, tileControlnetModel, upscaleModel, vae };
};

const addRangeReason = (reasons: string[], label: string, value: number, min: number, max: number): void => {
  if (!Number.isFinite(value) || value < min || value > max) {
    reasons.push(`${label} must be between ${min} and ${max}.`);
  }
};

export const getUpscaleValidationReasons = (values: UpscaleWidgetValues, models?: readonly ModelConfig[]): string[] => {
  const reasons: string[] = [];

  if (!values.inputImage) {
    reasons.push('Upscale needs an input image. Upload one or send one from Gallery.');
  }
  if (!values.upscaleModel) {
    reasons.push('Upscale needs a Spandrel image-to-image model.');
  }
  const architecture = upscaleArchitectureFor(values.model);

  if (!values.model) {
    reasons.push(`Upscale needs a ${supportedUpscaleArchitectureLabels()} main model.`);
  } else if (!architecture) {
    reasons.push(`Upscale supports only ${supportedUpscaleArchitectureLabels()} main models.`);
  }

  // `flux_denoise` refuses a Fill model without fill conditioning, which an upscale never supplies.
  if (values.model?.base === 'flux' && values.model.variant === 'dev_fill') {
    reasons.push('Upscale cannot use a FLUX Fill model.');
  }

  // Only asked for where the architecture has a tile ControlNet to anchor the second pass with.
  if (architecture?.usesTileControlNet) {
    if (!values.tileControlnetModel) {
      reasons.push('Upscale needs a Tile or Union ControlNet compatible with the main model.');
    } else if (!isTileControlNetCandidate(values.tileControlnetModel, values.model)) {
      reasons.push('The Tile ControlNet must match the main model base and be a Tile or Union model.');
    }
  }

  // A bare FLUX transformer cannot supply these itself, and `flux_model_loader` refuses without them.
  if (needsExplicitComponents(values.model)) {
    if (!values.t5EncoderModel) {
      reasons.push('Upscale needs a T5 encoder for this main model.');
    }
    if (!values.clipEmbedModel) {
      reasons.push('Upscale needs a CLIP Embed model for this main model.');
    }
    if (!values.vae) {
      reasons.push('Upscale needs a VAE for this main model.');
    }
  }

  addRangeReason(reasons, 'Scale', values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX);
  addRangeReason(reasons, 'Creativity', values.creativity, UPSCALE_CREATIVITY_MIN, UPSCALE_CREATIVITY_MAX);
  addRangeReason(reasons, 'Structure', values.structure, UPSCALE_STRUCTURE_MIN, UPSCALE_STRUCTURE_MAX);
  addRangeReason(reasons, 'Tile size', values.tileSize, UPSCALE_TILE_SIZE_MIN, UPSCALE_TILE_SIZE_MAX);
  addRangeReason(reasons, 'Tile overlap', values.tileOverlap, UPSCALE_TILE_OVERLAP_MIN, UPSCALE_TILE_OVERLAP_MAX);
  addRangeReason(reasons, 'Steps', values.steps, 1, 1000);
  addRangeReason(reasons, 'CFG scale', values.cfgScale, 1, 100);
  addRangeReason(reasons, 'Seed', values.seed, 0, SEED_MAX);

  if (models) {
    const installedByKey = new Map(models.map((model) => [model.key, model]));
    const required: ModelIdentifierConfig[] = [
      ...(values.model ? [values.model] : []),
      ...(values.upscaleModel ? [values.upscaleModel] : []),
      ...(values.tileControlnetModel ? [values.tileControlnetModel] : []),
      // Required for FLUX, so a deleted encoder must block the run rather than fail mid-graph.
      ...(values.t5EncoderModel ? [values.t5EncoderModel] : []),
      ...(values.clipEmbedModel ? [values.clipEmbedModel] : []),
      ...(values.vae ? [values.vae] : []),
    ];

    for (const selected of required) {
      const installed = installedByKey.get(selected.key);

      if (!installed) {
        reasons.push(`${selected.name} is no longer installed.`);
      }
    }

    const installedMain = values.model ? installedByKey.get(values.model.key) : undefined;
    const installedSpandrel = values.upscaleModel ? installedByKey.get(values.upscaleModel.key) : undefined;
    const installedControlNet = values.tileControlnetModel
      ? installedByKey.get(values.tileControlnetModel.key)
      : undefined;

    if (values.model && installedMain && !isSupportedUpscaleMainModel(installedMain)) {
      reasons.push(`${values.model.name} is not an installed SD1.5 or SDXL main model.`);
    }
    if (values.upscaleModel && installedSpandrel && !isSpandrelModelConfig(installedSpandrel)) {
      reasons.push(`${values.upscaleModel.name} is not an installed Spandrel model.`);
    }
    if (
      values.tileControlnetModel &&
      installedControlNet &&
      !isTileControlNetCandidate(installedControlNet, values.model)
    ) {
      reasons.push(`${values.tileControlnetModel.name} is not an installed compatible Tile or Union ControlNet.`);
    }
    if (values.vae) {
      const installedVae = installedByKey.get(values.vae.key);

      if (!installedVae) {
        reasons.push(`${values.vae.name} is no longer installed.`);
      } else if (
        !values.model ||
        !isVaeModelConfig(installedVae) ||
        !isVaeCompatibleWithGenerateModel(values.model, installedVae)
      ) {
        reasons.push(`${values.vae.name} is not compatible with ${values.model?.name ?? 'the main model'}.`);
      }
    }
    for (const lora of values.loras) {
      const installedLora = installedByKey.get(lora.model.key);

      if (!installedLora) {
        reasons.push(`${lora.model.name} is no longer installed.`);
      } else if (
        !isLoraModelConfig(installedLora) ||
        (values.model && !isLoraCompatibleWithModel(installedLora, values.model))
      ) {
        reasons.push(`${lora.model.name} is not compatible with ${values.model?.name ?? 'the main model'}.`);
      }
    }
  }

  return reasons;
};

export const getUpscaleOutputDimensions = (
  image: Pick<ImageWithDims, 'width' | 'height'>,
  scale: number
): { width: number; height: number } => ({
  height: Math.floor((image.height * scale) / 8) * 8,
  width: Math.floor((image.width * scale) / 8) * 8,
});

export const clearDeletedUpscaleInput = (
  values: UpscaleWidgetValues,
  deletedImageNames: ReadonlySet<string>
): UpscaleWidgetValues =>
  values.inputImage && deletedImageNames.has(values.inputImage.image_name) ? { ...values, inputImage: null } : values;

export const resolveUpscaleSeed = (values: UpscaleWidgetValues): number =>
  values.seedMode === 'random' ? Math.floor(Math.random() * SEED_MAX) : values.seed;

export const cloneUpscaleWidgetValues = (values: UpscaleWidgetValues): UpscaleWidgetValues => structuredClone(values);
