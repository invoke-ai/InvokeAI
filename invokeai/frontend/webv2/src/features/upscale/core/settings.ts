import type {
  ComponentModelConfig,
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaePrecision,
} from '@features/generation/contracts';
import type { GenerateSubmissionSource, ProjectPromptDraft } from '@features/generation/settings';
import type { ModelConfig } from '@features/models';

import {
  coerceSchedulerForGraph,
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  getDefaultGenerateSettings,
  getGenerationUiPolicy,
  getSchedulerOptions,
  hasArchitectureCapabilities,
  isAnimaQwen35Encoder,
  isAnimaQwen3Encoder,
  isBundledFlux2Main,
  isBundledMainForBase,
  isFlux2MistralEncoder,
  isFlux2Qwen3EncoderForModel,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isMossImageQwen35Encoder,
  isModelIdentifierConfig,
  isQwen3Vl4bEncoder,
  isVaeCompatibleWithGenerateModel,
  isVaeModelConfig,
  isZImageQwen3Encoder,
  MAX_NEGATIVE_PROMPT_HEIGHT_PX,
  MAX_POSITIVE_PROMPT_HEIGHT_PX,
  MIN_NEGATIVE_PROMPT_HEIGHT_PX,
  MIN_POSITIVE_PROMPT_HEIGHT_PX,
  sanitizeBatchCount,
  servesAsType,
} from '@features/generation/settings';
import { isSeedMode, SEED_MAX } from '@platform/core/seed';

import type { SpandrelModelConfig, TileControlNetModelConfig, UpscaleMode, UpscaleWidgetValues } from './types';

// Schedules of nodes that clip `denoising_start` by step index (`indexClipping`); drop once they clip by value.

/** First `steps` sigmas of a flow schedule shifted by `shift`, without the final 0 (diffusers: shift = exp(mu)). */
const shiftedSigmas = (shift: number, steps: number): number[] =>
  Array.from({ length: steps }, (_, index) => {
    const t = 1 - index / steps;

    return (shift * t) / (1 + (shift - 1) * t);
  });

/** `z_image_denoise.Z_IMAGE_SHIFT_FIT`, held at the fit's 4096-token end since the clamp (#378). */
const Z_IMAGE_SHIFT_MIN_SEQ_LEN = 256;
const Z_IMAGE_SHIFT_MAX_SEQ_LEN = 4096;
const Z_IMAGE_BASE_SHIFT = 0.5;
const Z_IMAGE_MAX_SHIFT = 1.15;

/** Z-Image's linear shift for a frame: one token per 16x16 pixels, as `z_image_denoise` patches it. */
const zImageScheduleShift = ({ width, height }: { width: number; height: number }): number => {
  const seqLen = Math.min(Math.floor(width / 16) * Math.floor(height / 16), Z_IMAGE_SHIFT_MAX_SEQ_LEN);
  const slope = (Z_IMAGE_MAX_SHIFT - Z_IMAGE_BASE_SHIFT) / (Z_IMAGE_SHIFT_MAX_SEQ_LEN - Z_IMAGE_SHIFT_MIN_SEQ_LEN);

  return Math.exp(seqLen * slope + Z_IMAGE_BASE_SHIFT - slope * Z_IMAGE_SHIFT_MIN_SEQ_LEN);
};

/** `ANIMA_SHIFT`: Anima's schedule is shifted by a constant, not by resolution. */
const ANIMA_SCHEDULE_SHIFT = 3;

/** `krea2/sampling_utils.py`: Turbo's fixed mu and Raw's token-count fit; Diffusers installs ship the same config. */
const KREA2_DISTILLED_MU = 1.15;
const KREA2_SHIFT_MIN_SEQ_LEN = 256;
const KREA2_SHIFT_MAX_SEQ_LEN = 6400;
const KREA2_BASE_SHIFT = 0.5;
const KREA2_MAX_SHIFT = 1.15;

/** `exp(mu)`, held at the fit's 6400-token end as `krea2_denoise` does; anything not Raw counts as distilled. */
const krea2ScheduleShift = (model: MainModelConfig, { width, height }: { width: number; height: number }): number => {
  if (model.variant !== 'krea2_base') {
    return Math.exp(KREA2_DISTILLED_MU);
  }

  const seqLen = Math.min(Math.floor(width / 16) * Math.floor(height / 16), KREA2_SHIFT_MAX_SEQ_LEN);
  const slope = (KREA2_MAX_SHIFT - KREA2_BASE_SHIFT) / (KREA2_SHIFT_MAX_SEQ_LEN - KREA2_SHIFT_MIN_SEQ_LEN);

  return Math.exp(seqLen * slope + KREA2_BASE_SHIFT - slope * KREA2_SHIFT_MIN_SEQ_LEN);
};

/** `qwen_image_denoise`'s scheduler: mu 0.5 at 256 tokens to 0.9 at 8192, held at the end, stretched to 0.02. */
const QWEN_IMAGE_SHIFT_MIN_SEQ_LEN = 256;
const QWEN_IMAGE_SHIFT_MAX_SEQ_LEN = 8192;
const QWEN_IMAGE_BASE_SHIFT = 0.5;
const QWEN_IMAGE_MAX_SHIFT = 0.9;
const QWEN_IMAGE_SHIFT_TERMINAL = 0.02;

/** As diffusers' FlowMatchEulerDiscreteScheduler builds them, including `stretch_shift_to_terminal`. */
const qwenImageSigmas = ({ width, height }: { width: number; height: number }, steps: number): number[] => {
  const seqLen = Math.min(Math.floor(width / 16) * Math.floor(height / 16), QWEN_IMAGE_SHIFT_MAX_SEQ_LEN);
  const slope =
    (QWEN_IMAGE_MAX_SHIFT - QWEN_IMAGE_BASE_SHIFT) / (QWEN_IMAGE_SHIFT_MAX_SEQ_LEN - QWEN_IMAGE_SHIFT_MIN_SEQ_LEN);
  const shifted = shiftedSigmas(
    Math.exp(seqLen * slope + (QWEN_IMAGE_BASE_SHIFT - slope * QWEN_IMAGE_SHIFT_MIN_SEQ_LEN)),
    steps
  );
  const scale = (1 - (shifted[steps - 1] ?? 0)) / (1 - QWEN_IMAGE_SHIFT_TERMINAL);

  return shifted.map((sigma) => 1 - (1 - sigma) / scale);
};

export type UpscaleComponentField =
  | 't5EncoderModel'
  | 'clipEmbedModel'
  | 'qwen3EncoderModel'
  | 'qwen35EncoderModel'
  | 'qwen3VlEncoderModel'
  | 'qwenVlEncoderModel'
  | 'mistralEncoderModel';

const UPSCALE_COMPONENT_TYPES: Record<UpscaleComponentField, string> = {
  clipEmbedModel: 'clip_embed',
  mistralEncoderModel: 'mistral_encoder',
  qwen35EncoderModel: 'qwen3_5_encoder',
  qwen3EncoderModel: 'qwen3_encoder',
  qwen3VlEncoderModel: 'qwen3_vl_encoder',
  qwenVlEncoderModel: 'qwen_vl_encoder',
  t5EncoderModel: 't5_encoder',
};

export const UPSCALE_COMPONENT_FIELDS = Object.keys(UPSCALE_COMPONENT_TYPES) as UpscaleComponentField[];

export interface UpscaleComponentSlot {
  readonly field: UpscaleComponentField;
  /** Which installed models fit. Also the default pick, so it must be as narrow as the loader is. */
  readonly accepts: (candidate: ModelConfig) => boolean;
  /** Read in validation messages: "Upscale needs a {label} for this main model." */
  readonly label: string;
}

const T5_ENCODER_SLOT: UpscaleComponentSlot = {
  accepts: (candidate) => candidate.type === 't5_encoder',
  field: 't5EncoderModel',
  label: 'T5 encoder',
};
// FLUX.1 encodes with CLIP-L; a CLIP-G (bundled in ComfyUI text_encoders folders) fails its text encoder.
const CLIP_EMBED_SLOT: UpscaleComponentSlot = {
  accepts: (candidate) => candidate.type === 'clip_embed' && candidate.variant === 'large',
  field: 'clipEmbedModel',
  label: 'CLIP Embed model',
};
// Z-Image and Anima both read a `qwen3_encoder`, but not the same one: Z-Image takes the 4B, Anima the 0.6B.
const Z_IMAGE_QWEN3_SLOT: UpscaleComponentSlot = {
  accepts: isZImageQwen3Encoder,
  field: 'qwen3EncoderModel',
  label: 'Qwen3 4B encoder',
};
const ANIMA_QWEN3_SLOT: UpscaleComponentSlot = {
  accepts: isAnimaQwen3Encoder,
  field: 'qwen3EncoderModel',
  label: 'Qwen3 0.6B encoder',
};
const ANIMA_QWEN35_SLOT: UpscaleComponentSlot = {
  accepts: isAnimaQwen35Encoder,
  field: 'qwen35EncoderModel',
  label: 'Qwen3.5 encoder',
};
// MOSS-Image reads the Qwen3.5 9B; the 4B Anima-3.8B uses is the same model type and its hidden size does not fit.
const MOSS_IMAGE_QWEN35_SLOT: UpscaleComponentSlot = {
  accepts: isMossImageQwen35Encoder,
  field: 'qwen35EncoderModel',
  label: 'Qwen3.5 9B encoder',
};
// Krea-2 reads the 4B; Ideogram 4's 8B is the same model type and `krea2_model_loader` refuses it.
const KREA2_QWEN3_VL_SLOT: UpscaleComponentSlot = {
  accepts: isQwen3Vl4bEncoder,
  field: 'qwen3VlEncoderModel',
  label: 'Qwen3-VL 4B encoder',
};
// `flux2_klein_model_loader` refuses the other size; Generate's filter decides the fit, this only names it.
const FLUX2_KLEIN_QWEN3_LABELS: Partial<Record<string, string>> = {
  klein_4b: 'Qwen3 4B encoder',
  klein_4b_base: 'Qwen3 4B encoder',
  klein_9b: 'Qwen3 8B encoder',
  klein_9b_base: 'Qwen3 8B encoder',
};
const flux2KleinQwen3Slot = (model: MainModelConfig): UpscaleComponentSlot => ({
  accepts: isFlux2Qwen3EncoderForModel(model),
  field: 'qwen3EncoderModel',
  label: (typeof model.variant === 'string' && FLUX2_KLEIN_QWEN3_LABELS[model.variant]) || 'Qwen3 encoder',
});
// ERNIE-Image's Ministral 3B shares the model type, but [dev] was not trained on it.
const FLUX2_MISTRAL_SLOT: UpscaleComponentSlot = {
  accepts: isFlux2MistralEncoder,
  field: 'mistralEncoderModel',
  label: 'Mistral encoder',
};
const QWEN_IMAGE_VL_SLOT: UpscaleComponentSlot = {
  accepts: (candidate) => candidate.type === 'qwen_vl_encoder',
  field: 'qwenVlEncoderModel',
  label: 'Qwen VL encoder',
};

/** What a main model's loader must be handed. A model that bundles its own parts needs none of it. */
export interface UpscaleComponents {
  readonly slots: readonly UpscaleComponentSlot[];
  readonly vae: boolean;
}

const NO_COMPONENTS: UpscaleComponents = { slots: [], vae: false };

export interface UpscaleTileControlNet {
  /** SD's multi-diffusion tiles drift apart without a tile ControlNet; elsewhere it is an optional anchor. */
  readonly required: boolean;
  /** Checkpoints do not state their mode (Z-Image Tile and Union 2.1 are identical tensors), so the name decides. */
  readonly matchesName: (name: string) => boolean;
}

const nameIncludes =
  (...needles: string[]) =>
  (name: string): boolean =>
    needles.some((needle) => name.toLowerCase().includes(needle));

export interface UpscaleIndexClipping {
  /** Noisiest first, without the final 0. */
  readonly sigmas: (
    model: MainModelConfig,
    size: { width: number; height: number },
    steps: number
  ) => readonly number[];
  /** `int(start * steps)`, or Python's half-to-even `int(round(start * steps))`. */
  readonly rounding: 'floor' | 'round';
}

/** Single source of truth: the compiler's `UPSCALE_PATH_BUILDERS` and validation both key off this record. */
export interface UpscaleArchitecture {
  /** Structure drives this ControlNet's weights; without one, Structure is hidden rather than inert. */
  readonly tileControlNet: UpscaleTileControlNet | null;
  /** The parts the loader refuses to run without; only bundled pipelines carry their own. */
  readonly components: (model: MainModelConfig) => UpscaleComponents;
  /** Spandrel rounds to 8, which `flux_denoise` (multiple_of=16) rejects after the upscale; coarser grids resize. */
  readonly denoiseGrid: number;
  /** The served `FeaturesFacet` answer, pinned by test; `cfg-gated` reads it only above CFG 1. */
  readonly negativePrompt: 'always' | 'cfg-gated' | 'never';
  /** Lets the compiler start an index-clipping node where FLUX would for the same Creativity; null otherwise. */
  readonly indexClipping: UpscaleIndexClipping | null;
  /**
   * Schedulers that start where Creativity puts the pass; null where all do. LCM on FLUX.1/Z-Image/Anima and
   * Anima's Heun build their own schedules past the blend. `test_upscale_index_clipped_starts.py` checks these.
   */
  readonly schedulers: readonly string[] | null;
  /**
   * Whether a new main model takes its served CFG (distilled Turbos want 1). SD and FLUX.1 keep the user's, since
   * SD's multi-diffusion is tuned to 2, and get 2 back when coming from another architecture.
   */
  readonly takesModelGuidance: boolean;
  /** Only FLUX.2: `flux2_denoise` re-spaces the clipped window to `num_steps`, so 30 steps would run Klein 7x over. */
  readonly takesModelSteps: boolean;
  readonly label: string;
}

const SD_TILE_CONTROLNET: UpscaleTileControlNet = { matchesName: nameIncludes('tile', 'union'), required: true };

export const UPSCALE_ARCHITECTURES = {
  'sd-1': {
    components: () => NO_COMPONENTS,
    denoiseGrid: 8,
    indexClipping: null,
    label: 'SD1.5',
    negativePrompt: 'always',
    schedulers: null,
    takesModelGuidance: false,
    takesModelSteps: false,
    tileControlNet: SD_TILE_CONTROLNET,
  },
  sdxl: {
    components: () => NO_COMPONENTS,
    denoiseGrid: 8,
    indexClipping: null,
    label: 'SDXL',
    negativePrompt: 'always',
    schedulers: null,
    takesModelGuidance: false,
    takesModelSteps: false,
    tileControlNet: SD_TILE_CONTROLNET,
  },
  flux: {
    // `flux_model_loader` refuses a bare transformer without T5, CLIP Embed and VAE; the backend checks SDNQ submodels.
    components: (model) =>
      model.format === 'sdnq_quantized' ? NO_COMPONENTS : { slots: [T5_ENCODER_SLOT, CLIP_EMBED_SLOT], vae: true },
    denoiseGrid: 16,
    indexClipping: null,
    label: 'FLUX.1',
    negativePrompt: 'never',
    // `flux/denoise.py` uses `z_image_denoise`'s schedulers, so this follows the backend contract's Z-Image list.
    schedulers: ['euler', 'heun'],
    takesModelGuidance: false,
    takesModelSteps: false,
    tileControlNet: null,
  },
  flux2: {
    // Both loaders fall back to a bundled main model's own encoder and VAE.
    components: (model) =>
      isBundledFlux2Main(model)
        ? NO_COMPONENTS
        : { slots: [model.variant === 'dev' ? FLUX2_MISTRAL_SLOT : flux2KleinQwen3Slot(model)], vae: true },
    denoiseGrid: 16,
    // `flux2_denoise` clips by sigma value like FLUX.1, then re-spaces the window to the requested steps.
    indexClipping: null,
    label: 'FLUX.2',
    negativePrompt: 'never',
    // With a start above zero `flux2_denoise` steps with Euler, whatever scheduler it is handed.
    schedulers: ['euler'],
    takesModelGuidance: true,
    takesModelSteps: true,
    tileControlNet: null,
  },
  'z-image': {
    // `z_image_model_loader` falls back to a bundled main model's own VAE and Qwen3.
    components: (model) =>
      isBundledMainForBase('z-image')(model) ? NO_COMPONENTS : { slots: [Z_IMAGE_QWEN3_SLOT], vae: true },
    denoiseGrid: 16,
    indexClipping: {
      rounding: 'floor',
      sigmas: (_model, size, steps) => shiftedSigmas(zImageScheduleShift(size), steps),
    },
    label: 'Z-Image',
    negativePrompt: 'cfg-gated',
    schedulers: ['euler', 'heun'],
    takesModelGuidance: true,
    takesModelSteps: false,
    tileControlNet: { matchesName: nameIncludes('tile'), required: false },
  },
  anima: {
    // Anima-3.8B's semantic connector also reads Qwen3.5; no other Anima model does.
    components: (model) => ({
      slots: model.variant === 'anima_qwen35' ? [ANIMA_QWEN3_SLOT, ANIMA_QWEN35_SLOT] : [ANIMA_QWEN3_SLOT],
      vae: true,
    }),
    denoiseGrid: 8,
    indexClipping: { rounding: 'floor', sigmas: (_model, _size, steps) => shiftedSigmas(ANIMA_SCHEDULE_SHIFT, steps) },
    label: 'Anima',
    negativePrompt: 'cfg-gated',
    schedulers: ['euler', 'dpmpp_2m', 'dpmpp_2m_sde', 'er_sde'],
    takesModelGuidance: true,
    takesModelSteps: false,
    tileControlNet: null,
  },
  'krea-2': {
    // `krea2_model_loader` refuses a single-file or GGUF transformer without Qwen3-VL and VAE.
    components: (model) => (model.format === 'diffusers' ? NO_COMPONENTS : { slots: [KREA2_QWEN3_VL_SLOT], vae: true }),
    denoiseGrid: 16,
    indexClipping: {
      rounding: 'round',
      sigmas: (model, size, steps) => shiftedSigmas(krea2ScheduleShift(model, size), steps),
    },
    label: 'Krea-2',
    negativePrompt: 'cfg-gated',
    // `krea2_denoise` has no scheduler input: it steps with Euler.
    schedulers: ['euler'],
    takesModelGuidance: true,
    takesModelSteps: false,
    tileControlNet: null,
  },
  'moss-image': {
    // `moss_image_model_loader` takes a Diffusers install's own Qwen3.5 and VAE; any other format needs both.
    components: (model) =>
      model.format === 'diffusers' ? NO_COMPONENTS : { slots: [MOSS_IMAGE_QWEN35_SLOT], vae: true },
    denoiseGrid: 16,
    // `moss_image_denoise` clips by sigma value like FLUX, then re-spaces the window to the requested steps.
    indexClipping: null,
    label: 'MOSS-Image',
    negativePrompt: 'cfg-gated',
    // `moss_image_denoise` has no scheduler input: it steps its own shifted Euler schedule.
    schedulers: ['euler'],
    takesModelGuidance: true,
    // Not distilled: the served 50 would run the whole re-spaced window at upscale resolution, two passes each.
    takesModelSteps: false,
    tileControlNet: null,
  },
  'qwen-image': {
    // The widget sends no component source, so any non-Diffusers format needs both.
    components: (model) => (model.format === 'diffusers' ? NO_COMPONENTS : { slots: [QWEN_IMAGE_VL_SLOT], vae: true }),
    denoiseGrid: 16,
    indexClipping: { rounding: 'round', sigmas: (_model, size, steps) => qwenImageSigmas(size, steps) },
    label: 'Qwen Image',
    negativePrompt: 'cfg-gated',
    // `qwen_image_denoise` has no scheduler input: it steps with Euler.
    schedulers: ['euler'],
    takesModelGuidance: true,
    takesModelSteps: false,
    tileControlNet: null,
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

export const upscaleComponentsFor = (model: MainModelConfig | null): UpscaleComponents => {
  const architecture = upscaleArchitectureFor(model);

  return model && architecture ? architecture.components(model) : NO_COMPONENTS;
};

/** CFG-gated architectures need a negative branch above CFG 1; a switched-off negative prompt still gets one. */
export const wiresUpscaleNegativePrompt = (model: MainModelConfig | null, cfgScale: number): boolean => {
  const usage = upscaleArchitectureFor(model)?.negativePrompt;

  return usage === 'always' || (usage === 'cfg-gated' && cfgScale > 1);
};

/** Index clipping starts on a whole step, so one step means pure noise or nothing; the pass needs two. */
export const upscaleMinSteps = (model: MainModelConfig | null): number =>
  upscaleArchitectureFor(model)?.indexClipping ? 2 : 1;

const UPSCALE_CFG_SCALE_MAX = 100;

/** The widget's 100, or the node's served guidance cap (`flux2_denoise` refuses above 20 at enqueue). */
export const upscaleMaxCfgScale = (model: MainModelConfig | null): number => {
  const served = model ? getGenerationUiPolicy(model).guidanceMax : null;

  return served === null ? UPSCALE_CFG_SCALE_MAX : Math.min(UPSCALE_CFG_SCALE_MAX, served);
};

/** The base's schedulers, less those that would ignore Creativity; see `schedulers`. */
export const upscaleSchedulerOptions = (model: MainModelConfig | null) => {
  const options = getSchedulerOptions(model ?? undefined);
  const allowed = upscaleArchitectureFor(model)?.schedulers;

  return allowed ? options.filter((option) => allowed.includes(option.value)) : options;
};

/** The stored value stays so switching back to SD restores it; graph and metadata record the one that runs. */
export const coerceUpscaleScheduler = (model: MainModelConfig | null, scheduler: string): string => {
  const coerced = coerceSchedulerForGraph(model ?? undefined, scheduler);
  const allowed = upscaleArchitectureFor(model)?.schedulers;

  return !allowed || allowed.includes(coerced) ? coerced : (allowed[0] ?? coerced);
};

/** What Spandrel's autoscale produces: the scaled dimension floored to a multiple of 8. */
export const spandrelAutoscaleDimension = (dimension: number, scale: number): number =>
  Math.floor(Math.trunc(dimension * scale) / 8) * 8;

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
  isRecord(value) && isModelIdentifierConfig(value) && servesAsType(value, type);

type UpscaleComponentValues = Record<UpscaleComponentField, ComponentModelConfig | null>;

/** A field the architecture has no slot for keeps its stored encoder, so switching away and back keeps a pick. */
const pickComponents = (
  stored: Partial<Record<UpscaleComponentField, unknown>>,
  models: readonly ModelConfig[],
  model: MainModelConfig | null
): UpscaleComponentValues => {
  const slots = upscaleComponentsFor(model).slots;
  const picked = {} as UpscaleComponentValues;

  for (const field of UPSCALE_COMPONENT_FIELDS) {
    const current = stored[field];
    const slot = slots.find((candidate) => candidate.field === field);
    const fits = (candidate: unknown): candidate is ComponentModelConfig =>
      isComponentModelOfType(candidate, UPSCALE_COMPONENT_TYPES[field]) &&
      (!slot || slot.accepts(candidate as ModelConfig));

    picked[field] = fits(current) ? current : slot ? (models.find(fits) ?? null) : null;
  }

  return picked;
};

/** The encoders the selected main model reads; every other field reads as null. */
export const activeUpscaleComponents = (values: Pick<UpscaleWidgetValues, 'model' | UpscaleComponentField>) => {
  const active: UpscaleComponentValues = {
    clipEmbedModel: null,
    mistralEncoderModel: null,
    qwen35EncoderModel: null,
    qwen3EncoderModel: null,
    qwen3VlEncoderModel: null,
    qwenVlEncoderModel: null,
    t5EncoderModel: null,
  };

  for (const slot of upscaleComponentsFor(values.model).slots) {
    active[slot.field] = values[slot.field];
  }

  return active;
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
  const base = model?.base;
  const tileControlNet = isUpscaleBase(base) ? UPSCALE_ARCHITECTURES[base].tileControlNet : null;

  return (
    tileControlNet !== null &&
    isTileControlNetModelConfig(value) &&
    value.base === base &&
    tileControlNet.matchesName(value.name)
  );
};

/** Only a required ControlNet is auto-picked; an optional one costs memory and would return after being cleared. */
const pickTileControlNet = (
  stored: unknown,
  models: readonly ModelConfig[],
  model: MainModelConfig | null
): TileControlNetModelConfig | null => {
  if (isTileControlNetCandidate(stored, model)) {
    return stored;
  }

  return upscaleArchitectureFor(model)?.tileControlNet?.required
    ? (models.find((candidate) => isTileControlNetCandidate(candidate, model)) ?? null)
    : null;
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

/** What SD and FLUX.1 run at until the user changes it; SD's multi-diffusion pass is tuned to it. */
const DEFAULT_UPSCALE_CFG_SCALE = 2;

/** See `takesModelGuidance`. Applied once at the change, so the served table must be loaded first. */
const guidanceAfterModelChange = (
  previous: MainModelConfig | null,
  next: MainModelConfig | null,
  current: number
): number => {
  if (next && upscaleArchitectureFor(next)?.takesModelGuidance) {
    return getDefaultGenerateSettings(next).cfgScale;
  }

  return upscaleArchitectureFor(previous)?.takesModelGuidance ? DEFAULT_UPSCALE_CFG_SCALE : current;
};

const DEFAULT_UPSCALE_STEPS = 30;

/** See `takesModelSteps`; leaving FLUX.2 restores 30, so SD does not inherit Klein's 4. */
const stepsAfterModelChange = (
  previous: MainModelConfig | null,
  next: MainModelConfig | null,
  current: number
): number => {
  if (next && upscaleArchitectureFor(next)?.takesModelSteps) {
    return getDefaultGenerateSettings(next).steps;
  }

  return upscaleArchitectureFor(previous)?.takesModelSteps ? DEFAULT_UPSCALE_STEPS : current;
};

export const createDefaultUpscaleWidgetValues = (models: readonly ModelConfig[] = []): UpscaleWidgetValues => {
  const model = (models.find(isSupportedUpscaleMainModel) as MainModelConfig | undefined) ?? null;
  const components = upscaleComponentsFor(model);

  return {
    ...pickComponents({}, models, model),
    batchCount: 1,
    cfgScale: guidanceAfterModelChange(null, model, DEFAULT_UPSCALE_CFG_SCALE),
    clipSkip: 0,
    creativity: 0,
    inputImage: null,
    loras: [],
    mode: 'refine',
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
    steps: stepsAfterModelChange(null, model, DEFAULT_UPSCALE_STEPS),
    structure: 0,
    tileControlnetModel: pickTileControlNet(null, models, model),
    tileOverlap: 128,
    tileSize: 1024,
    upscaleModel: (models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null,
    vae: components.vae ? pickCompatibleVae(null, models, model) : null,
    vaePrecision: 'fp32',
  };
};

/** Anything but Model only, a missing value included, is Refine: projects saved before modes existed refine. */
const normalizeUpscaleMode = (value: unknown): UpscaleMode => (value === 'model-only' ? 'model-only' : 'refine');

/** Heals partial persisted values without clamping user input; validation reports ranges. */
export const normalizeUpscaleWidgetValues = (value: unknown): UpscaleWidgetValues | null => {
  if (!isRecord(value)) {
    return null;
  }

  const defaults = createDefaultUpscaleWidgetValues();

  return {
    batchCount: sanitizeBatchCount(value.batchCount),
    cfgScale: isFiniteNumber(value.cfgScale) ? value.cfgScale : defaults.cfgScale,
    clipEmbedModel: isComponentModelOfType(value.clipEmbedModel, 'clip_embed') ? value.clipEmbedModel : null,
    mistralEncoderModel: isComponentModelOfType(value.mistralEncoderModel, 'mistral_encoder')
      ? value.mistralEncoderModel
      : null,
    qwen35EncoderModel: isComponentModelOfType(value.qwen35EncoderModel, 'qwen3_5_encoder')
      ? value.qwen35EncoderModel
      : null,
    qwen3EncoderModel: isComponentModelOfType(value.qwen3EncoderModel, 'qwen3_encoder')
      ? value.qwen3EncoderModel
      : null,
    qwen3VlEncoderModel: isComponentModelOfType(value.qwen3VlEncoderModel, 'qwen3_vl_encoder')
      ? value.qwen3VlEncoderModel
      : null,
    qwenVlEncoderModel: isComponentModelOfType(value.qwenVlEncoderModel, 'qwen_vl_encoder')
      ? value.qwenVlEncoderModel
      : null,
    clipSkip: isFiniteNumber(value.clipSkip) ? value.clipSkip : defaults.clipSkip,
    creativity: isFiniteNumber(value.creativity) ? value.creativity : defaults.creativity,
    inputImage:
      value.inputImage === null || value.inputImage === undefined
        ? null
        : isUpscaleImage(value.inputImage)
          ? value.inputImage
          : null,
    loras: normalizeLoras(value.loras),
    mode: normalizeUpscaleMode(value.mode),
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
  // A fresh widget, or one whose model was uninstalled, gets the CFG and steps a pick would have given.
  const cfgScale =
    model?.key === values.model?.key ? values.cfgScale : guidanceAfterModelChange(values.model, model, values.cfgScale);
  const steps =
    model?.key === values.model?.key ? values.steps : stepsAfterModelChange(values.model, model, values.steps);
  const storedSpandrel = values.upscaleModel ? modelsByKey.get(values.upscaleModel.key) : undefined;
  const upscaleModel: SpandrelModelConfig | null = isSpandrelModelConfig(storedSpandrel)
    ? storedSpandrel
    : ((models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null);
  const tileControlnetModel = pickTileControlNet(
    values.tileControlnetModel ? modelsByKey.get(values.tileControlnetModel.key) : undefined,
    models,
    model
  );
  const components = pickComponents(
    Object.fromEntries(
      UPSCALE_COMPONENT_FIELDS.map((field) => [field, values[field] ? modelsByKey.get(values[field].key) : undefined])
    ),
    models,
    model
  );
  const storedVae = values.vae ? modelsByKey.get(values.vae.key) : undefined;
  const vae =
    storedVae && isVaeModelConfig(storedVae) && model && isVaeCompatibleWithGenerateModel(model, storedVae)
      ? storedVae
      : upscaleComponentsFor(model).vae
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
    values.cfgScale === cfgScale &&
    values.steps === steps &&
    values.upscaleModel === upscaleModel &&
    values.tileControlnetModel === tileControlnetModel &&
    UPSCALE_COMPONENT_FIELDS.every((field) => values[field] === components[field]) &&
    values.vae === vae &&
    loras.length === values.loras.length &&
    loras.every((lora, index) => lora.model === values.loras[index]?.model)
  ) {
    return values;
  }

  return { ...values, ...components, cfgScale, loras, model, steps, tileControlnetModel, upscaleModel, vae };
};

export const selectUpscaleMainModel = (
  values: UpscaleWidgetValues,
  model: MainModelConfig,
  models: readonly ModelConfig[]
): UpscaleWidgetValues => {
  const synced = syncUpscaleWidgetValuesWithModels({ ...values, model }, models);

  return synced.model?.key === values.model?.key
    ? synced
    : {
        ...synced,
        cfgScale: guidanceAfterModelChange(values.model, synced.model, values.cfgScale),
        steps: stepsAfterModelChange(values.model, synced.model, values.steps),
      };
};

export type UpscaleSection = 'advanced' | 'generation' | 'source';

/** Without a section the form itself is held back (served architecture table not loaded). */
export interface UpscaleValidationIssue {
  message: string;
  section?: UpscaleSection;
}

const addRangeIssue = (
  issues: UpscaleValidationIssue[],
  section: UpscaleSection,
  label: string,
  value: number,
  min: number,
  max: number
): void => {
  if (!Number.isFinite(value) || value < min || value > max) {
    issues.push({ message: `${label} must be between ${min} and ${max}.`, section });
  }
};

/** Model only needs just an image, an installed upscaler and a Scale in range. */
const getModelOnlyValidationIssues = (
  values: UpscaleWidgetValues,
  models?: readonly ModelConfig[]
): UpscaleValidationIssue[] => {
  const issues: UpscaleValidationIssue[] = [];

  if (!values.inputImage) {
    issues.push({ message: 'Upscale needs an input image. Upload one or send one from Gallery.', section: 'source' });
  }
  if (!values.upscaleModel) {
    issues.push({ message: 'Upscale needs a Spandrel image-to-image model.', section: 'source' });
  }
  addRangeIssue(issues, 'source', 'Scale', values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX);

  if (models && values.upscaleModel) {
    const installed = models.find((model) => model.key === values.upscaleModel?.key);

    if (!installed) {
      issues.push({ message: `${values.upscaleModel.name} is no longer installed.`, section: 'source' });
    } else if (!isSpandrelModelConfig(installed)) {
      issues.push({ message: `${values.upscaleModel.name} is not an installed Spandrel model.`, section: 'source' });
    }
  }

  return issues;
};

export const getUpscaleValidationIssues = (
  values: UpscaleWidgetValues,
  models?: readonly ModelConfig[]
): UpscaleValidationIssue[] => {
  if (values.mode === 'model-only') {
    return getModelOnlyValidationIssues(values, models);
  }

  // Without the served table, VAEs read incompatible and schedulers fall back, so other reasons would mislead.
  if (!hasArchitectureCapabilities()) {
    return [
      {
        message:
          'Model capabilities are not available. Upscale is blocked until they load; if this persists, retry from the Upscale panel.',
      },
    ];
  }

  const issues: UpscaleValidationIssue[] = [];

  if (!values.inputImage) {
    issues.push({ message: 'Upscale needs an input image. Upload one or send one from Gallery.', section: 'source' });
  }
  if (!values.upscaleModel) {
    issues.push({ message: 'Upscale needs a Spandrel image-to-image model.', section: 'source' });
  }
  const architecture = upscaleArchitectureFor(values.model);

  if (!values.model) {
    issues.push({
      message: `Upscale needs a ${supportedUpscaleArchitectureLabels()} main model.`,
      section: 'generation',
    });
  } else if (!architecture) {
    issues.push({
      message: `Upscale supports only ${supportedUpscaleArchitectureLabels()} main models.`,
      section: 'generation',
    });
  }

  // `flux_denoise` refuses a Fill model without fill conditioning, which an upscale never supplies.
  if (values.model?.base === 'flux' && values.model.variant === 'dev_fill') {
    issues.push({ message: 'Upscale cannot use a FLUX Fill model.', section: 'generation' });
  }
  // Without a reference, `qwen_image_denoise` gives Edit a zero reference doubling image tokens: ~3x time for nothing.
  if (values.model?.base === 'qwen-image' && values.model.variant === 'edit') {
    issues.push({ message: 'Upscale cannot use a Qwen Image Edit model.', section: 'generation' });
  }

  const tileControlNet = architecture?.tileControlNet ?? null;

  if (tileControlNet?.required && !values.tileControlnetModel) {
    issues.push({
      message: 'Upscale needs a Tile or Union ControlNet compatible with the main model.',
      section: 'advanced',
    });
  } else if (
    tileControlNet &&
    values.tileControlnetModel &&
    !isTileControlNetCandidate(values.tileControlnetModel, values.model)
  ) {
    issues.push({
      message: tileControlNet.required
        ? 'The Tile ControlNet must match the main model base and be a Tile or Union model.'
        : 'The Tile ControlNet must match the main model base and be a Tile model.',
      section: 'advanced',
    });
  }

  // A bare transformer cannot supply these itself, and its loader refuses to run without them.
  const components = upscaleComponentsFor(values.model);
  const activeComponents = activeUpscaleComponents(values);

  for (const slot of components.slots) {
    if (!values[slot.field]) {
      issues.push({ message: `Upscale needs a ${slot.label} for this main model.`, section: 'advanced' });
    } else if (!slot.accepts(values[slot.field] as ModelConfig)) {
      issues.push({
        message: `${values[slot.field]?.name} is not a ${slot.label} this main model can use.`,
        section: 'advanced',
      });
    }
  }
  if (components.vae && !values.vae) {
    issues.push({ message: 'Upscale needs a VAE for this main model.', section: 'advanced' });
  }

  addRangeIssue(issues, 'source', 'Scale', values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX);
  addRangeIssue(issues, 'source', 'Creativity', values.creativity, UPSCALE_CREATIVITY_MIN, UPSCALE_CREATIVITY_MAX);
  addRangeIssue(issues, 'source', 'Structure', values.structure, UPSCALE_STRUCTURE_MIN, UPSCALE_STRUCTURE_MAX);
  addRangeIssue(issues, 'advanced', 'Tile size', values.tileSize, UPSCALE_TILE_SIZE_MIN, UPSCALE_TILE_SIZE_MAX);
  addRangeIssue(
    issues,
    'advanced',
    'Tile overlap',
    values.tileOverlap,
    UPSCALE_TILE_OVERLAP_MIN,
    UPSCALE_TILE_OVERLAP_MAX
  );
  addRangeIssue(issues, 'generation', 'Steps', values.steps, upscaleMinSteps(values.model), 1000);
  addRangeIssue(issues, 'generation', 'CFG scale', values.cfgScale, 1, upscaleMaxCfgScale(values.model));
  addRangeIssue(issues, 'generation', 'Seed', values.seed, 0, SEED_MAX);

  if (models) {
    const installedByKey = new Map(models.map((model) => [model.key, model]));
    const required: (readonly [ModelIdentifierConfig, UpscaleSection])[] = [
      ...(values.model ? [[values.model, 'generation'] as const] : []),
      ...(values.upscaleModel ? [[values.upscaleModel, 'source'] as const] : []),
      ...(values.tileControlnetModel ? [[values.tileControlnetModel, 'advanced'] as const] : []),
      // A deleted encoder must block the run rather than fail mid-graph.
      ...UPSCALE_COMPONENT_FIELDS.flatMap((field) =>
        activeComponents[field] ? [[activeComponents[field], 'advanced'] as const] : []
      ),
      ...(values.vae ? [[values.vae, 'advanced'] as const] : []),
    ];

    for (const [selected, section] of required) {
      const installed = installedByKey.get(selected.key);

      if (!installed) {
        issues.push({ message: `${selected.name} is no longer installed.`, section });
      }
    }

    const installedMain = values.model ? installedByKey.get(values.model.key) : undefined;
    const installedSpandrel = values.upscaleModel ? installedByKey.get(values.upscaleModel.key) : undefined;
    const installedControlNet = values.tileControlnetModel
      ? installedByKey.get(values.tileControlnetModel.key)
      : undefined;

    if (values.model && installedMain && !isSupportedUpscaleMainModel(installedMain)) {
      issues.push({
        message: `${values.model.name} is not an installed ${supportedUpscaleArchitectureLabels()} main model.`,
        section: 'generation',
      });
    }
    if (values.upscaleModel && installedSpandrel && !isSpandrelModelConfig(installedSpandrel)) {
      issues.push({ message: `${values.upscaleModel.name} is not an installed Spandrel model.`, section: 'source' });
    }
    if (
      values.tileControlnetModel &&
      installedControlNet &&
      !isTileControlNetCandidate(installedControlNet, values.model)
    ) {
      issues.push({
        message: `${values.tileControlnetModel.name} is not an installed compatible Tile ControlNet.`,
        section: 'advanced',
      });
    }
    if (values.vae) {
      const installedVae = installedByKey.get(values.vae.key);

      if (!installedVae) {
        issues.push({ message: `${values.vae.name} is no longer installed.`, section: 'advanced' });
      } else if (
        !values.model ||
        !isVaeModelConfig(installedVae) ||
        !isVaeCompatibleWithGenerateModel(values.model, installedVae)
      ) {
        issues.push({
          message: `${values.vae.name} is not compatible with ${values.model?.name ?? 'the main model'}.`,
          section: 'advanced',
        });
      }
    }
    for (const lora of values.loras) {
      const installedLora = installedByKey.get(lora.model.key);

      if (!installedLora) {
        issues.push({ message: `${lora.model.name} is no longer installed.`, section: 'generation' });
      } else if (
        !isLoraModelConfig(installedLora) ||
        (values.model && !isLoraCompatibleWithModel(installedLora, values.model))
      ) {
        issues.push({
          message: `${lora.model.name} is not compatible with ${values.model?.name ?? 'the main model'}.`,
          section: 'generation',
        });
      }
    }
  }

  return issues;
};

export const getUpscaleValidationReasons = (values: UpscaleWidgetValues, models?: readonly ModelConfig[]): string[] =>
  getUpscaleValidationIssues(values, models).map((issue) => issue.message);

/** The upscaler's recorded native scale, or null for a record installed before the probe recorded it. */
export const getSpandrelNativeScale = (model: SpandrelModelConfig | null | undefined): number | null =>
  typeof model?.scale === 'number' && Number.isInteger(model.scale) && model.scale >= 1 ? model.scale : null;

/** `spandrel_image_to_image_autoscale` runs until at least the target, then resizes; a 1x model runs once. */
export type ModelOnlyUpscalePlan =
  | { kind: 'no-enlarge' }
  | { kind: 'passes'; passes: number; /** The image is resized after the last pass. */ resizes: boolean };

export const getModelOnlyUpscalePlan = (nativeScale: number | null, scale: number): ModelOnlyUpscalePlan | null => {
  if (nativeScale === null || !Number.isFinite(scale) || scale <= 0) {
    return null;
  }
  if (nativeScale === 1) {
    return { kind: 'no-enlarge' };
  }

  // The node compares sizes, not factors; multiplying keeps 4x reaching 16 in exactly two passes.
  let passes = 1;
  let reached = nativeScale;

  while (reached < scale) {
    passes += 1;
    reached *= nativeScale;
  }

  return { kind: 'passes', passes, resizes: reached !== scale };
};

const MODEL_ONLY_PRESET_FACTORS = [1.5, 2];

/** Smallest first, only factors the plan reaches; null while the native scale is unknown. */
export const getModelOnlyUpscaleFactors = (model: SpandrelModelConfig): number[] | null => {
  const nativeScale = getSpandrelNativeScale(model);

  if (nativeScale === null) {
    return null;
  }

  const presets = MODEL_ONLY_PRESET_FACTORS.filter(
    (factor) => factor !== nativeScale && getModelOnlyUpscalePlan(nativeScale, factor)?.kind === 'passes'
  );

  return [...presets, nativeScale].sort((a, b) => a - b);
};

/** Refine: Spandrel's frame floored to the pass grid. Model only: floored to 8, or the input size for a 1x model. */
export const getUpscaleOutputDimensions = (
  image: Pick<ImageWithDims, 'width' | 'height'>,
  values: Pick<UpscaleWidgetValues, 'mode' | 'model' | 'scale' | 'upscaleModel'>
): { width: number; height: number } => {
  if (values.mode === 'model-only') {
    const scale = getSpandrelNativeScale(values.upscaleModel) === 1 ? 1 : values.scale;

    return {
      height: spandrelAutoscaleDimension(image.height, scale),
      width: spandrelAutoscaleDimension(image.width, scale),
    };
  }

  const grid = upscaleArchitectureFor(values.model)?.denoiseGrid ?? 8;

  return {
    height: upscaleDenoiseDimension(image.height, values.scale, grid),
    width: upscaleDenoiseDimension(image.width, values.scale, grid),
  };
};

/** Refine reads the prompt Upscale shares with Generate; Model only reads none, so nothing wraps or expands it. */
export const readsUpscalePrompt = (values: Record<string, unknown>): boolean =>
  normalizeUpscaleMode(values.mode) === 'refine';

/** Model only runs the upscaler alone, which gives the same image every time, so iterations would only repeat it. */
export const isUpscaleBatchCountFixed = (values: Record<string, unknown>): boolean =>
  normalizeUpscaleMode(values.mode) === 'model-only';

/** The metadata read must be of the current input, or the result would carry another image's metadata. */
export const isUpscaleSourceMetadataCurrent = (values: UpscaleWidgetValues, sourceImageName: string | undefined) =>
  values.mode !== 'model-only' || (values.inputImage !== null && sourceImageName === values.inputImage.image_name);

/** Null when Model only's metadata was read for another image. Model only carries no prompt, a fixed seed, one run. */
export const resolveUpscaleSubmissionValues = (
  values: UpscaleWidgetValues,
  {
    promptDraft,
    sourceImageName,
  }: {
    promptDraft: Pick<ProjectPromptDraft, 'negativePrompt' | 'negativePromptEnabled' | 'positivePrompt'>;
    sourceImageName: string | undefined;
  }
): UpscaleWidgetValues | null => {
  if (!isUpscaleSourceMetadataCurrent(values, sourceImageName)) {
    return null;
  }

  if (values.mode === 'model-only') {
    return { ...values, batchCount: 1, negativePrompt: '', positivePrompt: '', seedMode: 'fixed' };
  }

  const { negativePrompt, negativePromptEnabled, positivePrompt } = promptDraft;

  return { ...values, negativePrompt, negativePromptEnabled, positivePrompt };
};

/** Seeds and iterations from the run; template and dynamic prompts from the shared prompt, as Generate does. */
export const getUpscaleSubmissionSource = (
  values: UpscaleWidgetValues,
  promptDraft: ProjectPromptDraft
): GenerateSubmissionSource => ({
  batchCount: values.batchCount,
  negativePrompt: values.negativePrompt,
  negativePromptEnabled: values.negativePromptEnabled,
  positivePrompt: values.positivePrompt,
  seed: values.seed,
  seedMode: values.seedMode,
  ...(values.mode === 'model-only'
    ? { dynamicPromptsEnabled: false, promptTemplate: null }
    : {
        dynamicPromptsEnabled: promptDraft.dynamicPromptsEnabled,
        dynamicPromptsSeedBehaviour: promptDraft.dynamicPromptsSeedBehaviour,
        promptTemplate: promptDraft.promptTemplate,
      }),
});

export const clearDeletedUpscaleInput = (
  values: UpscaleWidgetValues,
  deletedImageNames: ReadonlySet<string>
): UpscaleWidgetValues =>
  values.inputImage && deletedImageNames.has(values.inputImage.image_name) ? { ...values, inputImage: null } : values;

export const resolveUpscaleSeed = (values: UpscaleWidgetValues): number =>
  values.seedMode === 'random' ? Math.floor(Math.random() * SEED_MAX) : values.seed;

export const cloneUpscaleWidgetValues = (values: UpscaleWidgetValues): UpscaleWidgetValues => structuredClone(values);
