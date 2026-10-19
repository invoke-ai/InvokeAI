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
  coerceSchedulerForGraph,
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  getDefaultGenerateSettings,
  getSchedulerOptions,
  hasArchitectureCapabilities,
  isAnimaQwen35Encoder,
  isAnimaQwen3Encoder,
  isBundledMainForBase,
  isKrea2Qwen3VlEncoder,
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

// The schedule shifts of the denoise nodes that clip `denoising_start` by step index, for the table's
// `indexClipping`; `getIndexClippedDenoisingStart` in graph.ts uses them to start those nodes where
// FLUX would. They go away when those nodes clip by value.

/** `z_image_denoise._calculate_shift`, held at the fit's 4096-token end since the clamp (#378). */
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

/**
 * `krea2/sampling_utils.py`: Turbo's fixed mu, and Raw's fit by token count. A Diffusers install samples
 * with its own `scheduler/scheduler_config.json`; Krea's ships these same numbers, and one that did not
 * would move Raw's starts below 1280x1280 without the backend contract noticing.
 */
const KREA2_DISTILLED_MU = 1.15;
const KREA2_SHIFT_MIN_SEQ_LEN = 256;
const KREA2_SHIFT_MAX_SEQ_LEN = 6400;
const KREA2_BASE_SHIFT = 0.5;
const KREA2_MAX_SHIFT = 1.15;

/**
 * Krea-2's linear shift, `exp(mu)`: fixed for Turbo, and for Raw (`krea2_base`) its fit at one token per
 * 16x16 pixels, held at the fit's 6400-token end as `krea2_denoise` holds it. Like the node, anything that
 * is not Raw counts as distilled.
 */
const krea2ScheduleShift = (model: MainModelConfig, { width, height }: { width: number; height: number }): number => {
  if (model.variant !== 'krea2_base') {
    return Math.exp(KREA2_DISTILLED_MU);
  }

  const seqLen = Math.min(Math.floor(width / 16) * Math.floor(height / 16), KREA2_SHIFT_MAX_SEQ_LEN);
  const slope = (KREA2_MAX_SHIFT - KREA2_BASE_SHIFT) / (KREA2_SHIFT_MAX_SEQ_LEN - KREA2_SHIFT_MIN_SEQ_LEN);

  return Math.exp(seqLen * slope + KREA2_BASE_SHIFT - slope * KREA2_SHIFT_MIN_SEQ_LEN);
};

/** A text encoder the widget can be asked for; each is its own persisted field. */
export type UpscaleComponentField =
  | 't5EncoderModel'
  | 'clipEmbedModel'
  | 'qwen3EncoderModel'
  | 'qwen35EncoderModel'
  | 'qwen3VlEncoderModel';

const UPSCALE_COMPONENT_TYPES: Record<UpscaleComponentField, string> = {
  clipEmbedModel: 'clip_embed',
  qwen35EncoderModel: 'qwen3_5_encoder',
  qwen3EncoderModel: 'qwen3_encoder',
  qwen3VlEncoderModel: 'qwen3_vl_encoder',
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
// FLUX.1 encodes with CLIP-L; a CLIP-G, which an installed ComfyUI text_encoders folder brings along, fails
// its text encoder.
const CLIP_EMBED_SLOT: UpscaleComponentSlot = {
  accepts: (candidate) => candidate.type === 'clip_embed' && candidate.variant === 'large',
  field: 'clipEmbedModel',
  label: 'CLIP Embed model',
};
// Z-Image and Anima both read a `qwen3_encoder`, but not the same one. Z-Image's caption embedder is
// 2560 wide, so it takes the 4B and nothing else: Generate's shared filter also lets FLUX.2 Klein's
// 8B through, which fails in the denoise -- here, after the upscale has run, and it would be the
// automatic pick wherever its name sorts first. Anima's is the 0.6B.
const Z_IMAGE_QWEN3_SLOT: UpscaleComponentSlot = {
  accepts: (candidate) => candidate.type === 'qwen3_encoder' && candidate.variant === 'qwen3_4b',
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
// Krea-2 reads the 4B; Ideogram 4's 8B is the same model type and `krea2_model_loader` refuses it.
const KREA2_QWEN3_VL_SLOT: UpscaleComponentSlot = {
  accepts: isKrea2Qwen3VlEncoder,
  field: 'qwen3VlEncoderModel',
  label: 'Qwen3-VL 4B encoder',
};

/** What a main model's loader must be handed. A model that bundles its own parts needs none of it. */
export interface UpscaleComponents {
  readonly slots: readonly UpscaleComponentSlot[];
  readonly vae: boolean;
}

const NO_COMPONENTS: UpscaleComponents = { slots: [], vae: false };

export interface UpscaleTileControlNet {
  /**
   * SD's multi-diffusion denoises tiles independently, and without a tile ControlNet they drift
   * apart. Elsewhere the pass runs whole and the ControlNet is an extra anchor it can do without.
   */
  readonly required: boolean;
  /**
   * Nothing in a checkpoint says which mode it was trained for -- Z-Image's Tile and Union 2.1 are
   * tensor-for-tensor identical -- so the name decides. Union models tile on SD and SDXL; Z-Image's
   * Union does Canny, HED, Depth, Pose, MLSD and inpainting, but not tile.
   */
  readonly matchesName: (name: string) => boolean;
}

const nameIncludes =
  (...needles: string[]) =>
  (name: string): boolean =>
    needles.some((needle) => name.toLowerCase().includes(needle));

/** How a denoise node that clips `denoising_start` by step index lands on its schedule. */
export interface UpscaleIndexClipping {
  /** The linear schedule shift the node runs this main model at, for the frame it denoises. */
  readonly shift: (model: MainModelConfig, size: { width: number; height: number }) => number;
  /**
   * Whether the node keeps `int(start * steps)` or Python's `int(round(start * steps))`, which rounds a
   * half to even and so cannot be aimed at the middle of a step.
   */
  readonly rounding: 'floor' | 'round';
}

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
   * A tile ControlNet anchors the second pass to the upscaled frame, and the Structure slider drives
   * its weights. Where the architecture has none, Structure has nothing to scale and is hidden rather
   * than left inert.
   */
  readonly tileControlNet: UpscaleTileControlNet | null;
  /**
   * The parts the loader refuses to run without. Bare transformer checkpoints are the common case
   * for every architecture past SDXL; only a bundled pipeline (Diffusers, complete SDNQ) carries its
   * own text encoder and VAE.
   */
  readonly components: (model: MainModelConfig) => UpscaleComponents;
  /**
   * The size granularity the architecture's denoise node demands. Spandrel's autoscale rounds to a
   * multiple of 8, which `flux_denoise` (multiple_of=16) rejects outright — after the upscale has
   * already run. Where this is coarser than 8 the compiler resizes the frame down to it and passes
   * the size as a literal, so the graph cannot be built with a size the backend will refuse.
   */
  readonly denoiseGrid: number;
  /**
   * The served `FeaturesFacet` answer, pinned to it by test. Guidance-distilled architectures ignore
   * a negative prompt, so showing one invites a silent no-op; `cfg-gated` ones read it only above
   * CFG 1, which is the only case the graph wires it.
   */
  readonly negativePrompt: 'always' | 'cfg-gated' | 'never';
  /**
   * How a denoise node that clips `denoising_start` by step index rather than by value does it. The
   * compiler uses it to start such a node where FLUX would start for the same Creativity; null where
   * the node clips by value or runs its own schedule.
   */
  readonly indexClipping: UpscaleIndexClipping | null;
  /**
   * The schedulers that start where Creativity puts the pass; null where all of the base's do. The
   * img2img blend noises the frame to the clipped schedule's first sigma, but Heun and LCM on the
   * flow models build their own schedule from pure noise and denoise the frame as if it were noise.
   * Anima's Heun clips its own schedule, which runs from 1 to nearly 0, so it starts below the blend
   * and leaves noise behind. `test_upscale_index_clipped_starts.py` checks the index-clipped lists.
   */
  readonly schedulers: readonly string[] | null;
  /**
   * Whether a new main model sets CFG to that model's served default. Z-Image Turbo and Krea-2 Turbo are
   * distilled for CFG 1, and anything above runs a second, unconditional pass per step at upscale
   * resolution; Z-Image Base, Anima and Krea-2 Raw want 4 to 6. SD and FLUX keep the user's CFG -- SD's
   * multi-diffusion pass is tuned to the widget's 2 -- and get that 2 back when the user comes from one
   * of the others.
   */
  readonly takesModelGuidance: boolean;
  /** Shown in validation messages, so a user reads "FLUX.1" rather than "flux". */
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
    tileControlNet: SD_TILE_CONTROLNET,
  },
  flux: {
    // `flux_model_loader` refuses a bare transformer without T5, CLIP Embed *and* VAE. The backend
    // also checks a self-contained SDNQ pipeline really has those submodels, which the model list
    // does not expose; one missing a submodel passes here and the loader refuses it precisely.
    components: (model) =>
      model.format === 'sdnq_quantized' ? NO_COMPONENTS : { slots: [T5_ENCODER_SLOT, CLIP_EMBED_SLOT], vae: true },
    denoiseGrid: 16,
    indexClipping: null,
    label: 'FLUX.1',
    negativePrompt: 'never',
    // `flux/denoise.py` hands Heun and LCM to the same diffusers schedulers as `z_image_denoise`, so
    // when the backend contract finds Z-Image's Heun starting at the blend, this list is due as well.
    schedulers: ['euler'],
    takesModelGuidance: false,
    tileControlNet: null,
  },
  'z-image': {
    // `z_image_model_loader` falls back to a bundled main model's own VAE and Qwen3.
    components: (model) =>
      isBundledMainForBase('z-image')(model) ? NO_COMPONENTS : { slots: [Z_IMAGE_QWEN3_SLOT], vae: true },
    denoiseGrid: 16,
    indexClipping: { rounding: 'floor', shift: (_model, size) => zImageScheduleShift(size) },
    label: 'Z-Image',
    negativePrompt: 'cfg-gated',
    schedulers: ['euler'],
    takesModelGuidance: true,
    tileControlNet: { matchesName: nameIncludes('tile'), required: false },
  },
  anima: {
    // Anima-3.8B's semantic connector also reads Qwen3.5; no other Anima model does.
    components: (model) => ({
      slots: model.variant === 'anima_qwen35' ? [ANIMA_QWEN3_SLOT, ANIMA_QWEN35_SLOT] : [ANIMA_QWEN3_SLOT],
      vae: true,
    }),
    denoiseGrid: 8,
    indexClipping: { rounding: 'floor', shift: () => ANIMA_SCHEDULE_SHIFT },
    label: 'Anima',
    negativePrompt: 'cfg-gated',
    schedulers: ['euler', 'dpmpp_2m', 'dpmpp_2m_sde', 'er_sde'],
    takesModelGuidance: true,
    tileControlNet: null,
  },
  'krea-2': {
    // Only a Diffusers install carries its Qwen3-VL encoder and VAE; `krea2_model_loader` refuses a
    // single-file or GGUF transformer without both.
    components: (model) => (model.format === 'diffusers' ? NO_COMPONENTS : { slots: [KREA2_QWEN3_VL_SLOT], vae: true }),
    denoiseGrid: 16,
    indexClipping: { rounding: 'round', shift: krea2ScheduleShift },
    label: 'Krea-2',
    negativePrompt: 'cfg-gated',
    // `krea2_denoise` has no scheduler input: it steps with Euler.
    schedulers: ['euler'],
    takesModelGuidance: true,
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

/** The parts the selected main model must be given explicitly; none where it bundles them or is unsupported. */
export const upscaleComponentsFor = (model: MainModelConfig | null): UpscaleComponents => {
  const architecture = upscaleArchitectureFor(model);

  return model && architecture ? architecture.components(model) : NO_COMPONENTS;
};

/**
 * Whether the graph has a negative branch. The CFG-gated architectures run an unconditional pass only
 * above CFG 1, and need a negative conditioning to run it on. As in Generate, a switched-off negative
 * prompt still gets its branch -- the queue fills it with an empty string -- so CFG runs at the value
 * the slider and the image's metadata show.
 */
export const wiresUpscaleNegativePrompt = (model: MainModelConfig | null, cfgScale: number): boolean => {
  const usage = upscaleArchitectureFor(model)?.negativePrompt;

  return usage === 'always' || (usage === 'cfg-gated' && cfgScale > 1);
};

/**
 * Index clipping starts the pass on a whole step of the schedule, so a single step leaves nothing between
 * starting from pure noise and not denoising at all; the pass needs at least two.
 */
export const upscaleMinSteps = (model: MainModelConfig | null): number =>
  upscaleArchitectureFor(model)?.indexClipping ? 2 : 1;

/** The base's schedulers, less those that would ignore Creativity; see `schedulers`. */
export const upscaleSchedulerOptions = (model: MainModelConfig | null) => {
  const options = getSchedulerOptions(model ?? undefined);
  const allowed = upscaleArchitectureFor(model)?.schedulers;

  return allowed ? options.filter((option) => allowed.includes(option.value)) : options;
};

/**
 * The scheduler the pass runs with. The stored value is left alone, so a switch back to SD brings back
 * the SD scheduler the flow models do not have; until then the graph and the metadata record the one
 * that runs, and so does the field where there is more than one to choose.
 */
export const coerceUpscaleScheduler = (model: MainModelConfig | null, scheduler: string): string => {
  const coerced = coerceSchedulerForGraph(model ?? undefined, scheduler);
  const allowed = upscaleArchitectureFor(model)?.schedulers;

  return !allowed || allowed.includes(coerced) ? coerced : (allowed[0] ?? coerced);
};

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

type UpscaleComponentValues = Record<UpscaleComponentField, ComponentModelConfig | null>;

/**
 * For each slot the architecture has: the stored encoder if the slot still accepts it, otherwise the
 * first installed one that fits. A field it has no slot for keeps its stored encoder while that is
 * still installed, so switching to another architecture and back does not lose a hand-picked one;
 * `activeUpscaleComponents` keeps it out of the graph and the metadata meanwhile.
 */
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
    qwen35EncoderModel: null,
    qwen3EncoderModel: null,
    qwen3VlEncoderModel: null,
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

/**
 * Keep a stored ControlNet that still fits. Only a required one is picked automatically: an optional
 * one costs a second network's memory and time, and auto-picking it here would also re-add it every
 * time the user cleared it.
 */
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

/** What SD and FLUX run at until the user changes it; SD's multi-diffusion pass is tuned to it. */
const DEFAULT_UPSCALE_CFG_SCALE = 2;

/**
 * The CFG a change of main model leaves, see `takesModelGuidance`. Leaving a model that took its
 * served guidance for one that keeps the user's restores the widget's default, so SD's tuned pass
 * does not inherit Turbo's 1, nor FLUX Anima-3.8B's 6.
 *
 * The default is applied once, at the change, so the change must not happen before the served table
 * has loaded: the widget is gated on it, and validation refuses to run without it.
 */
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
    tileControlnetModel: pickTileControlNet(null, models, model),
    tileOverlap: 128,
    tileSize: 1024,
    upscaleModel: (models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null,
    vae: components.vae ? pickCompatibleVae(null, models, model) : null,
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
    qwen35EncoderModel: isComponentModelOfType(value.qwen35EncoderModel, 'qwen3_5_encoder')
      ? value.qwen35EncoderModel
      : null,
    qwen3EncoderModel: isComponentModelOfType(value.qwen3EncoderModel, 'qwen3_encoder')
      ? value.qwen3EncoderModel
      : null,
    qwen3VlEncoderModel: isComponentModelOfType(value.qwen3VlEncoderModel, 'qwen3_vl_encoder')
      ? value.qwen3VlEncoderModel
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
  // A fresh widget, or one whose model was uninstalled, gets its model here rather than from a pick,
  // and so the CFG a pick would have given it.
  const cfgScale =
    model?.key === values.model?.key ? values.cfgScale : guidanceAfterModelChange(values.model, model, values.cfgScale);
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
    values.upscaleModel === upscaleModel &&
    values.tileControlnetModel === tileControlnetModel &&
    UPSCALE_COMPONENT_FIELDS.every((field) => values[field] === components[field]) &&
    values.vae === vae &&
    loras.length === values.loras.length &&
    loras.every((lora, index) => lora.model === values.loras[index]?.model)
  ) {
    return values;
  }

  return { ...values, ...components, cfgScale, loras, model, tileControlnetModel, upscaleModel, vae };
};

/** A main model the user picked: everything synced to it, and the CFG it should run at. */
export const selectUpscaleMainModel = (
  values: UpscaleWidgetValues,
  model: MainModelConfig,
  models: readonly ModelConfig[]
): UpscaleWidgetValues => {
  const synced = syncUpscaleWidgetValuesWithModels({ ...values, model }, models);

  return synced.model?.key === values.model?.key
    ? synced
    : { ...synced, cfgScale: guidanceAfterModelChange(values.model, synced.model, values.cfgScale) };
};

const addRangeReason = (reasons: string[], label: string, value: number, min: number, max: number): void => {
  if (!Number.isFinite(value) || value < min || value > max) {
    reasons.push(`${label} must be between ${min} and ${max}.`);
  }
};

export const getUpscaleValidationReasons = (values: UpscaleWidgetValues, models?: readonly ModelConfig[]): string[] => {
  // As in Generate: without the served table every VAE reads as incompatible and the schedulers fall
  // back, so the other reasons would mislead and the graph would be wrong.
  if (!hasArchitectureCapabilities()) {
    return [
      'Model capabilities are not available. Upscale is blocked until they load; if this persists, retry from the Upscale panel.',
    ];
  }

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

  const tileControlNet = architecture?.tileControlNet ?? null;

  if (tileControlNet?.required && !values.tileControlnetModel) {
    reasons.push('Upscale needs a Tile or Union ControlNet compatible with the main model.');
  } else if (
    tileControlNet &&
    values.tileControlnetModel &&
    !isTileControlNetCandidate(values.tileControlnetModel, values.model)
  ) {
    reasons.push(
      tileControlNet.required
        ? 'The Tile ControlNet must match the main model base and be a Tile or Union model.'
        : 'The Tile ControlNet must match the main model base and be a Tile model.'
    );
  }

  // A bare transformer cannot supply these itself, and its loader refuses to run without them.
  const components = upscaleComponentsFor(values.model);
  const activeComponents = activeUpscaleComponents(values);

  for (const slot of components.slots) {
    if (!values[slot.field]) {
      reasons.push(`Upscale needs a ${slot.label} for this main model.`);
    } else if (!slot.accepts(values[slot.field] as ModelConfig)) {
      reasons.push(`${values[slot.field]?.name} is not a ${slot.label} this main model can use.`);
    }
  }
  if (components.vae && !values.vae) {
    reasons.push('Upscale needs a VAE for this main model.');
  }

  addRangeReason(reasons, 'Scale', values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX);
  addRangeReason(reasons, 'Creativity', values.creativity, UPSCALE_CREATIVITY_MIN, UPSCALE_CREATIVITY_MAX);
  addRangeReason(reasons, 'Structure', values.structure, UPSCALE_STRUCTURE_MIN, UPSCALE_STRUCTURE_MAX);
  addRangeReason(reasons, 'Tile size', values.tileSize, UPSCALE_TILE_SIZE_MIN, UPSCALE_TILE_SIZE_MAX);
  addRangeReason(reasons, 'Tile overlap', values.tileOverlap, UPSCALE_TILE_OVERLAP_MIN, UPSCALE_TILE_OVERLAP_MAX);
  addRangeReason(reasons, 'Steps', values.steps, upscaleMinSteps(values.model), 1000);
  addRangeReason(reasons, 'CFG scale', values.cfgScale, 1, 100);
  addRangeReason(reasons, 'Seed', values.seed, 0, SEED_MAX);

  if (models) {
    const installedByKey = new Map(models.map((model) => [model.key, model]));
    const required: ModelIdentifierConfig[] = [
      ...(values.model ? [values.model] : []),
      ...(values.upscaleModel ? [values.upscaleModel] : []),
      ...(values.tileControlnetModel ? [values.tileControlnetModel] : []),
      // A deleted encoder must block the run rather than fail mid-graph.
      ...UPSCALE_COMPONENT_FIELDS.flatMap((field) => (activeComponents[field] ? [activeComponents[field]] : [])),
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
      reasons.push(`${values.model.name} is not an installed ${supportedUpscaleArchitectureLabels()} main model.`);
    }
    if (values.upscaleModel && installedSpandrel && !isSpandrelModelConfig(installedSpandrel)) {
      reasons.push(`${values.upscaleModel.name} is not an installed Spandrel model.`);
    }
    if (
      values.tileControlnetModel &&
      installedControlNet &&
      !isTileControlNetCandidate(installedControlNet, values.model)
    ) {
      reasons.push(`${values.tileControlnetModel.name} is not an installed compatible Tile ControlNet.`);
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

/**
 * The size the result comes out at: Spandrel's scaled frame, floored to the grid of a pass that
 * resizes it. One exception cannot be seen from here: a 1x restoration model leaves the frame at the
 * source size, and SD's pass keeps that size. Spandrel's model configs do not carry the scale.
 */
export const getUpscaleOutputDimensions = (
  image: Pick<ImageWithDims, 'width' | 'height'>,
  scale: number,
  model: MainModelConfig | null
): { width: number; height: number } => {
  const grid = upscaleArchitectureFor(model)?.denoiseGrid ?? 8;

  return {
    height: upscaleDenoiseDimension(image.height, scale, grid),
    width: upscaleDenoiseDimension(image.width, scale, grid),
  };
};

export const clearDeletedUpscaleInput = (
  values: UpscaleWidgetValues,
  deletedImageNames: ReadonlySet<string>
): UpscaleWidgetValues =>
  values.inputImage && deletedImageNames.has(values.inputImage.image_name) ? { ...values, inputImage: null } : values;

export const resolveUpscaleSeed = (values: UpscaleWidgetValues): number =>
  values.seedMode === 'random' ? Math.floor(Math.random() * SEED_MAX) : values.seed;

export const cloneUpscaleWidgetValues = (values: UpscaleWidgetValues): UpscaleWidgetValues => structuredClone(values);
